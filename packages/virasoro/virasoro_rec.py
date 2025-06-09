import torch
from numba import cuda,types,njit
import math
import numpy as np
import torch.cuda.nvtx as nvtx  
import os



@cuda.jit(device=True,inline=True)
def denom(m, n,b):
        prod = 1 + 0j                      # complex unity
        for k in range(-m + 1, m + 1):
            for l in range(-n + 1, n + 1):
                if (k == 0 and l == 0) or (k == m and l == n):
                    continue
                prod *= 0.5 * (k / b + l * b)
        return prod


@cuda.jit(device=True,inline=True)
def lam_pq(p,q,b):
    return 0.5 * (p / b + q * b)

@cuda.jit(device=True,inline=True)
def Ppq(b,lam_L2,lam_H2,p,q):
    return (lam_pq(p, q,b) ** 2 - 4 * lam_L2) * \
                       (lam_pq(p, q,b) ** 2 - 4 * lam_H2) * \
                       lam_pq(p, q,b) ** 4

@cuda.jit(device=True,inline=True)
def hmn(b,m,n):
    return  0.25 * (b + 1 / b) ** 2 - lam_pq(m, n,b) ** 2
 
@cuda.jit
def calc_Rmn_kernel(NN,lam_L2_arr, lam_H2_arr, b_arr, R):
    Nth = b_arr.size
    
    i= cuda.grid(1)
    if i >= Nth:
        return
    b = b_arr[i]
    lam_L2 = lam_L2_arr[i]
    lam_H2 = lam_H2_arr[i]

    R[i, 1, 2] = 2 * Ppq(b, lam_L2, lam_H2, 0, 1) / denom(1, 2, b)
    R[i, 2, 1] = 2 * Ppq(b, lam_L2, lam_H2, 1, 0) / denom(2, 1, b)
    R[i, 2, 2] = 2 * Ppq(b, lam_L2, lam_H2, 1, -1) * Ppq(b, lam_L2, lam_H2, 1, 1) / denom(2, 2, b)

    for m in range(4, NN + 1, 2):
        fac = Ppq(b, lam_L2, lam_H2, m - 1, 0) * lam_pq(m, 1, b) / lam_pq(m - 2, 1, b)
        for nn in (0, 1):
            for k in (-m + 1, -m + 2, m - 1, m):
                fac *= 2 / (k / b + nn * b)
        R[i, m, 1] = R[i, m - 2, 1] * fac

    for m in range(3, NN + 1):
        fac = Ppq(b, lam_L2, lam_H2, m - 1, -1) * Ppq(b, lam_L2, lam_H2, m - 1, 1) * lam_pq(m, 2, b) / lam_pq(m - 2, 2, b)
        for nn in (-1, 0, 1, 2):
            for k in (-m + 1, -m + 2, m - 1, m):
                fac *= 2 / (k / b + nn * b)
        R[i, m, 2] = R[i, m - 2, 2] * fac

    for m in range(1, NN + 1):
        for n in range(3, NN // m + 1):
            if (m * n) & 1:
                continue
            fac = lam_pq(m, n, b) / lam_pq(m, n - 2, b)
            for p in range(-m + 1, m, 2):
                fac *= Ppq(b, lam_L2, lam_H2, p, n - 1)
            for nn in (-n + 1, -n + 2, n - 1, n):
                for k in range(-m + 1, m + 1):
                    fac *= 2 / (k / b + nn * b)
            R[i, m, n] = R[i, m, n - 2] * fac

base_dir = os.path.dirname(os.path.abspath(__file__))

factors_path = os.path.join(base_dir, "factors.npy")
counts_path  = os.path.join(base_dir, "counts.npy")

FACTORS_CONST = np.load(factors_path)
COUNTS_CONST  = np.load(counts_path)

@cuda.jit(device=True,inline=False)
def get_mn(k):
    factors = cuda.const.array_like(FACTORS_CONST)
    counts  = cuda.const.array_like(COUNTS_CONST)
    return counts[k//2-1], factors[k//2-1]

@cuda.jit(device=True,inline=False)
def get_mn_all(k):
    factors = cuda.const.array_like(FACTORS_CONST)
    counts  = cuda.const.array_like(COUNTS_CONST)
    return counts[:k//2], factors[:k//2]

# ──────────────────────────────────────────────────────────────────────

@cuda.jit(fastmath=True)
def calc_Hmn_kernel_k(b_arr,NN, R,Hmn,k):

    Nth = R.shape[0]
    n,m,th=cuda.grid(3)
    if th >= Nth or m > NN or n > NN//m:
        return
    b=b_arr[th]

  
       
        
    sum=0.+0j      
    for i in range(2, k+1,2):

        count_l, factor_l = get_mn(i)#,factors,counts)
        for l in range(count_l):
                ml=factor_l[l,0]
                nl=factor_l[l,1]
                if i==k:
                    sum+= R[th,ml,nl]/(hmn(b,m,n)+m*n-hmn(b,ml,nl))  
                else:      
                    sum+=R[th,ml,nl]/(hmn(b,m,n)+m*n-hmn(b,ml,nl)) * Hmn[th,k-i,ml,nl]

    Hmn[th,k,m,n] = sum

 


def calc_Hmn_loop(NN,b_arr,R,Hmn,blocks,TPB):
    #nvtx.range_push("H")
    
    for k in range(2, NN + 1,2):
        calc_Hmn_kernel_k[blocks, TPB](b_arr,NN, R,Hmn,k)
        cuda.synchronize()

   # nvtx.range_pop()
    return Hmn




# Load the precomputed map
mn_map_np    = np.load(os.path.join(base_dir, "mn_map.npy"))    # shape = (tot_mn, 2)
tot_mn, _    = mn_map_np.shape

# Copy into on‐chip constant memory (max ~64 KiB total)

ALL_MN=np.load(os.path.join(base_dir, "mn_pairs.npy"))
ALL_COUNTS=np.load(os.path.join(base_dir, "mn_counts.npy"))

@cuda.jit(fastmath=True)
def calc_Hmn_all_k(b_arr, NN, R, Hmn
                   ):
    """
    One block per theory (th), one thread per (m,n) pair.
    We loop over k inside the block, using __syncthreads()
    to enforce the Hmn[th, k-2, ...] → Hmn[th, k, ...] dependency.
    """
    all_factors=cuda.const.array_like(ALL_MN[:NN//2])
    all_counts=ALL_COUNTS[NN//2-1]
 
    th = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    if th >= b_arr.size or tid >= all_counts:
        return
    

    # unpack (m,n) from a small lookup table in constant/shared mem
    m = all_factors[tid, 0]
    n = all_factors[tid, 1]
    b = b_arr[th]

    # we assume Hmn[th, 0, ...] is already initialized (zero)
    # now fill Hmn[th, 2, ...], Hmn[th, 4, ...], … up to NN
    for k in range(2, NN+1, 2):
        sum=0.+0j      
        for i in range(2, k+1,2):

            count_l, factor_l = get_mn(i)#,factors,counts)
            for l in range(count_l):
                    ml=factor_l[l,0]
                    nl=factor_l[l,1]
                    if i==k:
                        sum+= R[th,ml,nl]/(hmn(b,m,n)+m*n-hmn(b,ml,nl))  
                    else:      
                        sum+=R[th,ml,nl]/(hmn(b,m,n)+m*n-hmn(b,ml,nl)) * Hmn[th,k-i,ml,nl]

        Hmn[th,k,m,n] = sum

        # barrier: wait until *all* threads in this block
        # have written Hmn[th, k, ...] before next k
        cuda.syncthreads()


@cuda.jit(fastmath=True)
def calc_H_kernel(b_arr,NNhalf, R,Hmn,H,h_arr):
 #   factors = cuda.const.array_like(FACTORS_CONST)
 #   counts  = cuda.const.array_like(COUNTS_CONST)
    Nth,Nst = h_arr.shape

    st,th,khalf=cuda.grid(3)
    if th >= Nth or khalf >NNhalf or st >= Nst:
        return
    k=2*khalf
    b=b_arr[th]
    h=h_arr[th,st]
    if k==0:
        H[th,st,0] = 1.+0j
        return
    
    sum=0.+0j
    for i in range(2,k+1,2):
        count, factor = get_mn(i)#,factors,counts)
        for l in range(count):
            ml=factor[l,0]
            nl=factor[l,1] 
            if i==k:
                sum+= R[th,ml,nl]/(h-hmn(b,ml,nl))  
            else:
                sum+= R[th,ml,nl]/(h-hmn(b,ml,nl)) * Hmn[th,k-i,ml,nl]          
    H[th,st,khalf] = sum
        




import time

def calc_H_rec(c_arr, hL_arr, hH_arr, h_arr, NN,print_time=False):
    assert  NN & 1 == 0, "NN must be even"
    NNhalf= NN // 2
    #NN_half = NN // 2
    c_arr=c_arr.to(torch.complex128)
    hL_arr=hL_arr.to(torch.complex128)
    hH_arr=hH_arr.to(torch.complex128)
    h_arr=h_arr.to(torch.complex128)
    device=c_arr.device
    Nth,Nst=h_arr.shape
    
    
    #factors=cuda.to_device(factors)
   # counts=cuda.to_device(counts)

    # ── complex Liouville parameter ────────────────────────────────

    b = torch.sqrt(c_arr - 13 + torch.sqrt(c_arr * c_arr - 26 * c_arr + 25)) / (2 *  math.sqrt(3))
    lam_L2 = 0.25 * (b + 1 / b) ** 2 - hL_arr
    lam_H2 = 0.25 * (b + 1 / b) ** 2 - hH_arr

    # ── R_{m,n} ----------------------------------------------------

    R =torch.zeros((Nth,NN + 1, NN + 1), dtype=torch.complex128, device=device)
    Hmn=torch.zeros((Nth,NN + 1,NN + 1, NN + 1), dtype=torch.complex128, device=device)
   # H=torch.zeros((Nth,Nst,NN//2 + 1), dtype=torch.complex128, device=device)
    # 1) Allocate H as zeros
    H = torch.zeros((Nth, Nst, NNhalf+1),
                    dtype=torch.complex128,
                    device=device)

    # 2) Set the k=0 slice (last-dimension index 0) to 1+0j
    H[..., 0] = 1 + 0j
    R=cuda.as_cuda_array(R)
    Hmn=cuda.as_cuda_array(Hmn)
    H=cuda.as_cuda_array(H)
    lam_L2=cuda.as_cuda_array(lam_L2)
    lam_H2=cuda.as_cuda_array(lam_H2)
    b=cuda.as_cuda_array(b)
    h_arr=cuda.as_cuda_array(h_arr)

    TPB = 64
    blocks = math.ceil(Nth/TPB)  

    
        
    start = time.time()
    calc_Rmn_kernel[blocks, TPB](NN,lam_L2,lam_H2,b,R)
    if print_time:
         print(f"Rmn kernel time{(time.time() - start)*1000} ms")
   

    #start = time.time()
    # calc_Hmn_kernel0[blocks, TPB](b,NN,R,Hmn)
    # if print_time:
    #     print(f"Hmn kernel time{(time.time() - start)*1000} ms")

   # TPB3 = (4,4,8)
   # blocks3 = (math.ceil((NN+1)/TPB3[0]),math.ceil((NN+1)/TPB3[1]),math.ceil(Nth/TPB3[2]))  
    
    
   # Hmn=calc_Hmn_loop(NN,b,R,Hmn,blocks3,TPB3)
    # precompute mn_map (shape (tot_mn,2)), factors, counts
# copy them to device or constant memory once at import

  

    # launch one block per theory, one thread per (m,n)
    threads = tot_mn
    blocks  = Nth
    start = time.time()
    
    calc_Hmn_all_k[blocks, threads](
        b, NN, R, Hmn
    )

    if print_time:
        print(f"Hmn kernel time{(time.time() - start)*1000} ms")

    TPB1 = (2, 16,16)
    blocks1 = (math.ceil(Nst/TPB1[0]),math.ceil(Nth/TPB1[1]),math.ceil((NN//2+1)/TPB1[2])) 

    start = time.time()
    calc_H_kernel[blocks1, TPB1](b,NNhalf, R,Hmn,H,h_arr)
    if print_time:
        print(f"H kernel time{(time.time() - start)*1000} ms")  

    # # ── flatten (m,n) → 1-D lists ----------------------------------
    # start,tot=start_list(NN_half)



    # # Allocate on GPU with torch
    # Rlist = torch.zeros(tot, dtype=torch.complex128, device=device)
    # hlist = torch.zeros(tot, dtype=torch.complex128, device=device)
    # mnhalf = torch.zeros(tot, dtype=torch.int32, device=device)
    # fill = torch.zeros(NN_half + 1, dtype=torch.int32, device=device)

    # # Convert to Numba CUDA arrays
    # Rlist = cuda.as_cuda_array(Rlist)
    # hlist = cuda.as_cuda_array(hlist)
    # mnhalf = cuda.as_cuda_array(mnhalf)
    # fill = cuda.as_cuda_array(fill)




    


    return torch.as_tensor(H).real

  
    


    
# device="cuda"
# N = 100  # or any desired batch size

# c_arr = torch.full((N,), 0.7999999 + 0j, dtype=torch.complex128, device=device)
# hL_arr = torch.full((N,), 0.125 + 0j, dtype=torch.complex128, device=device)
# hH_arr = torch.full((N,), 0.125 + 0j, dtype=torch.complex128, device=device)
# h_arr = torch.full((N,), 0.6666667 + 0j, dtype=torch.complex128, device=device)
# H=torch.as_tensor(calc_H_rec(c_arr, hL_arr, hH_arr, h_arr, 10,print_time=False))
# nvtx.range_push("H")
# H=torch.as_tensor(calc_H_rec(c_arr, hL_arr, hH_arr, h_arr, 20,print_time=False))
# nvtx.range_pop()

#print(H)


