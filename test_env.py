import math
import numpy as np
import torch

# For zero‐copy bridging between Torch and Numba/CuPy
import cupy as cp
import numba
from numba import cuda
import matplotlib.pyplot as plt

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from packages.virasoro.virasoro_rec import calc_H_rec

###############################################################################
#                           GLOBAL CONFIG
###############################################################################
torch.set_default_dtype(torch.float64)  # HPC in double precision

MAX_TERMS = 1024
TOL = 1e-15

###############################################################################
#       Torch <-> Numba Float64 Bridging
###############################################################################
def torch_to_numba_devicearray(t: torch.Tensor):
    if not t.is_cuda:
        raise ValueError("Tensor must be on GPU.")
    if not t.is_contiguous():
        t = t.contiguous()
    dlpack = torch.utils.dlpack.to_dlpack(t)
    cupy_arr = cp.from_dlpack(dlpack)
    return numba.cuda.as_cuda_array(cupy_arr)

def numba_devicearray_to_torch(arr: numba.cuda.cudadrv.devicearray.DeviceNDArray):
    cupy_arr = cp.asarray(arr)
    dlpack = cupy_arr.toDlpack()
    return torch.utils.dlpack.from_dlpack(dlpack)

###############################################################################
#                     HPC Kernels in Float64
###############################################################################
@cuda.jit(device=True)
def _2F1_device(a, b, c, zr, zi):
    """Truncated hypergeometric expansion with complex ops in float64."""
    if c == 0.0:
        return 1.0, 0.0
    real_accum = 1.0
    imag_accum = 0.0
    term_r = 1.0
    term_i = 0.0
    for n in range(1, MAX_TERMS):
        denom = n * (c + n - 1.0)
        poch = ((a + n - 1.0) * (b + n - 1.0)) / denom
        new_r = term_r * zr - term_i * zi
        new_i = term_r * zi + term_i * zr
        term_r = poch * new_r
        term_i = poch * new_i
        if abs(term_r) < TOL and abs(term_i) < TOL:
            break
        real_accum += term_r
        imag_accum += term_i
    return real_accum, imag_accum

@cuda.jit(device=True)
def compute_g_device(d_val, s_val, x, y):
    h  = 0.5 * (d_val + s_val)
    hb = 0.5 * (d_val - s_val)
    fhz_r, fhz_i       = _2F1_device(h,  h,  2.0 * h,  x,  y)
    fhbz_b_r, fhbz_b_i = _2F1_device(hb, hb, 2.0 * hb, x, -y)
    fhz_b_r, fhz_i_temp   = _2F1_device(h,  h,  2.0 * h,  x, -y)
    fhb_z_r, fhb_z_i   = _2F1_device(hb, hb, 2.0 * hb, x,  y)
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    d_ = h + hb
    s_ = h - hb
    r_pow_d = r ** d_
    cos_s_th = math.cos(s_ * theta)
    sin_s_th = math.sin(s_ * theta)
    step1_r = fhz_r * fhbz_b_r - fhz_i * fhbz_b_i
    step1_i = fhz_r * fhbz_b_i + fhz_i * fhbz_b_r
    tmp1_r  = step1_r * cos_s_th - step1_i * sin_s_th
    tmp1_i  = step1_r * sin_s_th + step1_i * cos_s_th
    T1_r = r_pow_d * tmp1_r
    T1_i = r_pow_d * tmp1_i
    step2_r = fhz_b_r * fhb_z_r - fhz_i_temp * fhb_z_i
    step2_i = fhz_b_r * fhb_z_i + fhz_i_temp * fhb_z_r
    tmp2_r  = step2_r * cos_s_th + step2_i * sin_s_th
    tmp2_i  = -step2_r * sin_s_th + step2_i * cos_s_th
    T2_r = r_pow_d * tmp2_r
    T2_i = r_pow_d * tmp2_i
    return T1_r + T2_r, T1_i + T2_i

@cuda.jit(device=True)
def compute_G_element(d_val, s_val, x, y, dphi):
    xp = 1.0 - x
    yp = -y
    r1 = math.sqrt((x - 1.0) * (x - 1.0) + y * y)
    r2 = math.sqrt(x * x + y * y)
    r1_pow = r1 ** (2.0 * dphi)
    r2_pow = r2 ** (2.0 * dphi)
    g1_r, _ = compute_g_device(d_val, s_val, x, y)
    g2_r, _ = compute_g_device(d_val, s_val, xp, yp)
    return r1_pow * g1_r - r2_pow * g2_r

@cuda.jit
def compute_g_delta_kernel(d_arr, s_arr, x_arr, y_arr, dphi, g_delta_mat):
    i_g, i_z, i_state = cuda.grid(3)
    N_g, N_state = d_arr.shape
    N_z = x_arr.size
    stride_g, stride_z, stride_state = cuda.gridsize(3)
    for kk in range(i_z, N_z, stride_z):
        for ii in range(i_g, N_g, stride_g):
            for jj in range(i_state, N_state, stride_state):
                x_val = x_arr[kk]
                y_val = y_arr[kk]
                d_val = d_arr[ii, jj]
                s_val = s_arr[jj]
                g_delta_mat[ii, kk, jj] = compute_G_element(d_val, s_val, x_val, y_val, dphi)

# @cuda.jit
# def compute_W_v(d_max, x_arr, y_arr, dphi, W, v):
#     idx = cuda.grid(1)
#     if idx < x_arr.size:
#         x_val = x_arr[idx]
#         y_val = y_arr[idx]
#         r1 = math.sqrt((x_val - 1.0) * (x_val - 1.0) + y_val * y_val)
#         r2 = math.sqrt(x_val * x_val + y_val * y_val)
#         r1_pow = r1 ** (2.0 * dphi)
#         r2_pow = r2 ** (2.0 * dphi)
#         G0 = compute_G_element(d_max, 0.0, x_val, y_val, dphi)
#         if abs(G0) < 1e-6:
#             W[idx] = 1e6
#         else:
#             W[idx] = 1.0 / (G0 * G0)
#         v[idx] = r1_pow - r2_pow

@cuda.jit
def compute_W_v(d_max, x_arr, y_arr, dphi, W, v):
    # Get 3D thread indices.
    i, j, k = cuda.grid(3)
    # Get the overall grid size in each dimension.
    G0, G1, G2 = cuda.gridsize(3)
    # Flatten the 3D index into a single 1D index.
    idx = i + j * G0 + k * G0 * G1
    if idx < x_arr.size:
        x_val = x_arr[idx]
        y_val = y_arr[idx]
        r1 = math.sqrt((x_val - 1.0) * (x_val - 1.0) + y_val * y_val)
        r2 = math.sqrt(x_val * x_val + y_val * y_val)
        r1_pow = r1 ** (2.0 * dphi)
        r2_pow = r2 ** (2.0 * dphi)
        G0_val = compute_G_element(d_max, 0.0, x_val, y_val, dphi)
        if abs(G0_val) < 1e-6:
            W[idx] = 1e6
        else:
            W[idx] = 1.0 / (G0_val * G0_val)
        v[idx] = r1_pow - r2_pow

@cuda.jit
def compute_W(d_max, x_arr, y_arr, dphi, W):
    # Get 3D thread indices.
    i, j, k = cuda.grid(3)
    # Get the overall grid size in each dimension.
    G0, G1, G2 = cuda.gridsize(3)
    # Flatten the 3D index into a single 1D index.
    idx = i + j * G0 + k * G0 * G1
    if idx < x_arr.size:
        x_val = x_arr[idx]
        y_val = y_arr[idx]
        G0_val = compute_G_element(d_max, 0.0, x_val, y_val, dphi)
        if abs(G0_val) < 1e-6:
            W[idx] = 1e6
        else:
            W[idx] = 1.0 / (G0_val * G0_val)

###############################################################################
#               HPC Reward Functions in Float64
###############################################################################



def build_G_v_th_batch_synth(h_mat,h_ans, z, zbar, c,c_ans, hext,hext_ans,ope_ans,idx_n, val_n, ofs_n,
     idx_d, val_d, ofs_d, kmax=3,kmax1=10,analytic_H=False):
    """
    h_mat : (Nth,p)
    Returns   G (Nh,Nz,p) ,  v (Nz,)
    """
    
    h_all = torch.cat([h_mat, h_ans.unsqueeze(0)], dim=0)
    c= torch.cat([c, c_ans], dim=0)
    hext = torch.cat([hext, hext_ans], dim=0)
    
    Nth,p=h_all.size()


    assert len(c)==Nth
    assert len(hext)==Nth
    #h_all=torch.cat([torch.zeros((Nth,1),device=h_mat.device,dtype=h_mat.dtype),h_mat],dim=1) # length Nh*p+1
    assert z.size() == zbar.size()
    Nz   = z.size()[0]

    z1m=1-z
    zbar1m=1-zbar
    qqbar_all=z_to_q_cuda( torch.concat((z,zbar,z1m,zbar1m)))
    q=qqbar_all[0:Nz]
    qbar=qqbar_all[Nz:2*Nz]
    q1mz=qqbar_all[2*Nz:3*Nz]
    qbar1mz=qqbar_all[3*Nz:]


    # nome arrays
    q_all = torch.cat([
        q,
        q1mz,
        qbar,
        qbar1mz
    ], dim=0)                               
    z_all=torch.cat([
        z,
        z1m,
        zbar,
        zbar1m
    ], dim=0)        

    Nz_tot   =len(z_all)                 # total number of q’s


 
    

    # identity row for v(z)
    
    
    
    if kmax<=6 and analytic_H:
        Harr  = build_H_cuda(c, h_all, hext,
                        idx_n, val_n, ofs_n,
                        idx_d, val_d, ofs_d,
                        kmax)
    else:
        Harr  =calc_H_rec(c,hext,hext,h_all,kmax)

    h_d    = cuda.as_cuda_array(h_all)
    hext= cuda.as_cuda_array(hext)
    c=cuda.as_cuda_array(c)

    # ── device buffers
    qs_d   = cuda.as_cuda_array(q_all)
    zs_d=cuda.as_cuda_array(z_all)
   
    H_d    = cuda.as_cuda_array(Harr)
    tbl_z  = cuda.device_array((Nth,p, Nz_tot), np.float64)
   

    # ── launch  block-evaluation  (3-D grid)
    TPB1 = (2, 8, 8)                                 # ih , iq , ip
    blocks1 = ( math.ceil(Nz_tot/TPB1[0]),
               math.ceil((p)/TPB1[1]),
               math.ceil(Nth/TPB1[2]), 
              
              )

    block_table_kernel[blocks1, TPB1](qs_d,zs_d, h_d,
                                      H_d, hext,
                                      c,kmax, kmax1,
                                      tbl_z)
    cuda.synchronize()

    # ── allocate & combine
    G_big = cuda.device_array((p,Nth, Nz), np.float64)

 
    tbl_zb=tbl_z[:,:,2*Nz:]
    tbl_z=tbl_z[:,:,:2*Nz]
    
    make_G_kernel_th[blocks1, TPB1](tbl_z, tbl_zb,
                                   np.int32(Nz), G_big)
    cuda.synchronize()

    G_all = torch.as_tensor(G_big)
    
    Nth-=1 # remove identity row for v(z)
    # discard identity row & reshape to (Nh,Nz,p)
    G_phys = G_all[:,:-1].reshape(Nth, p, Nz).permute(0,2,1)  # (Nth,Nz,p)
       

    v=torch.einsum('zp,p->z', G_all[:,-1].permute(1,0),ope_ans)
    return G_phys, v

def build_G_v_th_batch(h_mat, z, zbar, c, hext,idx_n, val_n, ofs_n,
     idx_d, val_d, ofs_d, kmax=10,kmax1=10,analytic_H=False):
    """
    h_mat : (Nth,p)
    Returns   G (Nh,Nz,p) ,  v (Nz,)
    """
    Nth,p=h_mat.size()
    assert len(c)==Nth
    assert len(hext)==Nth
    h_all=torch.cat([torch.zeros((Nth,1),device=h_mat.device,dtype=h_mat.dtype),h_mat],dim=1) # length Nh*p+1
    assert z.size() == zbar.size()
    Nz   = z.size()[0]

    z1m=1-z
    zbar1m=1-zbar
    qqbar_all=z_to_q_cuda( torch.concat((z,zbar,z1m,zbar1m)))
    q=qqbar_all[0:Nz]
    qbar=qqbar_all[Nz:2*Nz]
    q1mz=qqbar_all[2*Nz:3*Nz]
    qbar1mz=qqbar_all[3*Nz:]


    # nome arrays
    q_all = torch.cat([
        q,
        q1mz,
        qbar,
        qbar1mz
    ], dim=0)                               
    z_all=torch.cat([
        z,
        z1m,
        zbar,
        zbar1m
    ], dim=0)        

    Nz_tot   =len(z_all)                 # total number of q’s


 
    

    # identity row for v(z)
    
    
    

    if kmax<=4 and analytic_H:
        Harr  = build_H_cuda(c, h_all, hext,
                        idx_n, val_n, ofs_n,
                        idx_d, val_d, ofs_d,
                        kmax)
    else:
        Harr  =calc_H_rec(c,hext,hext,h_all,kmax+2)
    
    h_d    = cuda.as_cuda_array(h_all)
    hext= cuda.as_cuda_array(hext)
    c=cuda.as_cuda_array(c)

    # ── device buffers
    qs_d   = cuda.as_cuda_array(q_all)
    zs_d=cuda.as_cuda_array(z_all)
   
    H_d    = cuda.as_cuda_array(Harr)
    tbl_z  = cuda.device_array((Nth,p+1, Nz_tot), np.float64)
   

    # ── launch  block-evaluation  (3-D grid)
    TPB1 = (2, 8, 8)                                 # ih , iq , ip
    blocks1 = ( math.ceil(Nz_tot/TPB1[0]),
               math.ceil((p+1)/TPB1[1]),
               math.ceil(Nth/TPB1[2]), 
              
              )

    block_table_kernel[blocks1, TPB1](qs_d,zs_d, h_d,
                                      H_d, hext,
                                      c,kmax, kmax1,
                                      tbl_z)
    cuda.synchronize()

    # ── allocate & combine
    G_big = cuda.device_array((p+1,Nth, Nz), np.float64)
    v_d   = cuda.device_array((Nth,Nz), np.float64)

 
    tbl_zb=tbl_z[:,:,2*Nz:]
    tbl_z=tbl_z[:,:,:2*Nz]
    
    make_G_v_kernel_th[blocks1, TPB1](tbl_z, tbl_zb,
                                   np.int32(Nz), G_big, v_d)
    cuda.synchronize()

    G_all = torch.as_tensor(G_big)
    v     = torch.as_tensor(v_d)

    # discard identity row & reshape to (Nh,Nz,p)
    G_phys = G_all[1:].reshape(Nth, p, Nz).permute(0,2,1)  # (Nth,Nz,p)

    return G_phys, v


def calculate_c_rew_th_synth(h,h_ans, z,zbar, hext,hext_ans,c,c_ans,ope_ans,coeffs,N_lsq=20,kmax=2,kmax1=10,device="cuda"):
    idx_n, val_n, ofs_n,idx_d, val_d, ofs_d=coeffs
    N_z = len(z)
    N_th = len(h)
    assert len(c)==N_th
    assert len(hext)==N_th
    assert N_z%N_lsq==0, "N_z should be integer times N_lsq"
    N_stat=N_z//N_lsq #how many times to calculate lsq for std stats

    N_state=len(h[0])
    G, v = build_G_v_th_batch_synth(h,h_ans, z, zbar, c,c_ans, hext,hext_ans,ope_ans,idx_n, val_n, ofs_n,
     idx_d, val_d, ofs_d, kmax=kmax,kmax1=kmax1)
  
   # build_G_v(h, z, zbar, c, hext, kmax, coeffs)

    G=torch.as_tensor(G, device=device) # (N_deltas,N_z, N_state)
    G=G.view((N_th,N_stat,N_lsq,N_state)).permute(1,0,2,3) #(N_stat,N_deltas,N_lsq,N_state)


    v=torch.as_tensor(v, device=device)#N_z
    v= v.unsqueeze(0).repeat(N_th,1)  # N_deltas,N_z
    v=v.view((N_th,N_stat,N_lsq)).permute(1,0,2) #N_stat, N_deltas, N_lsq

    
    

    
    # We sum over z: 'gzn,gzm->gnm'
    #GT_G = torch.einsum('sgzn,sgzm->sgnm', G, G)  
    #G_v = torch.einsum('sgzn,sgz->sgn', G, v)

    #c = torch.linalg.solve(GT_G, G_v)  # Solve for each batch
    c = torch.linalg.lstsq(G, v).solution  # Solve for each batch
    Gc = torch.einsum('sgzn,sgn->sgz', G, c)
    # Step 2: Compute residual vector (G.c - v)
    residual_vector = Gc + v  # Residual vector: [N_stat,N_deltas, N_lsq]

    # Step 3: Compute weighted residuals 

    # Step 4: Compute the final residual for each batch
    residual = torch.einsum('sgz,sgz->sg', residual_vector, residual_vector)  # Result: [N_stat,N_deltas]


    return c,residual #[N_stat,N_deltas, N_state] and [N_stat,N_deltas]

def calculate_c_rew_th(h, z,zbar, hext,c,coeffs,N_lsq=20,kmax=2,kmax1=10,device="cuda",analytic_H=False):
    idx_n, val_n, ofs_n,idx_d, val_d, ofs_d=coeffs
    N_z = len(z)
    N_th = len(h)
    assert len(c)==N_th
    assert len(hext)==N_th
    assert N_z%N_lsq==0, "N_z should be integer times N_lsq"
    N_stat=N_z//N_lsq #how many times to calculate lsq for std stats

    N_state=len(h[0])
    G, v = build_G_v_th_batch(h, z, zbar, c, hext,idx_n, val_n, ofs_n,
     idx_d, val_d, ofs_d, kmax=kmax,kmax1=kmax1,analytic_H=analytic_H)
   # build_G_v(h, z, zbar, c, hext, kmax, coeffs)

    G=torch.as_tensor(G, device=device) # (N_deltas,N_z, N_state)
    G=G.view((N_th,N_stat,N_lsq,N_state)).permute(1,0,2,3) #(N_stat,N_deltas,N_lsq,N_state)


    v=torch.as_tensor(v, device=device)#N_z
    #v= v0.unsqueeze(0).repeat(N_th,1)  # N_deltas,N_z
    v=v.view((N_th,N_stat,N_lsq)).permute(1,0,2) #N_stat, N_deltas, N_lsq

    
    

    
    # We sum over z: 'gzn,gzm->gnm'
    #GT_G = torch.einsum('sgzn,sgzm->sgnm', G, G)  
    #G_v = torch.einsum('sgzn,sgz->sgn', G, v)

    #c = torch.linalg.solve(GT_G, G_v)  # Solve for each batch
    c = torch.linalg.lstsq(G, v).solution  # Solve for each batch
    Gc = torch.einsum('sgzn,sgn->sgz', G, c)
    # Step 2: Compute residual vector (G.c - v)
    residual_vector = Gc -v  # Residual vector: [N_stat,N_deltas, N_lsq]

    # Step 3: Compute weighted residuals 

    # Step 4: Compute the final residual for each batch
    residual = torch.einsum('sgz,sgz->sg', residual_vector, residual_vector)  # Result: [N_stat,N_deltas]


    return c,residual #[N_stat,N_deltas, N_state] and [N_stat,N_deltas]


def set_numba_device(device: str):
    """
    Select the Numba CUDA device from a string of the form "cuda:N".

    Parameters
    ----------
    device : str
        Must be "cuda:<index>", e.g. "cuda:0" or "cuda:1".

    Raises
    ------
    ValueError
        If the format is wrong or the index isn’t an integer.
    """
    if device == "cuda":
        cuda.select_device(0)
    else:
        prefix = "cuda:"
        if not device.startswith(prefix):
            raise ValueError(f"Invalid device string {device!r}, expected format 'cuda:N'")
        # parse the index after the colon
        idx_str = device[len(prefix):]
        try:
            idx = int(idx_str)
        except ValueError:
            raise ValueError(f"Invalid CUDA index {idx_str!r} in {device!r}")

        # tear down any existing context and select the new device
        #cuda.close()           # closes current context if one exists
        cuda.select_device(idx)
    
def least_sq_std_rew_and_c_th_synth(theory, z,zbar,phys_state,ope_ans,coeffs,N_lsq=16,kmax=2,n_states_rew=2,kmax1=10,device="cuda"):
    set_numba_device(device)
    h=theory[:,:-2]
    hext=theory[:,-1]
    c_val=theory[:,-2]
    phys_state=phys_state.unsqueeze(1) # (1,3)
    h_ans=phys_state[0]
    c_ans=phys_state[1]
    hext_ans=phys_state[2]
    
    # 2) Construct G matrix using GPU
    # d_values = deltas, s_values = spins
    # According to definition: h=(Δ+s)/2, hb=(Δ-s)/2
    # The kernel expects d_val=Δ, s_val=s
    
 
    cs,rews = calculate_c_rew_th_synth(h,h_ans, z,zbar, hext,hext_ans,c_val,c_ans,ope_ans,coeffs,N_lsq,kmax,kmax1,device=device)
   
    c_mean=torch.mean(cs,0) #[N_deltas, N_state] 
    c_std=torch.std(cs,0) #[N_deltas, N_state]
    r_stat=c_std/c_mean #[N_deltas, N_state]
    r=-torch.sum(torch.log(torch.abs(r_stat[:,0:n_states_rew])),dim=1) #[N_deltas, ]

   # r = torch.nan_to_num(r, nan=-20.0)
    # replace -inf/+inf with zeros, or some finite floor
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
  
        
    return r,c_mean,c_std


def res_rew_and_c_th(theory, z,zbar,coeffs,N_lsq,kmax=2,n_states_rew=2,kmax1=10,device="cuda",analytic_H=False):    
 


    set_numba_device(device)

    h=theory[:,:-2]
    c_val=theory[:,-2]
    hext=theory[:,-1]
    # 2) Construct G matrix using GPU
    # d_values = deltas, s_values = spins
    # According to definition: h=(Δ+s)/2, hb=(Δ-s)/2
    # The kernel expects d_val=Δ, s_val=s
    
 
    cs,res = calculate_c_rew_th(h, z,zbar, hext,c_val,coeffs,N_lsq,kmax,kmax1,device=device,analytic_H=analytic_H)
   
    r=-res #[N_deltas, ]

    
        
    return r
        
def least_sq_std_rew_and_c_th(theory, z,zbar,coeffs,N_lsq=16,kmax=2,n_states_rew=2,kmax1=10,device="cuda",analytic_H=False):    
 


    set_numba_device(device)

    h=theory[:,:-2]
    c_val=theory[:,-2]
    hext=theory[:,-1]
    # 2) Construct G matrix using GPU
    # d_values = deltas, s_values = spins
    # According to definition: h=(Δ+s)/2, hb=(Δ-s)/2
    # The kernel expects d_val=Δ, s_val=s
    
 
    cs,rews = calculate_c_rew_th(h, z,zbar, hext,c_val,coeffs,N_lsq,kmax,kmax1,device=device,analytic_H=analytic_H)
    c_mean=torch.mean(cs,0) #[N_deltas, N_state] 
    c_std=torch.std(cs,0) #[N_deltas, N_state]
    r_stat=c_std/c_mean #[N_deltas, N_state]
    r=-torch.sum(torch.log(torch.abs(r_stat[:,0:n_states_rew])),dim=1) #[N_deltas, ]

    
        
    return r,c_mean,c_std

@cuda.jit
def make_G_kernel_th(table_z, table_zb,        # (Nh,Nqp)
                    Nz, G):                # G:(Nh,Nz), v:(Nz,)
    i0z,i0h,i0th= cuda.grid(3)
    stride_z,stride_h,stride_th=cuda.gridsize(3)
    Nth,Nh, Nqp = table_z.shape

    for iz in range(i0z,Nz,stride_z):
        iz1 = iz + Nz
        for ih in range(i0h,Nh,stride_h):
            for ith in range(i0th,Nth,stride_th):

                
                prod = table_z[ith,ih, iz] * table_zb[ith,ih, iz] \
                    - table_z[ith,ih, iz1] * table_zb[ith,ih, iz1]
             
                G[ih,ith, iz] = prod
        

@cuda.jit
def make_G_v_kernel_th(table_z, table_zb,        # (Nh,Nqp)
                    Nz, G, v):                # G:(Nh,Nz), v:(Nz,)
    i0z,i0h,i0th= cuda.grid(3)
    stride_z,stride_h,stride_th=cuda.gridsize(3)
    Nth,Nh, Nqp = table_z.shape

    for iz in range(i0z,Nz,stride_z):
        iz1 = iz + Nz
        for ih in range(i0h,Nh,stride_h):
            for ith in range(i0th,Nth,stride_th):

                
                prod = table_z[ith,ih, iz] * table_zb[ith,ih, iz] \
                    - table_z[ith,ih, iz1] * table_zb[ith,ih, iz1]
             
                G[ih,ith, iz] = prod
                if ih == 0:                               # h=0 row →  v_z
                    v[ith,iz] = -prod

###############################################################################
#      Additional HPC Reward Functions using a Weighting Function (W_func)
###############################################################################
@cuda.jit
def compute_v_kernel(x_arr, y_arr, dphi, v):
    idx = cuda.grid(1)
    if idx < x_arr.size:
        x = x_arr[idx]
        y = y_arr[idx]
        r1 = math.sqrt((x - 1.0) * (x - 1.0) + y * y)
        r2 = math.sqrt(x * x + y * y)
        v[idx] = r1 ** (2.0 * dphi) - r2 ** (2.0 * dphi)




def calculate_c_rew_W_func(d_values, s_values, zs, dphi, W_func, N_lsq=20,device ="cuda:0"):
    # Ensure inputs are CUDA tensors with float64
    for t in [d_values, s_values]:
        assert t.is_cuda and t.dtype == torch.float64

    x_values = zs.real
    y_values = zs.imag
    N_z = len(x_values)
    N_deltas = len(d_values)
    assert N_z % N_lsq == 0, "N_z should be integer times N_lsq"
    N_stat = N_z // N_lsq
    N_state = len(d_values[0])

    host_array = np.zeros((N_deltas, N_z, N_state), dtype=np.float64)
    g_delta_device = cuda.to_device(host_array)

    d_device = torch_to_numba_devicearray(d_values)
    s_device = torch_to_numba_devicearray(s_values)
    x_device = torch_to_numba_devicearray(x_values)
    y_device = torch_to_numba_devicearray(y_values)

    # threads = (4, 8, 4)
    threads_per_block = (2, 2, 8) # MAGIC
    bx = math.ceil(N_deltas / threads_per_block[0])
    by = math.ceil(N_z / threads_per_block[1])
    bz = math.ceil(N_state / threads_per_block[2])
    compute_g_delta_kernel[(bx, by, bz), threads_per_block](
        d_device, s_device, x_device, y_device, dphi, g_delta_device
    )

    # For compute_v_kernel, use optimal configuration: block size 32, grid size 32.
    optimal_block_v = 32
    optimal_grid_v = 32
    v_device = cuda.device_array(N_z, dtype=np.float64)
    compute_v_kernel[(optimal_grid_v,), (optimal_block_v,)](x_device, y_device, dphi, v_device)

    G = torch.as_tensor(g_delta_device.copy_to_host(), device=device, dtype=torch.float64)
    G = G.view((N_deltas, N_stat, N_lsq, N_state)).permute(1, 0, 2, 3)
    x_torch = torch.as_tensor(x_device.copy_to_host(), device=device, dtype=torch.float64)
    y_torch = torch.as_tensor(y_device.copy_to_host(), device=device, dtype=torch.float64)
    W_diag = W_func(x_torch, y_torch)
    W_diag = W_diag.view((N_stat, N_lsq))
    v0 = torch.as_tensor(v_device.copy_to_host(), device=device, dtype=torch.float64)
    v = v0.unsqueeze(0).repeat(N_deltas, 1)
    v = v.view((N_deltas, N_stat, N_lsq)).permute(1, 0, 2)
    W = torch.diag_embed(W_diag)
    WG = torch.einsum('szz,sgzn->sgzn', W, G)
    GT_WG = torch.einsum('sgzn,sgzm->sgnm', G, WG)
    WG_v = torch.einsum('sgzn,sgz->sgn', WG, v)

    c = -1.0 * torch.linalg.solve(GT_WG, WG_v)

    #c = -1.0 * torch.linalg.solve(GT_WG, WG_v)
    Gc = torch.einsum('sgzn,sgn->sgz', G, c)
    residual_vector = Gc + v
    W_residual = torch.einsum('sy,sgy->sgy', W_diag, residual_vector)
    residual = torch.einsum('sgz,sgz->sg', W_residual, residual_vector)
    return c, residual

def least_sq_std_rew_W_func(d_values, zs, s_values, dSigma, W_func, N_lsq=20, n_states_rew=2):
    cs, _ = calculate_c_rew_W_func(d_values, s_values, zs, dSigma, W_func, N_lsq=N_lsq)
    if cs is None:
        return torch.tensor(0.)    
    c_mean = torch.mean(cs, 0)
    c_std = torch.std(cs, 0)
    r_stat = c_std / c_mean
    r = -torch.sum(torch.log(torch.abs(r_stat[:, :n_states_rew])), dim=1)
    return r,c_mean,c_std


def get_W0(zs,delta_Sigma):
    x_device = cuda.as_cuda_array(zs.real)
    y_device = cuda.as_cuda_array(zs.imag)
    W_device = cuda.device_array(len(zs), dtype=np.float64)
    v_device = cuda.device_array(len(zs), dtype=np.float64)
    threadsperblock = 32
    blockspergrid = (len(zs)+ (threadsperblock - 1)) // threadsperblock
    compute_W_v[blockspergrid,threadsperblock](7.0, x_device, y_device, delta_Sigma,W_device,v_device)
    return torch.as_tensor(W_device)

def calculate_c_rew_synth(d_vals0, s_vals0,c0,d_vals, s_vals, x_vals, y_vals, dphi, d_max=9.0, N_lsq=32):
    # Ensure inputs are CUDA tensors with float64
    for t in [d_vals, s_vals,d_vals0, s_vals0,c0, x_vals, y_vals]:
        assert t.is_cuda and t.dtype == torch.float64

    N_deltas = d_vals.shape[0]
    N_deltas0 = d_vals0.shape[0]
    N_state = d_vals.shape[1]
    N_state0 = d_vals0.shape[1]
    N_z = x_vals.shape[0]
    assert N_z % N_lsq == 0, "N_z must be multiple of N_lsq"
    N_stat = N_z // N_lsq

    d_nb = torch_to_numba_devicearray(d_vals)
    s_nb = torch_to_numba_devicearray(s_vals)

    d_nb0 = torch_to_numba_devicearray(d_vals0)
    s_nb0 = torch_to_numba_devicearray(s_vals0)

    x_nb = torch_to_numba_devicearray(x_vals)
    y_nb = torch_to_numba_devicearray(y_vals)

    g_delta_dev = cuda.device_array((N_deltas, N_z, N_state), dtype=np.float64)
    g_delta_dev0 = cuda.device_array((N_deltas0, N_z, N_state0), dtype=np.float64)
    W_dev = cuda.device_array(N_z, dtype=np.float64)

    # threads = (4, 8, 4)
    threads = (2, 2, 8) # MAGIC
    bx = math.ceil(N_deltas / threads[0])
    by = math.ceil(N_z / threads[1])
    bz = math.ceil(N_state / threads[2])
    compute_g_delta_kernel[(bx, by, bz), threads](d_nb, s_nb, x_nb, y_nb, dphi, g_delta_dev)
    compute_g_delta_kernel[(bx, by, bz), threads](d_nb0, s_nb0, x_nb, y_nb, dphi, g_delta_dev0)


    threads_2 = (2, 4, 4) # MAGIC
    cx = math.ceil(N_deltas / threads_2[0])
    cy = math.ceil(N_z / threads_2[1])
    cz = math.ceil(N_state / threads_2[2])
    compute_W[(cx, cy, cz), threads_2](d_max, x_nb, y_nb, dphi, W_dev)

    # # For 1D kernel compute_W_v, use optimal configuration: block size 32, grid size 64.
    # block_size_W_v = 32
    # grid2 = 64  # forced grid size
    # compute_W_v[(grid2,), (block_size_W_v,)](d_max, x_nb, y_nb, dphi, W_dev, v_dev)


    G_torch = numba_devicearray_to_torch(g_delta_dev)
    G_torch0 = numba_devicearray_to_torch(g_delta_dev0)    
    W_torch = numba_devicearray_to_torch(W_dev)

    G = G_torch.view(N_deltas, N_stat, N_lsq, N_state).permute(1, 0, 2, 3)
    G0 = G_torch0.view(N_deltas0, N_stat, N_lsq, N_state0).permute(1, 0, 2, 3)  

    W_diag = W_torch.view(N_stat, N_lsq)
    v = torch.einsum('sgzn,n->sgz', G0, c0) #N_stat, N_deltas, N_lsq

    W_mat = torch.diag_embed(W_diag)
    WG = torch.einsum('szz,sgzn->sgzn', W_mat, G)
    GT_WG = torch.einsum('sgzn,sgzm->sgnm', G, WG)
    WG_v = torch.einsum('sgzn,sgz->sgn', WG, v)


    c = -1.0 * torch.linalg.solve(GT_WG, WG_v)
    Gc = torch.einsum('sgzn,sgn->sgz', G, c)
    res_vec = Gc + v

    for i in range(N_stat):
        res_vec[i] = W_diag[i].unsqueeze(0) * res_vec[i]

    residual = torch.sum(res_vec * (Gc + v), dim=-1)
    return c, residual


def least_sq_std_rew_synth(d_vals0, s_vals0,c0,d_values, zs, s_vals, dSigma, d_max, N_lsq=20, n_states_rew=2):
    x_vals = zs.real
    y_vals = zs.imag
    c_out, res_out = calculate_c_rew_synth(d_vals0, s_vals0,c0,d_values, s_vals, x_vals, y_vals, dSigma, d_max, N_lsq=N_lsq)
     
    c_mean = c_out.mean(dim=0)
    c_std = c_out.std(dim=0)
    r_stat = c_std / (c_mean + 1e-12)
    r = -torch.sum(torch.log(torch.clamp(torch.abs(r_stat[:, :n_states_rew]), 1e-12)), dim=1)
    return r,c_mean,c_std


def precompute_G_v(d_values, s_values, x_values, y_values, dphi, d_max=9.0,device="cuda:0"):
    """
    Precompute the G matrix and the vector v without weighting, using your existing CUDA kernels.
    This function returns:
      - G: Torch tensor of shape (N_g, N_z, N_state) (same order as final usage)
      - v: Torch tensor of shape (N_g, N_z)
    
    Here, N_g = d_values.shape[0], N_state = d_values.shape[1], and N_z = len(x_values).
    """
    N_z = len(x_values)
    N_g = len(d_values)
    N_state = len(d_values[0])
    
    # Allocate host and device arrays
    host_array = np.zeros((N_g, N_z, N_state), dtype=np.float64)
    g_delta_device = cuda.to_device(host_array)

    d_device =  cuda.as_cuda_array(d_values)
    s_device =  cuda.as_cuda_array(s_values)
    x_device = cuda.as_cuda_array(x_values)
    y_device = cuda.as_cuda_array(y_values)

    threads_per_block = (4,8,4)
    blocks_per_grid_z = math.ceil(N_g / threads_per_block[2])
    blocks_per_grid_y = math.ceil(N_z / threads_per_block[1])
    blocks_per_grid_x = math.ceil(N_state / threads_per_block[0])

    # Compute G
    compute_g_delta_kernel[(blocks_per_grid_x,blocks_per_grid_y,blocks_per_grid_z), threads_per_block](
        d_device, s_device, x_device, y_device, dphi, g_delta_device
    )
    # Bring G back to CPU and then to torch tensor
    G_np = g_delta_device.copy_to_host()  # shape (N_g, N_z, N_state)
    G = torch.as_tensor(G_np, device=device, dtype=torch.float64)

    # Compute W and v for the *reference channel* (without W weighting)
    # Actually, we won't use W here. We'll just get v from G0 calculation:
    W_device = cuda.device_array(N_z, dtype=np.float64)
    v_device = cuda.device_array(N_z, dtype=np.float64)
    threadsperblock = 32
    blockspergrid = (N_z + (threadsperblock - 1)) // threadsperblock
    compute_W_v[blockspergrid, threadsperblock](d_max, x_device, y_device, dphi, W_device, v_device)

 
    v = torch.as_tensor(v_device, device=device, dtype=torch.float64)
   
    return G, v

def least_sq_std_rew_with_W_G_test(W,G,G0,c0,N_lsq=20,n_states_rew=2):
 



    # 2) Construct G matrix using GPU
    # d_values = deltas, s_values = spins
    # According to definition: h=(Δ+s)/2, hb=(Δ-s)/2
    # The kernel expects d_val=Δ, s_val=s
    
 
    cs,_ = calculate_c_rew_with_w_G_test(W,G,G0,c0,N_lsq=N_lsq) #[N_stat,N_deltas, N_state] and [N_stat,N_deltas]
    c_mean=torch.mean(cs,0) #[N_deltas, N_state] 
    c_std=torch.std(cs,0) #[N_deltas, N_state]
    r_stat=c_std/c_mean #[N_deltas, N_state]
    r=-torch.sum(torch.log(torch.abs(r_stat[:,0:n_states_rew])),dim=1) #[N_deltas, ]

   
        
    return r


def calculate_c_rew_with_w_G_test(W_diag,G,G0,c0,N_lsq=20):
    N_deltas, N_z,N_state = G.size()
    N_deltas_test, N_z,N_state_test = G0.size()
    assert N_z%N_lsq==0, "N_z should be integer times N_lsq"
    N_stat=N_z//N_lsq #how many times to calculate lsq for std stats

   

    G=G.view((N_deltas,N_stat,N_lsq,N_state)).permute(1,0,2,3)
    G0=G0.view((N_deltas_test,N_stat,N_lsq,N_state_test)).permute(1,0,2,3)  #N_stat, N_deltas, N_lsq, N_state

    W_diag=W_diag.view((N_stat,N_lsq))
    
    v = torch.einsum('sgzn,n->sgz', G0, c0) #N_stat, N_deltas, N_lsq

   # v= v.unsqueeze(0).repeat(N_deltas,1)  # N_deltas,N_z
   # v=v.view((N_deltas,N_stat,N_lsq)).permute(1,0,2) #N_stat, N_deltas, N_lsq

    
    

    # Step 1: Apply W to G
    W=torch.diag_embed(W_diag) #N_stat,N_lsq,N_lsq

    WG = torch.einsum('szz,sgzn->sgzn', W, G)

    # WG: (N_deltas, N_z, N_state)
    # G: (N_deltas, N_z, N_state)
    # We sum over z: 'gzn,gzm->gnm'
    GT_WG = torch.einsum('sgzn,sgzm->sgnm', G, WG)  
    WG_v = torch.einsum('sgzn,sgz->sgn', WG, v)

    c =-1.* torch.linalg.solve(GT_WG, WG_v)  # Solve for each batch
    Gc = torch.einsum('sgzn,sgn->sgz', G, c)
    # Step 2: Compute residual vector (G.c - v)
    residual_vector = Gc + v  # Residual vector: [N_stat,N_deltas, N_lsq]

    # Step 3: Compute weighted residuals 
    W_residual =  torch.einsum('sy,sgy->sgy', W_diag,residual_vector )    # Weighted residual vector: [N_g, N_z]

    # Step 4: Compute the final residual for each batch
    residual = torch.einsum('sgz,sgz->sg', W_residual, residual_vector)  # Result: [N_stat,N_deltas]
    del W,WG,GT_WG,WG_v,Gc,residual_vector,W_residual
    torch.cuda.empty_cache()

    return c,residual #[N_stat,N_deltas, N_state] and [N_stat,N_deltas]


"""
GPU Elliptic K with Numba-CUDA + PyTorch
---------------------------------------

Implements  K(m) = π / (2 AGM(1, √(1-m))).
Gradient dK/dm is provided so you can wrap it in a
torch.autograd.Function if needed.
"""

import math, torch, numpy as np
from numba import cuda, float32, float64


import torch
def generate_gaussian_complex_points(n, mean=0.5, std=0.1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    x = torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64)
    y = torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64)

    return x,y



def generate_gaussian_points_positive(n, mean=0.5, std=0.1,  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Uniform distribution in [0, 1] for both x and y
    x = torch.clamp(torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64),0.01,.99)
    y = torch.clamp(torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64),0.01,.99)
    return x, y
# ─────────────────────────────────────────────────────────
# device function: AGM loop (converges quadratically)  ────
# stopping tol chosen ~ double precision machine‐eps
# ─────────────────────────────────────────────────────────
@cuda.jit(device=True, inline=True)
def _agm_K(m):
    tol=1e-14
    a = 1.0
    b = math.sqrt(1.0 - m)
    it = 0
    while abs(a - b) > tol * a and it < 40:          # 40 ⇒ overkill
        tmp = 0.5 * (a + b)
        b   = math.sqrt(a * b)
        a   = tmp
        it += 1
    return math.pi / (2.0 * a)


# ─────────────────────────────────────────────────────────
# 1-D kernel: one thread = one element
# m_arr, out_arr are contiguous device 1-D views
# ─────────────────────────────────────────────────────────
@cuda.jit
def _ellipk_kernel(m_arr, out_arr):
    i = cuda.grid(1)
    if i < m_arr.size:
        out_arr[i] = _agm_K(m_arr[i])

@cuda.jit
def z_to_q_kernel(z, q):
    i = cuda.grid(1)
    if i < z.size:
        q[i] =math.exp(-math.pi* _agm_K(1.0-z[i])/_agm_K(z[i]))



# ─────────────────────────────────────────────────────────
# public API   ellipk(t)   (PyTorch tensor in/out, GPU)
# ─────────────────────────────────────────────────────────
def z_to_q_cuda(t: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    t : torch.Tensor on *CUDA* containing m = k**2 in [0, 1)

    Returns
    -------
    torch.Tensor of same shape/dtype/device with K(m)
    """
    assert t.is_cuda, "Input must be on CUDA device"
    assert t.dtype in (torch.float32, torch.float64)


    m_nb     = cuda.as_cuda_array(t)

    # allocate output
    out_nb = cuda.device_array(m_nb.shape, dtype=m_nb.dtype)

    # launch
    TPB = 16
    blocks = (m_nb.size + TPB - 1) // TPB
    z_to_q_kernel[blocks, TPB](m_nb, out_nb)

    # back to torch (zero-copy)
    return torch.as_tensor(out_nb)



#----------------------------------------------
# 0.  JSON  →  flat arrays   (numerator & denominator split)
# -------------------------------------------------------------
import json, numpy as np
from pathlib import Path



def load_coeffs_packed_split(path: Path):
    """
    Returns six arrays:

        idx_n , val_n , ofs_n   – numerator terms
        idx_d , val_d , ofs_d   – denominator terms

    * idx_* : int32 [ N* , 3 ]   (i , j , l)
    * val_* : float64 [ N* ]
    * ofs_* : int32  [ kmax+2 ]  slice offsets per level k
              →   slice k  =  [ ofs[k] : ofs[k+1] ]
    """
    with open(path) as f:
        raw = json.load(f)

    # sort by integer level
    items = sorted(raw.items(), key=lambda kv: int(kv[0]))
    kmax  = int(items[-1][0])

    # containers
    idx_num, val_num, ofs_num = [], [], [0]
    idx_den, val_den, ofs_den = [], [], [0]

    # loop over levels
    for k_str, rules in items:
        # ---------- numerator ----------
        for (i, j, l), c in rules["num"]:
            idx_num.append((i, j, l))
            val_num.append(float(c))
        ofs_num.append(len(idx_num))

        # ---------- denominator ----------
        for (i, j, l), c in rules["den"]:
            idx_den.append((i, j, l))
            val_den.append(float(c))
        ofs_den.append(len(idx_den))

    # to numpy
    idx_n = np.asarray(idx_num, dtype=np.int32)
    val_n = np.asarray(val_num, dtype=np.float64)
    ofs_n = np.asarray(ofs_num, dtype=np.int32)

    idx_d = np.asarray(idx_den, dtype=np.int32)
    val_d = np.asarray(val_den, dtype=np.float64)
    ofs_d = np.asarray(ofs_den, dtype=np.int32)

    return idx_n, val_n, ofs_n, idx_d, val_d, ofs_d#, kmax


# -------------------------------------------------------------
# 1.  Build H(c, h, hext)  on the GPU
#     (kernel code from the previous answer)
# -------------------------------------------------------------
import math
import numba as nb
from numba import cuda, float64

@cuda.jit(device=True, inline=True)
def _eval_poly_slice(idx, val, start, stop, b, h, hext):
    s = 0.0
    for t in range(start, stop):
        i = idx[t, 0]
        j = idx[t, 1]
        l = idx[t, 2]
        s += val[t] * math.pow(b, i) * math.pow(h, j) * math.pow(hext, l)
    return s

# ────────────────────────────────────────────────────────────────
# CUDA kernel   one thread → one  (ih , is , k) entry
# h_arr   : float64[Nh, Ns]
# Htbl    : float64[Nh, Ns, kmax+1]
# ────────────────────────────────────────────────────────────────
@cuda.jit
def _H_kernel(b_arr, hext_arr,
              h_arr,
              idx_n, val_n, ofs_n,
              idx_d, val_d, ofs_d,
              kmax, Htbl):
    ith,state,k = cuda.grid(3)

    Nth,Np = h_arr.shape
    assert Nth==b_arr.size
    assert hext_arr.size==Nth
    Nk = kmax + 1
 
    if ith >= Nth or k>=Nk or state>=Np:
        return

   
    h  = h_arr[ith,state]
    b_val=b_arr[ith]
    hext=hext_arr[ith]


    num = _eval_poly_slice(idx_n, val_n, ofs_n[k], ofs_n[k+1],
                           b_val, h, hext)
    den = _eval_poly_slice(idx_d, val_d, ofs_d[k], ofs_d[k+1],
                           b_val, h, hext)

    Htbl[ith,state, k] = 0.0 if (num == 0.0 and den == 0.0) else num / den


def build_H_cuda(c_arr, h_grid, hext_arr,
                 idx_n, val_n, ofs_n,
                 idx_d, val_d, ofs_d,
                 kmax):
    """Return H[Nh, kmax+1] (host)"""
    inner =torch.sqrt(c_arr*c_arr - 26.0*c_arr + 25.0)
    b_val = cuda.as_cuda_array(torch.sqrt(-(c_arr - 13.0 + inner)) / (2.0*math.sqrt(3.0)))

    h_grid =cuda.as_cuda_array(h_grid)
    hext_arr=cuda.as_cuda_array(hext_arr)
    Nth,p = h_grid.shape

 
    # device copies
    h_d    = cuda.to_device(h_grid)
    idx_n_d, val_n_d, ofs_n_d = map(cuda.to_device, (idx_n, val_n, ofs_n))
    idx_d_d, val_d_d, ofs_d_d = map(cuda.to_device, (idx_d, val_d, ofs_d))

    H_d = cuda.device_array((Nth,p, kmax+1), dtype=np.float64)
    
    
    TPB1 = (8, 8, 2)                                 # ih , iq , ip
    blocks1 = ( math.ceil(Nth/TPB1[0]),
               math.ceil(p/TPB1[1]),
               math.ceil((kmax+1)/TPB1[2]))

    _H_kernel[blocks1, TPB1](b_val, hext_arr, h_d,
                           idx_n_d, val_n_d, ofs_n_d,
                           idx_d_d, val_d_d, ofs_d_d,
                           kmax, H_d)
    cuda.synchronize()
    return H_d

# ─────────────────────────────────────────────────────────────
# 2.  Kernels
# ─────────────────────────────────────────────────────────────
@cuda.jit(device=True, inline=True)
def theta3_trunc(q, kmax):
    s = 1.0
    for n in range(kmax):
        s += 2.0 * math.pow(q, (n+1)*(n+1))
    return s

@cuda.jit(device=True, inline=True)
def theta2_trunc(q, kmax):
    s = 1.0
    for n in range(kmax):
        s += math.pow(q, (n+1)*(n+2))
    return 2.0*math.pow(q, 0.25)*s
@cuda.jit
def block_table_kernel(q_all,    
                       z_all,             # (2, Nqp)
                       h_vals,               # (Nh,)
                       Harr,                 # (Nh,kmax+1)
                       hext_arr, c_arr, kmax,kmax1,
                       out_z):       # (Nh,Nqp)
     #TODO use constant mem
    iz,state,ith = cuda.grid(3)                # ip = 0 / 1
    Nth, Np,_   = Harr.shape
    Nz        = q_all.size
    assert z_all.size==Nz
    if ith >= Nth or state>=Np or iz >= Nz:
        return
   
    q   = q_all[iz]
    z=z_all[iz]
    h   = h_vals[ith,state]
    c   = c_arr[ith]
    hext = hext_arr[ith]
    # Σ_k H_k q^{2k}
    qsq, qpow, series = q*q, 1.0, 0.0
    for k in range(kmax + 1):
        if k: qpow *= qsq
        series += Harr[ith,state, k] * qpow

    # prefactor
   
    t3  = theta3_trunc(q, kmax1)
    exp_pref = (1.0 - c)/24.0 + h
    exp_z    = (c - 1.0)/24.0 - 2.0*hext
    exp_th   = (c - 1.0)/2.0  - 16.0*hext

    pref = (math.pow(16.0, exp_pref) * math.pow(q, exp_pref) *
            math.pow(1.0 - z, exp_z) * math.pow(z, exp_z) *
            math.pow(t3,      exp_th))

    val = series * pref
    out_z [ith,state, iz] = val


@cuda.jit
def make_G_v_kernel(table_z, table_zb,        # (Nh,Nqp)
                    Nz, G, v):                # G:(Nh,Nz), v:(Nz,)
    ih, iz = cuda.grid(2)
    Nh, Nqp = table_z.shape
    if ih >= Nh or iz >= Nz:
        return
    iz1 = iz + Nz
    prod = table_z[ih, iz] * table_zb[ih, iz] \
         - table_z[ih, iz1] * table_zb[ih, iz1]
    G[ih, iz] = prod
    if ih == 0:                               # h=0 row →  v_z
        v[iz] = -prod

if __name__ == "__main__":
    device = "cuda"
    dtype  = torch.float64
    coeffs     = load_coeffs_packed_split("C:/Users/User/Desktop/git/Conformal_Bootstrap/packages/virasoro/virasoro_coeffs.json")
    def plot_potential(
        phys_state,
        bounds2,
        coeffs,
        device="cuda",
        dtype=torch.float64,
        contour_std=0.1,
        kmax=10,
        kmax1=10,
        n_states_rew=3
    ):
        # Make contour of the real reward on the (d1,d3) plane
        nx, ny = 200, 200
        xs = np.linspace(bounds2[0][0], bounds2[0][1], nx)
        ys = np.linspace(bounds2[1][0], bounds2[1][1], ny)
        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

        pts = torch.from_numpy(np.stack([Xg.ravel(), Yg.ravel()],1)).to(device, dtype=dtype)
        Sgrid = torch.empty((pts.size(0),3), device=device, dtype=dtype)
        Sgrid[:,0] = pts[:,0]
        Sgrid[:,1] = phys_state[1]
        Sgrid[:,2] = pts[:,1]
        z, zbar = generate_gaussian_points_positive(400, std=contour_std, device=device)
        Rf, _, _ = least_sq_std_rew_and_c_th(
            Sgrid, z, zbar, coeffs,
            N_lsq=20, kmax=kmax, kmax1=kmax1,
            n_states_rew=n_states_rew,
            device=device
        )
        Rg = Rf.cpu().numpy().reshape(nx,ny)

        fig, ax = plt.subplots(figsize=(6,6))
        cf = ax.contourf(Xg, Yg, Rg, levels=50, cmap="viridis")
        fig.colorbar(cf, label="reward")

    plot_potential(
        phys_state=torch.tensor([1/7, -13/14-.001,-1/14], dtype=dtype, device=device),
        bounds2=[(0.02,0.6),(-0.6,-0.01)],
        coeffs=coeffs,
        device=device,
        dtype=dtype)