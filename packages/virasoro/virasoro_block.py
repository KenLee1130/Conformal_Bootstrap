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

z,zbar=generate_gaussian_complex_points(1024, mean=0.5, std=0.1)
qqbar=z_to_q_cuda( torch.concat((z,zbar)))
q=qqbar[0:100]
qbar=qqbar[100:]

# -------------------------------------------------------------
# 0.  JSON  →  flat arrays   (numerator & denominator split)
# -------------------------------------------------------------
import os
import json
from pathlib import Path

def load_virasoro_coeffs():
    dir_path = os.path.dirname(__file__)
    
    coeffs_path = os.path.join(dir_path, "virasoro_coeffs.json")
    return load_coeffs_packed_split(coeffs_path)


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
def _H_kernel(b_val, hext,
              h_arr,
              idx_n, val_n, ofs_n,
              idx_d, val_d, ofs_d,
              kmax, Htbl):
    tid = cuda.grid(1)
    Nh = h_arr.size
    Nk = kmax + 1
    total = Nh * Nk
    if tid >= total:
        return

    ih = tid // Nk
    k  = tid - ih*Nk
    h  = h_arr[ih]

    num = _eval_poly_slice(idx_n, val_n, ofs_n[k], ofs_n[k+1],
                           b_val, h, hext)
    den = _eval_poly_slice(idx_d, val_d, ofs_d[k], ofs_d[k+1],
                           b_val, h, hext)

    Htbl[ih, k] = 0.0 if (num == 0.0 and den == 0.0) else num / den


def build_H_cuda(c, h_grid, hext,
                 idx_n, val_n, ofs_n,
                 idx_d, val_d, ofs_d,
                 kmax):
    """Return H[Nh, kmax+1] (host)"""
    h_grid =cuda.as_cuda_array(h_grid)
    Nh = h_grid.size

    inner = math.sqrt(c*c - 26.0*c + 25.0)
    b_val = math.sqrt(-(c - 13.0 + inner)) / (2.0*math.sqrt(3.0))

    # device copies
    h_d    = cuda.to_device(h_grid)
    idx_n_d, val_n_d, ofs_n_d = map(cuda.to_device, (idx_n, val_n, ofs_n))
    idx_d_d, val_d_d, ofs_d_d = map(cuda.to_device, (idx_d, val_d, ofs_d))

    H_d = cuda.device_array((Nh, kmax+1), dtype=np.float64)

    TPB = 256
    blocks = (Nh*(kmax+1) + TPB - 1)//TPB
    _H_kernel[blocks, TPB](b_val, hext, h_d,
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
                       hext, c_const, kmax,
                       out_z, out_zb):       # (Nh,Nqp)
    ih, iq, ip = cuda.grid(3)                # ip = 0 / 1
    Nh, klen   = Harr.shape
    Nqp        = q_all.shape[1]
    if ih >= Nh or iq >= Nqp or ip >= 2:
        return

    q   = q_all[ip, iq]
    z=z_all[ip, iq]
    h   = h_vals[ih]
    c   = c_const

    # Σ_k H_k q^{2k}
    qsq, qpow, series = q*q, 1.0, 0.0
    for k in range(kmax + 1):
        if k: qpow *= qsq
        series += Harr[ih, k] * qpow

    # prefactor
   
    t3  = theta3_trunc(q, kmax)
    exp_pref = (1.0 - c)/24.0 + h
    exp_z    = (c - 1.0)/24.0 - 2.0*hext
    exp_th   = (c - 1.0)/2.0  - 16.0*hext

    pref = (math.pow(16.0, exp_pref) * math.pow(q, exp_pref) *
            math.pow(1.0 - z, exp_z) * math.pow(z, exp_z) *
            math.pow(t3,      exp_th))

    val = series * pref
    if ip == 0:
        out_z [ih, iq] = val
    else:
        out_zb[ih, iq] = val


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

@cuda.jit
def calc_virasoro_cross_kernel(table_z, table_zb,        # (Nh,Nqp)
                    Nz, G):                # G:(Nh,Nz), v:(Nz,)
    ih, iz = cuda.grid(2)
    Nh, Nqp = table_z.shape
    if ih >= Nh or iz >= Nz:
        return
    iz1 = iz + Nz
    prod = table_z[ih, iz] * table_zb[ih, iz] \
         - table_z[ih, iz1] * table_zb[ih, iz1]
    G[ih, iz] = prod
    


# ─────────────────────────────────────────────────────────────
# 3.  User-facing functions
# ─────────────────────────────────────────────────────────────
def build_virasoro_block_cross(h, z, zbar, c, hext,coeffs, kmax=3):
    """
    h_mat : (Nh,p)
    Returns   G (Nh,Nz,p) ,  v (Nz,)
    """
    (idx_n, val_n, ofs_n,
     idx_d, val_d, ofs_d) = coeffs
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
    q_pair = torch.cat([
        q,
        q1mz
    ], dim=0)                                # shape: (Nqp,)
    z_pair=torch.cat([
        z,
        z1m
    ], dim=0)      

    qb_pair = torch.cat([
        qbar,
        qbar1mz
    ], dim=0)                               # shape: (Nqp,)
    zbar_pair=torch.cat([
        zbar,
        zbar1m
    ], dim=0)   

    q_all = torch.stack([q_pair, qb_pair], dim=0)  # shape: (2, Nqp)
    z_all = torch.stack([z_pair, zbar_pair], dim=0)  # shape: (2, Nqp)

    Nqp   = q_pair.numel()                         # total number of q’s

    z=cuda.as_cuda_array(z)
    zbar=cuda.as_cuda_array(zbar)
    h=cuda.as_cuda_array(h)

    

    # identity row for v(z)
    
    
    

    Harr  = build_H_cuda(c, h, hext,
                     idx_n, val_n, ofs_n,
                     idx_d, val_d, ofs_d,
                     kmax)



    # ── device buffers
    qs_d   = cuda.as_cuda_array(q_all)
    zs_d=cuda.as_cuda_array(z_all)
    h_d    = cuda.as_cuda_array(h)
    H_d    = cuda.as_cuda_array(Harr)
    tbl_z  = cuda.device_array((len(h), Nqp), np.float64)
    tbl_zb = cuda.device_array_like(tbl_z)

    # ── launch  block-evaluation  (3-D grid)
    TPB1 = (8, 8, 2)                                 # ih , iq , ip
    blocks1 = (math.ceil(h.size/TPB1[0]),
               math.ceil(Nqp/TPB1[1]),
               2)

    block_table_kernel[blocks1, TPB1](qs_d,zs_d, h_d,
                                      H_d, float64(hext),
                                      float64(c), kmax,
                                      tbl_z, tbl_zb)

    # ── allocate & combine
    G_big = cuda.device_array((h_d.size, Nz), np.float64)
   

    TPB2 = (16, 16)
    blocks2 = (math.ceil(h_d.size/TPB2[0]),
               math.ceil(Nz/TPB2[1]))

    calc_virasoro_cross_kernel[blocks2, TPB2](tbl_z, tbl_zb,
                                   np.int32(Nz), G_big)

    G_all = torch.as_tensor(G_big)
   

    # discard identity row & reshape to (Nh,Nz,p)
    #G_phys = G_all[1:].reshape(Nh, p, Nz).permute(0,2,1)  # (Nh,Nz,p)

    return G_all

if __name__ == "__main__":
    coeffs=load_virasoro_coeffs()
    z=torch.tensor([.4],device="cuda:0",dtype=torch.float64)
    zb=torch.tensor([.5],device="cuda:0",dtype=torch.float64)

    answer = build_virasoro_block_cross(torch.tensor([0., 0.66667, 1.4, 0.06667],device="cuda:0",dtype=torch.float64), z, zb, .69999, 3/80, coeffs, kmax=3)
    print(answer)