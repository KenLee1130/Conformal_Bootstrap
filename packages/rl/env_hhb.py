import math
import numpy as np
import torch

# For zero‐copy bridging between Torch and Numba/CuPy
from numba import cuda

from ..virasoro.virasoro_block import build_virasoro_block_cross, load_virasoro_coeffs
from ..virasoro.virasoro_rec import calc_H_rec, z_to_q_cuda, build_H_cuda, calc_H_rec, block_table_kernel
from ..global_block import torch_to_numba_devicearray, numba_devicearray_to_torch, compute_G_element, compute_g_delta_kernel

###############################################################################
#               Format of translation of typing
###############################################################################
from typing import Optional, Tuple, Union, Sequence

def normalize_theory_inputs(
    h_vals: Union[Sequence, torch.Tensor],
    hb_vals: Union[Sequence, torch.Tensor],
    h_ext:  Union[float, Sequence, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype]  = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将各种格式的 (h_vals, hb_vals, h_ext) 归一化为
      h:   Tensor[N_th, N_state]
      hb:  Tensor[N_th, N_state]
      hext:Tensor[N_th]
    """
    # 1) convert to tensors
    h   = torch.as_tensor(h_vals,  device=device, dtype=dtype)
    hb  = torch.as_tensor(hb_vals, device=device, dtype=dtype)
    hext= torch.as_tensor(h_ext,  device=device, dtype=dtype)

    # 2) normalize h, hb to 2D
    def _to_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0:
            # scalar → shape (1,1)
            return x.unsqueeze(0).unsqueeze(1)
        elif x.ndim == 1:
            # (N,) → (N,1)
            return x.unsqueeze(1)
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(f"Expected <=2 dims for h/hb, got {x.ndim}")
    h  = _to_2d(h)
    hb = _to_2d(hb)

    # 3) normalize hext to 1D
    if hext.ndim == 0:
        hext = hext.unsqueeze(0)
    elif hext.ndim == 1:
        pass
    else:
        raise ValueError(f"Expected 0 or 1 dims for h_ext, got {hext.ndim}")

    # 4) sanity checks
    N_th, N_state = h.shape
    if hb.shape != (N_th, N_state):
        raise ValueError(f"hb_vals must have same shape as h_vals; got {hb.shape} vs {h.shape}")
    if hext.shape[0] != N_th:
        raise ValueError(f"Length of h_ext ({hext.shape[0]}) must equal number of rows of h_vals ({N_th})")

    return h, hb, hext


###############################################################################
#               HPC Reward Functions of Pure Global
###############################################################################
@cuda.jit
def compute_W_v_Global(d_max, x_arr, y_arr, dphi, W, v):
    # Get 3D thread indices.
    i, j, k = cuda.grid(3)
    # Get the overall grid size in each dimension.
    G0, G1, G2 = cuda.gridsize(3)
    # Flatten the 3D index into a single 1D index.
    idx = i + j * G0 + k * G0 * G1
    if idx < x_arr.size:
        x_val = x_arr[idx]
        y_val = y_arr[idx]
        r1 = (x_val * y_val)
        r2 = ((1-x_val) * (1-y_val))
        r1_pow = r1 ** (2.0 * dphi)
        r2_pow = r2 ** (2.0 * dphi)
        G0_val = compute_G_element(d_max, 0.0, x_val, y_val, dphi)
        if abs(G0_val) < 1e-6:
            W[idx] = 1e6
        else:
            W[idx] = 1.0 / (G0_val * G0_val)
        v[idx] = (r2_pow - r1_pow)

def calculate_c_rew_Global(d_vals, s_vals, x_vals, y_vals, dphi, d_max=9.0, N_lsq=32):
    # Ensure inputs are CUDA tensors with float64
    for t in [d_vals, s_vals, x_vals, y_vals]:
        assert t.is_cuda and t.dtype == torch.float64

    N_deltas = d_vals.shape[0]
    N_state = d_vals.shape[1]
    N_z = x_vals.shape[0]
    assert N_z % N_lsq == 0, "N_z must be multiple of N_lsq"
    N_stat = N_z // N_lsq

    d_nb = torch_to_numba_devicearray(d_vals)
    s_nb = torch_to_numba_devicearray(s_vals)
    x_nb = torch_to_numba_devicearray(x_vals)
    y_nb = torch_to_numba_devicearray(y_vals)

    g_delta_dev = cuda.device_array((N_deltas, N_z, N_state), dtype=np.float64)
    W_dev = cuda.device_array(N_z, dtype=np.float64)
    v_dev = cuda.device_array(N_z, dtype=np.float64)

    # threads = (4, 8, 4)
    threads = (2, 2, 8) # MAGIC
    bx = math.ceil(N_deltas / threads[0])
    by = math.ceil(N_z / threads[1])
    bz = math.ceil(N_state / threads[2])
    compute_g_delta_kernel[(bx, by, bz), threads](d_nb, s_nb, x_nb, y_nb, dphi, g_delta_dev)

    threads_2 = (2, 4, 4) # MAGIC
    cx = math.ceil(N_deltas / threads_2[0])
    cy = math.ceil(N_z / threads_2[1])
    cz = math.ceil(N_state / threads_2[2])
    compute_W_v_Global[(cx, cy, cz), threads_2](d_max, x_nb, y_nb, dphi, W_dev, v_dev)

    G_torch = numba_devicearray_to_torch(g_delta_dev)
    W_torch = numba_devicearray_to_torch(W_dev)
    v_torch = numba_devicearray_to_torch(v_dev)

    G = G_torch.view(N_deltas, N_stat, N_lsq, N_state).permute(1, 0, 2, 3)
    W_diag = W_torch.view(N_stat, N_lsq)
    v = v_torch.unsqueeze(0).expand(N_deltas, -1).view(N_deltas, N_stat, N_lsq).permute(1, 0, 2)

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

def least_sq_std_rew_Global(h_vals, hb_vals, h_ext, x_vals, y_vals, N_lsq=20, n_states_rew=2):
    d_values = (h_vals + hb_vals)
    s_vals = (h_vals - hb_vals)
    d_max = max(d_values)
    c_out, res_out = calculate_c_rew_Global(d_values, s_vals, x_vals, y_vals, 2*h_ext, d_max, N_lsq=N_lsq)
    c_mean = c_out.mean(dim=0)
    c_std = c_out.std(dim=0)
    r_stat = c_std / (c_mean + 1e-12)
    r = -torch.sum(torch.log(torch.clamp(torch.abs(r_stat[:, :n_states_rew]), 1e-12)), dim=1)
    return r, c_mean, c_std


###############################################################################
#               HPC Reward Functions of Hybrid method
###############################################################################
@cuda.jit
def compute_W_v_Hybrid(d_max, x_arr, y_arr, block_dev, V_c, dphi, W, v):
    # Get 3D thread indices.
    i, j, k = cuda.grid(3)
    G0, G1, G2 = cuda.gridsize(3)
    idx = i + j * G0 + k * G0 * G1
    if idx < x_arr.size:
        x_val = x_arr[idx]
        y_val = y_arr[idx]
        # compute W as before
        G0_val = compute_G_element(d_max, 0.0, x_val, y_val, dphi)
        W[idx] = 1e6 if abs(G0_val) < 1e-6 else 1.0 / (G0_val * G0_val)
        tmp = 0.0
        for s in range(V_c.shape[0]):
            tmp += V_c[s] * block_dev[s, idx] * (x_val*y_val*(1-x_val)*(1-y_val))**dphi
        v[idx]=tmp

def calculate_c_rew_Hybrid(d_vals, s_vals, x_vals, y_vals, virasoro_info, dphi, d_max=9.0, N_lsq=32):
    # Ensure inputs are CUDA tensors with float64
    for t in [d_vals, s_vals, x_vals, y_vals]:
        assert t.is_cuda and t.dtype == torch.float64
    central_charge=virasoro_info["central_charge"]-1e-5
    V_c=virasoro_info["V_c"]
    V_d=virasoro_info["V_d"]
    
    N_deltas = d_vals.shape[0]
    N_state = d_vals.shape[1]
    N_z = x_vals.shape[0]
    assert N_z % N_lsq == 0, "N_z must be multiple of N_lsq"
    N_stat = N_z // N_lsq
    
    d_nb = torch_to_numba_devicearray(d_vals)
    s_nb = torch_to_numba_devicearray(s_vals)
    x_nb = torch_to_numba_devicearray(x_vals)
    y_nb = torch_to_numba_devicearray(y_vals)
    block_table=build_virasoro_block_cross(V_d, x_vals, y_vals, central_charge, dphi/2, kmax=3)
    
    g_delta_dev = cuda.device_array((N_deltas, N_z, N_state), dtype=np.float64)
    W_dev = cuda.device_array(N_z, dtype=np.float64)
    v_dev = cuda.device_array(N_z, dtype=np.float64)
    block_dev = cuda.to_device(block_table)
    Vc_dev    = cuda.to_device(V_c)

    # threads = (4, 8, 4)
    threads = (2, 2, 8) # MAGIC
    bx = math.ceil(N_deltas / threads[0])
    by = math.ceil(N_z / threads[1])
    bz = math.ceil(N_state / threads[2])
    compute_g_delta_kernel[(bx, by, bz), threads](d_nb, s_nb, x_nb, y_nb, dphi, g_delta_dev)

    threads_2 = (2, 4, 4) # MAGIC
    cx = math.ceil(N_deltas / threads_2[0])
    cy = math.ceil(N_z / threads_2[1])
    cz = math.ceil(N_state / threads_2[2])
    compute_W_v_Hybrid[(cx, cy, cz), threads_2](d_max, x_nb, y_nb, block_dev, Vc_dev, dphi, W_dev, v_dev)
    
    G_torch = numba_devicearray_to_torch(g_delta_dev)
    W_torch = numba_devicearray_to_torch(W_dev)
    v_torch = numba_devicearray_to_torch(v_dev)
    
    G = G_torch.view(N_deltas, N_stat, N_lsq, N_state).permute(1, 0, 2, 3)
    W_diag = W_torch.view(N_stat, N_lsq)
    v = v_torch.unsqueeze(0).expand(N_deltas, -1).view(N_deltas, N_stat, N_lsq).permute(1, 0, 2)

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

def least_sq_std_rew_Hybrid(h_vals, hb_vals, h_ext, 
                            x_vals, y_vals, 
                            virasoro_info, 
                            N_lsq=20, n_states_rew=2):
    d_values = (h_vals + hb_vals)
    s_vals = (h_vals - hb_vals)
    d_max = max(d_values)
    c_out, res_out = calculate_c_rew_Hybrid(d_values, s_vals, x_vals, y_vals, virasoro_info, 2*h_ext, d_max, N_lsq=N_lsq)
    c_mean = c_out.mean(dim=0)
    c_std = c_out.std(dim=0)
    
    r_stat = c_std / (c_mean + 1e-12)
    r = -torch.sum(torch.log(torch.clamp(torch.abs(r_stat[:, :n_states_rew]), 1e-12)), dim=1)
    return r, c_mean, c_std


###############################################################################
#               HPC Reward Functions of Pure Virasoro
###############################################################################
coeffs = load_virasoro_coeffs()

def set_numba_device(device: str):
    """
    Select the Numba CUDA device from a string of the form "cuda:N".

    Parameters
    ----------
    device : str
        Must be "cuda:<index>", e.g. "cuda:0" or "cuda:1".
    """
    if device == "cuda":
        cuda.select_device(0)
    else:
        prefix = "cuda:"
        if not device.startswith(prefix):
            raise ValueError(f"Invalid device string {device!r}, expected format 'cuda:N'")
        idx_str = device[len(prefix):]
        try:
            idx = int(idx_str)
        except ValueError:
            raise ValueError(f"Invalid CUDA index {idx_str!r} in {device!r}")
        cuda.select_device(idx)

@cuda.jit
def make_G_v_kernel_th_spin(table_z, table_zb, Nz, G, v):
    i0z, i0h, i0th = cuda.grid(3)
    stride_z, stride_h, stride_th = cuda.gridsize(3)
    Nth, Nh_full, Nqp = table_z.shape
    Nh = Nh_full // 2

    for iz in range(i0z, Nz, stride_z):
        iz1 = iz + Nz
        for ih in range(i0h, Nh + 1, stride_h):
            ihb = 0 if ih == 0 else ih + Nh
            for ith in range(i0th, Nth, stride_th):
                prod = (
                    table_z[ith, ih, iz] * table_zb[ith, ihb, iz]
                    - table_z[ith, ih, iz1] * table_zb[ith, ihb, iz1]
                )
                G[ih, ith, iz] = prod
                if ih == 0:
                    v[ith, iz] = -prod

def build_G_v_th_batch_spin(
    h_mat, hbar_mat, z, zbar, c, hext,
    idx_n, val_n, ofs_n, idx_d, val_d, ofs_d,
    kmax=10, kmax1=10, analytic_H=False
):
    """
    h_mat : (Nth, p)
    Returns G_phys (Nth, Nz, p) and v (Nth, Nz)
    """
    Nth, p = h_mat.size()
    assert h_mat.size() == hbar_mat.size()
    assert len(c) == Nth
    assert len(hext) == Nth

    h_all = torch.cat([torch.zeros((Nth, 1), device=h_mat.device, dtype=h_mat.dtype),
                       h_mat, hbar_mat], dim=1)
    assert z.size() == zbar.size()
    Nz = z.size()[0]

    z1m = 1 - z
    zbar1m = 1 - zbar
    qqbar_all = z_to_q_cuda(torch.concat((z, zbar, z1m, zbar1m)))
    q = qqbar_all[0:Nz]
    qbar = qqbar_all[Nz:2*Nz]
    q1mz = qqbar_all[2*Nz:3*Nz]
    qbar1mz = qqbar_all[3*Nz:]

    q_all = torch.cat([q, q1mz, qbar, qbar1mz], dim=0)
    z_all = torch.cat([z, z1m, zbar, zbar1m], dim=0)
    Nz_tot = len(z_all)

    if kmax <= 4 and analytic_H:
        Harr = build_H_cuda(c, h_all, hext, idx_n, val_n, ofs_n, idx_d, val_d, ofs_d, kmax)
    else:
        Harr = calc_H_rec(c, hext, hext, h_all, kmax + 2)

    h_d = cuda.as_cuda_array(h_all)
    hext_d = cuda.as_cuda_array(hext)
    c_d = cuda.as_cuda_array(c)

    qs_d = cuda.as_cuda_array(q_all)
    zs_d = cuda.as_cuda_array(z_all)
    H_d = cuda.as_cuda_array(Harr)
    tbl_z = cuda.device_array((Nth, 2 * p + 1, Nz_tot), np.float64)

    TPB1 = (2, 8, 8)
    blocks1 = (
        math.ceil(Nz_tot / TPB1[0]),
        math.ceil((p + 1) / TPB1[1]),
        math.ceil(Nth / TPB1[2]),
    )

    block_table_kernel[blocks1, TPB1](
        qs_d, zs_d, h_d, H_d, hext_d, c_d, kmax, kmax1, tbl_z
    )
    cuda.synchronize()

    G_big = cuda.device_array((p + 1, Nth, Nz), np.float64)
    v_d = cuda.device_array((Nth, Nz), np.float64)

    tbl_zb = tbl_z[:, :, 2 * Nz:]
    tbl_z = tbl_z[:, :, :2 * Nz]

    make_G_v_kernel_th_spin[blocks1, TPB1](tbl_z, tbl_zb, np.int32(Nz), G_big, v_d)
    cuda.synchronize()

    G_all = torch.as_tensor(G_big)
    v = torch.as_tensor(v_d)

    G_phys = G_all[1:].reshape(Nth, p, Nz).permute(0, 2, 1)
    return G_phys, v

def calculate_c_rew_Virasoro(
    h, hbar, z, zbar, hext, c_val,
    N_lsq=20, kmax=2, kmax1=10,
    device="cuda", analytic_H=False
):
    idx_n, val_n, ofs_n, idx_d, val_d, ofs_d = coeffs
    N_z = len(z)
    N_th = len(h)
    assert len(c_val) == N_th
    assert len(hext) == N_th
    assert N_z % N_lsq == 0, "N_z should be integer times N_lsq"
    N_stat = N_z // N_lsq
    N_state = len(h[0])

    G, v = build_G_v_th_batch_spin(
        h, hbar, z, zbar, c_val, hext,
        idx_n, val_n, ofs_n, idx_d, val_d, ofs_d,
        kmax=kmax, kmax1=kmax1, analytic_H=analytic_H
    )

    G = torch.as_tensor(G, device=device)
    G = G.view((N_th, N_stat, N_lsq, N_state)).permute(1, 0, 2, 3)

    v = torch.as_tensor(v, device=device)
    v = v.view((N_th, N_stat, N_lsq)).permute(1, 0, 2)

    c = torch.linalg.lstsq(G, v).solution
    Gc = torch.einsum('sgzn,sgn->sgz', G, c)
    residual_vector = Gc - v
    residual = torch.einsum('sgz,sgz->sg', residual_vector, residual_vector)

    return c, residual

def least_sq_std_rew_Virasoro(
    h_vals, hb_vals, h_ext,
    x_vals, y_vals,
    central_charge, 
    N_lsq=20, kmax=2, n_states_rew=2,
    kmax1=10, device="cuda", analytic_H=False
):
    set_numba_device(device)
    cs, rews = calculate_c_rew_Virasoro(
        h_vals, hb_vals, x_vals, y_vals, h_ext, central_charge,
        N_lsq=N_lsq, kmax=kmax,
        kmax1=kmax1, device=device, analytic_H=analytic_H
    )
    c_mean = torch.mean(cs, 0)
    c_std = torch.std(cs, 0)
    r_stat = c_std / c_mean
    r = -torch.sum(torch.log(torch.abs(r_stat[:, :n_states_rew])), dim=1)

    return r, c_mean, c_std
