import math
import numpy as np
import torch

# For zero‐copy bridging between Torch and Numba/CuPy
from numba import cuda

from ..virasoro.virasoro_block import build_virasoro_block_cross, load_virasoro_coeffs
from ..virasoro.virasoro_rec import calc_H_rec
from ..global_block import torch_to_numba_devicearray, numba_devicearray_to_torch, compute_G_element, compute_g_delta_kernel

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
#               HPC Reward Functions of Monte Carlo paper
###############################################################################
@cuda.jit
def compute_W_v_MC(d_max, x_arr, y_arr, dphi, W, v):
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

def calculate_c_rew_MC(d_vals, s_vals, x_vals, y_vals, dphi, d_max=9.0, N_lsq=32):
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
    compute_W_v_MC[(cx, cy, cz), threads_2](d_max, x_nb, y_nb, dphi, W_dev, v_dev)

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

def least_sq_std_rew_MC_Global(h_vals, hb_vals, h_ext, x_vals, y_vals, N_lsq=20, n_states_rew=2):
    d_vals = (h_vals + hb_vals)
    s_vals = (h_vals - hb_vals)
    d_max = torch.max(d_vals).item()
    c_out, res_out = calculate_c_rew_MC(d_vals, s_vals, x_vals, y_vals, 2*h_ext, d_max, N_lsq=N_lsq)
    R_full = torch.sum(res_out, dim=0)               # (N_deltas,)

    # 4) 论文里要归一化 1/N_z
    N_stat = res_out.shape[0]
    N_z = x_vals.shape[0]      # 必须等于 N_stat * N_lsq
    assert N_z == N_stat * N_lsq
    S_full = R_full / float(N_z)                     # (N_deltas,)

    # 5) 最终 reward = exp(S_full)
    reward = torch.exp(S_full)                        # (N_deltas,)

    return reward, S_full


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

def least_sq_std_rew_Hybrid(h_vals, hb_vals, h_ext, x_vals, y_vals, virasoro_info, N_lsq=20, n_states_rew=2):
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
    assert len(hext) == N_th
    assert N_z % N_lsq == 0, "N_z should be integer times N_lsq"
    N_stat = N_z // N_lsq
    N_state = len(h)

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
    virasoro_info, 
    N_lsq=20, kmax=2, n_states_rew=2,
    kmax1=10, device="cuda", analytic_H=False
):
    set_numba_device(device)
    central_charge=virasoro_info["central_charge"]-1e-5
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

def least_sq_std_rew_MC_Virasoro(h_vals, hb_vals, h_ext, x_vals, y_vals, central_charge, 
                                    N_lsq=20, kmax=2,
                                    kmax1=10, device="cuda", analytic_H=False):
    d_vals = (h_vals + hb_vals)
    s_vals = (h_vals - hb_vals)
    d_max = torch.max(d_vals).item()
    c_out, res_out = calculate_c_rew_Virasoro(
        h_vals, hb_vals, x_vals, y_vals, h_ext, central_charge,
        N_lsq=N_lsq, kmax=kmax,
        kmax1=kmax1, device=device, analytic_H=analytic_H)
    R_full = torch.sum(res_out, dim=0)               # (N_deltas,)

    # 4) 论文里要归一化 1/N_z
    N_stat = res_out.shape[0]
    N_z = x_vals.shape[0]      # 必须等于 N_stat * N_lsq
    assert N_z == N_stat * N_lsq
    S_full = R_full / float(N_z)                     # (N_deltas,)

    # 5) 最终 reward = exp(S_full)
    reward = torch.exp(S_full)                        # (N_deltas,)

    return reward, S_full


"""
GPU Elliptic K with Numba-CUDA + PyTorch
---------------------------------------

Implements  K(m) = π / (2 AGM(1, √(1-m))).
Gradient dK/dm is provided so you can wrap it in a
torch.autograd.Function if needed.
"""
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