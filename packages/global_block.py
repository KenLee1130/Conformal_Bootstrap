# global_blocks.py

import math
import numpy as np
import torch

# For zero‐copy bridging between Torch and Numba/CuPy
import cupy as cp
import numba
from numba import cuda

import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

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
    """
    将一个 GPU 上的 torch.Tensor (float64) 零拷贝 (via DLPack) 转成 numba.cuda 的 DeviceNDArray。
    """
    if not t.is_cuda:
        raise ValueError("Tensor 必须在 GPU 上。")
    if not t.is_contiguous():
        t = t.contiguous()
    dlpack = torch.utils.dlpack.to_dlpack(t)
    cupy_arr = cp.from_dlpack(dlpack)
    return numba.cuda.as_cuda_array(cupy_arr)

def numba_devicearray_to_torch(arr: numba.cuda.cudadrv.devicearray.DeviceNDArray):
    """
    将 numba.cuda DeviceNDArray（GPU 上的 CuPy 底层数据）转回 torch.Tensor。
    """
    cupy_arr = cp.asarray(arr)
    dlpack = cupy_arr.toDlpack()
    return torch.utils.dlpack.from_dlpack(dlpack)

###############################################################################
#                     HPC Kernels in Float64 (global blocks)
###############################################################################
@cuda.jit(device=True)
def _2F1_device(a, b, c, zr, zi):
    """
    Truncated hypergeometric expansion (2F1) with complex operations in float64.
    zr + i*zi 是复数 z。返回 real, imag 两个分量。
    """
    if c == 0.0:
        return 1.0, 0.0
    real_accum = 1.0
    imag_accum = 0.0
    term_r = 1.0
    term_i = 0.0
    for n in range(1, MAX_TERMS):
        denom = n * (c + n - 1.0)
        poch = ((a + n - 1.0) * (b + n - 1.0)) / denom
        # (term_r + i*term_i) * (zr + i*zi) = (new_r + i*new_i)
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
    """
    计算单个 Δ (d_val)、s_val、复数 z = x + i*y 对应的 2F1 组合。
    h = (Δ + s)/2, hb = (Δ - s)/2。
    按照 Cross-channel 套用 hypergeometric，
    返回 real, imag 两部分的 global block 值。
    """
    h  = 0.5 * (d_val + s_val)
    hb = 0.5 * (d_val - s_val)
    # _2F1_device(a, b, c, zr, zi) 中 zr + i*zi
    # 对应四个 2F1:
    fhz_r, fhz_i         = _2F1_device(h,  h,  2.0 * h,  x,  y)
    fhbz_b_r, fhbz_b_i   = _2F1_device(hb, hb, 2.0 * hb, x, -y)
    fhz_b_r, fhz_i_temp  = _2F1_device(h,  h,  2.0 * h,  x, -y)
    fhb_z_r, fhb_z_i     = _2F1_device(hb, hb, 2.0 * hb, x,  y)

    # 将结果合并成 z^h+hb * [ step1 + step2 ] 形式：
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    d_ = h + hb
    s_ = h - hb
    r_pow_d = r ** d_
    cos_s_th = math.cos(s_ * theta)
    sin_s_th = math.sin(s_ * theta)

    # 第一部分：2F1(h,h;2h; z)*2F1(hb,hb;2hb; conj(z))
    step1_r = fhz_r * fhbz_b_r - fhz_i * fhbz_b_i
    step1_i = fhz_r * fhbz_b_i + fhz_i * fhbz_b_r
    tmp1_r  = step1_r * cos_s_th - step1_i * sin_s_th
    tmp1_i  = step1_r * sin_s_th + step1_i * cos_s_th
    T1_r = r_pow_d * tmp1_r
    T1_i = r_pow_d * tmp1_i

    # 第二部分：2F1(h,h;2h; conj(z))*2F1(hb,hb;2hb; z)
    step2_r = fhz_b_r * fhb_z_r - fhz_i_temp * fhb_z_i
    step2_i = fhz_b_r * fhb_z_i + fhz_i_temp * fhb_z_i
    tmp2_r  = step2_r * cos_s_th + step2_i * sin_s_th
    tmp2_i  = -step2_r * sin_s_th + step2_i * cos_s_th
    T2_r = r_pow_d * tmp2_r
    T2_i = r_pow_d * tmp2_i

    return T1_r + T2_r, T1_i + T2_i

@cuda.jit(device=True)
def compute_G_element(d_val, s_val, x, y, dphi):
    """
    计算 cross-channel 中的 G 元素：G(d_val, s_val; z, z̄, dphi)
    xp = 1 - x, yp = -y 表示 conj(1-z)
    返回 r1^2φ * g(h,hb; z) - r2^2φ * g(h,hb; 1-z)，
    r1 = |1-z|, r2 = |z|.
    """
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
    """
    GPU kernel (3D grid)：
    - d_arr: shape (N_deltas, N_state)
    - s_arr: shape (N_state,)
    - x_arr, y_arr: shape (N_z,)
    - dphi: 标量
    - g_delta_mat: 预分配的输出，shape (N_deltas, N_z, N_state)
    """
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

