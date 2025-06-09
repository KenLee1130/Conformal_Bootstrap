# -*- coding: utf-8 -*-
"""
Created on Fri May  2 23:15:23 2025

@author: User
"""
import math
import torch
import numpy as np
import cupy as cp
import numba
from numba import cuda
from pathlib import Path
from virasoro.virasoro_block import build_virasoro_block_cross, load_coeffs_packed_split

from env import _2F1_device, torch_to_numba_devicearray, numba_devicearray_to_torch

def generate_gaussian_complex_points(n, mean=0.5, std=0.1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    real = torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64)
    imag = torch.normal(mean=0.0, std=std, size=(n,),device=device,dtype=torch.float64)

    return torch.complex(real, imag)

def generate_gaussian_complex_points_v1(n, mean=0.5, std=0.1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    x = torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64)
    y = torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64)

    return x,y

            
@cuda.jit(device=True)
def I_ising(x, y):
    sqrt_x = math.sqrt(x)
    sqrt_y = math.sqrt(y)

    term1 = math.sqrt(1.0 - sqrt_x) + math.sqrt(1.0 + sqrt_x)
    term2 = math.sqrt(1.0 - sqrt_y) + math.sqrt(1.0 + sqrt_y)
    denom = 4.0 * (1.0 - x)**(1.0/8.0) * (1.0 - y)**(1.0/8.0)

    return (term1 * term2) / denom

@cuda.jit
def compute_W_v_TIsing(x_arr, y_arr, dphi, v):
    # Get 3D thread indices.
    i, j, k = cuda.grid(3)
    G0, G1, G2 = cuda.gridsize(3)
    idx = i + j * G0 + k * G0 * G1
    if idx < x_arr.size:
        x_val = x_arr[idx]
        y_val = y_arr[idx]
        r1_sq = (1.0 - x_val)*(1.0 - y_val)
        r2_sq = x_val*y_val 
        
        v[idx] = ((r1_sq**dphi)*I_ising(x_val, y_val)-(r2_sq**dphi)*I_ising(1.0-x_val, 1.0-y_val))

def get_V_block(x_vals: torch.Tensor,
                y_vals: torch.Tensor,
                dphi: float,
                N_lsq: int = 32) -> torch.Tensor:
    """
    输入:
      x_vals, y_vals:  shape (N_z,) 的 CUDA float64 Tensor
      dphi:             标量
      N_lsq:            每组统计的点数

    返回:
      v: shape (N_stat, 1, N_lsq) 的 Tensor，
         其中 N_stat = N_z // N_lsq
    """
    assert x_vals.is_cuda and y_vals.is_cuda
    assert x_vals.dtype == torch.float64 and y_vals.dtype == torch.float64

    Nz = x_vals.numel()
    N_stat = Nz // N_lsq

    # 1) 转为 Numba device array
    x_nb = cuda.as_cuda_array(x_vals)
    y_nb = cuda.as_cuda_array(y_vals)

    # 2) 分配输出
    v_dev = cuda.device_array(Nz, dtype=np.float64)

    # 3) 启动 kernel
    TPB    = 128
    blocks = (Nz + TPB - 1) // TPB
    compute_W_v_TIsing[blocks, TPB](x_nb, y_nb, dphi, v_dev)

    # 4) 拷回 PyTorch tensor
    v_t = numba_devicearray_to_torch(v_dev)   # shape (Nz,)

    return v_t

if __name__ == "__main__":
    JSON_PATH = Path("C:/Users/User/Desktop/git/ConformalBootstrapRL/packages/rl/virasoro/virasoro_coeffs.json")
    coeffs=load_coeffs_packed_split(JSON_PATH)
    
    device = "cuda"
    dphi=1/8
    virasoro_info={'central_charge':0.5-(1e-5), "V_c":[1], "V_d":[0.]}
    central_charge=virasoro_info["central_charge"]
    V_c=torch.tensor(virasoro_info["V_c"], device=device,dtype=torch.float64).T
    V_d=torch.tensor(virasoro_info["V_d"], device=device,dtype=torch.float64)
    
    x, y = generate_gaussian_complex_points_v1(10)
    print(f"x: {x}\ny: {y}")
    print("Recursion: ", build_virasoro_block_cross(V_d, x, y, central_charge, dphi/2, coeffs, kmax=3))
    print("Analytic: ", get_V_block(x, y, dphi))