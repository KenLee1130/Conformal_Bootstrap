# sobol_sampler.py

import numpy as np
from scipy.stats import qmc
import torch


class SobolSampler:
    """
    Sobol 序列采样器，用于在 [x_lo, x_hi]^d 空间中均匀生成样本点。

    用法示例：
        sampler = SobolSampler(dim=9, scramble=True)
        # 返回 NumPy 数组 shape (n, d)
        pts_np = sampler.sample(n=64, lo=np.zeros(9), hi=np.ones(9))
        # 如果需要 Torch Tensor：
        pts_torch = torch.from_numpy(pts_np).to(device)
    """

    def __init__(self, dim: int, scramble: bool = True, seed: int = None):
        """
        Args:
            dim: 采样空间维度 d
            scramble: 是否对 Sobol 序列进行筛选扰动（默认 True）
            seed: 随机种子（可选），仅在 scramble=True 时生效
        """
        self.dim = dim
        self.sampler = qmc.Sobol(d=dim, scramble=scramble, seed=seed)

    def sample(self, n: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """
        在 [lo, hi]^d 区间内生成 n 个 Sobol 点。

        Args:
            n: 需要生成的样本数量
            lo: NumPy 数组，形状 (d,) 表示各维度下界
            hi: NumPy 数组，形状 (d,) 表示各维度上界

        Returns:
            pts: NumPy 数组，形状 (n, d)，每行是一个采样点
        """
        if lo.shape != (self.dim,) or hi.shape != (self.dim,):
            raise ValueError(f"lo 和 hi 必须是长度为 {self.dim} 的一维数组")
        # Sobol 输出在 [0,1]^d
        raw = self.sampler.random(n)  # shape (n, d), dtype float64
        # 线性映射到 [lo, hi]
        return lo + raw * (hi - lo)


# ===================== 使用示例 =====================
if __name__ == "__main__":
    import torch

    dim = 8
    n = 1
    x_lo = np.zeros(dim, dtype=np.float64)
    x_hi = np.ones(dim, dtype=np.float64) * 2.0

    sampler = SobolSampler(dim=dim, scramble=True, seed=42)
    pts_np = sampler.sample(n=n, lo=x_lo, hi=x_hi)
    print("Sobol points (NumPy):", pts_np)

    # 如果需要直接生成 Torch Tensor：
    pts_torch = torch.from_numpy(pts_np).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Sobol points (Torch):", pts_torch)
