# fd_adam.py

import torch
from typing import List, Tuple, Optional

import cupy as cp
import numba
from numba import cuda

from ..rl.make_shift import make_shifted_h1
from ..rl.env_th import least_sq_std_rew_and_c_th
from ..virasoro.virasoro_rec import load_coeffs_packed_split
import os
from pathlib import Path

this_file = Path(__file__).resolve()
adam_dir  = this_file.parent
project_root = adam_dir.parent
virasoro_dir = project_root / "virasoro"
JSON_PATH=os.path.join(virasoro_dir, Path("virasoro_coeffs.json"))
COEFFS = load_coeffs_packed_split(JSON_PATH)

class SpectrumBuffer:
    """
    用 GPU (CuPy) 并行化存储和更新“好”的 spectrum。
    - spectra 以 CuPy ndarray 存储在 GPU 上，形状 (n_buf, d)。
    - 当插入一个新的 spectrum 时，计算它与 buffer 中所有条目的欧氏距离（在 GPU 上完成）：
        • 如果最小距离 < threshold，则比较 reward：若新 reward 更高，则替换对应条目；
          否则放弃。
        • 如果所有距离 ≥ threshold，则将新 spectrum 追加到 buffer 末尾。
    """

    def __init__(self, threshold: float):
        """
        Args:
            threshold: 相似度判定的欧氏距离阈值
        """
        self.threshold = threshold
        # spectra_gpu 存储在 GPU 上的 CuPy 数组，初始为 None
        self.spectra_gpu: Optional[cp.ndarray] = None  # shape (n_buf, d), dtype float64
        self.rewards: List[float] = []

    def insert(self, spectrum: torch.Tensor, reward: float):
        """
        尝试将 (spectrum, reward) 插入 buffer。
        Args:
            spectrum: torch.Tensor, 形状 (d,), 在 GPU 上
            reward: float
        """
        # 将 torch.Tensor (GPU) 转为 CuPy ndarray（零拷贝 via DLPack）
        dlpack = torch.utils.dlpack.to_dlpack(spectrum.contiguous())
        spec_cp: cp.ndarray = cp.fromDlpack(dlpack)  # shape (d,), dtype float64

        if self.spectra_gpu is None:
            # buffer 为空，直接创建
            self.spectra_gpu = spec_cp[None, :].copy()  # shape (1, d)
            self.rewards = [reward]
            return

        # 计算与现有所有条目的距离
        # self.spectra_gpu: (n_buf, d), spec_cp: (d,)
        diffs = self.spectra_gpu - spec_cp[None, :]          # (n_buf, d)
        dists = cp.linalg.norm(diffs, axis=1)                # (n_buf,)

        # 找到最小距离及其索引
        idx_min = int(cp.argmin(dists))
        min_dist = float(dists[idx_min])

        if min_dist < self.threshold:
            # 与某条目“相似”，比较 reward
            if reward > self.rewards[idx_min]:
                # 替换该条目
                self.spectra_gpu[idx_min] = spec_cp.copy()
                self.rewards[idx_min] = reward
            # 否则不插入
        else:
            # 与所有条目都不相似，追加到 buffer
            self.spectra_gpu = cp.concatenate([self.spectra_gpu, spec_cp[None, :]], axis=0)
            self.rewards.append(reward)

    def get_all(self) -> List[torch.Tensor]:
        """
        返回所有存储的 spectra 列表，每个元素为 torch.Tensor (d,) 在 GPU 上。
        """
        if self.spectra_gpu is None:
            return []
        out_list: List[torch.Tensor] = []
        for i in range(self.spectra_gpu.shape[0]):
            arr_cp = self.spectra_gpu[i]  # (d,)
            dlpack = arr_cp.toDlpack()
            tensor = torch.utils.dlpack.from_dlpack(dlpack).to(device="cuda").clone()
            out_list.append(tensor)
        return out_list

    def get_rewards(self) -> List[float]:
        """返回所有存储的 rewards 列表。"""
        return self.rewards.copy()

    def __len__(self):
        return 0 if self.spectra_gpu is None else self.spectra_gpu.shape[0]


class FiniteDiffAdam:
    """
    GPU-parallelized finite-difference + Adam optimizer。

    使用流程：
        fd_adam.COEFFS = load_coeffs_packed_split("virasoro_coeffs.json")
        optimizer = FiniteDiffAdam(
            h_step_obs=1e-3,
            fixed_vars=[1],
            z=z_tensor,
            zbar=zbar_tensor,
            N_lsq=20,
            kmax=10,
            n_states_rew=3,
            device='cuda'
        )
        best_X, best_R = optimizer.optimize(
            X0=initial_X,       # torch.Tensor (B, d)
            x_lo=x_lo_tensor,   # torch.Tensor (d,)
            x_hi=x_hi_tensor,   # torch.Tensor (d,)
            lr=1e-2,
            num_steps=50
        )
    """

    def __init__(
        self,
        h_step_obs: float,
        fixed_vars: List[int],
        z: torch.Tensor,
        zbar: torch.Tensor,
        N_lsq: int = 20,
        kmax: int = 10,
        n_states_rew: int = 2,
        device: str = "cuda",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """
        Args:
            h_step_obs: 有限差分步长 ε
            fixed_vars: 哪些维度索引要固定不动 (0-based list)
            z: torch.Tensor, 用于 least_sq_std_rew_and_c_th (N_z,)
            zbar: torch.Tensor, 与 z 对应
            N_lsq: least-squares 内部参数
            kmax: least_sq_std_rew_and_c_th 内部迭代次数
            n_states_rew: least_sq_std_rew_and_c_th 返回 reward 时用的状态数
            device: 'cuda' 或 'cpu'
            beta1, beta2, eps: Adam 超参数
        """
        if COEFFS is None:
            raise RuntimeError("全局 COEFFS 未定义，请先设置 COEFFS 后再使用 FiniteDiffAdam。")
        self.h_step = h_step_obs
        self.fixed_vars = fixed_vars.copy()
        self.z = z.to(device)
        self.zbar = zbar.to(device)
        self.N_lsq = N_lsq
        self.kmax = kmax
        self.n_states_rew = n_states_rew
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.unfixed: Optional[List[int]] = None

    @torch.no_grad()
    def optimize(
        self,
        X0: torch.Tensor,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        lr: float,
        num_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        并行运行 Adam + 有限差分 梯度上升，返回每条轨迹的最优位置与对应 reward。

        Args:
            X0: 初始点矩阵，torch.Tensor of shape (B, d)，dtype=torch.float64
            x_lo: (d,) 下界
            x_hi: (d,) 上界
            lr: 学习率
            num_steps: 迭代步数

        Returns:
            best_X: torch.Tensor (B, d)，历史最优点
            best_R: torch.Tensor (B,) ，对应 best_X 的 reward
        """
        X = X0.to(self.device).clone()  # (B, d)
        x_lo = x_lo.to(self.device)
        x_hi = x_hi.to(self.device)

        B, d = X.shape
        # 计算自由维度列表
        all_dims = set(range(d))
        for idx in self.fixed_vars:
            if idx < 0 or idx >= d:
                raise IndexError(f"fixed_vars 中的索引 {idx} 超出范围 (0..{d-1})")
        self.unfixed = sorted(list(all_dims - set(self.fixed_vars)))
        du = len(self.unfixed)

        # 初始化 Adam 动量
        M = torch.zeros_like(X, device=self.device)  # (B, d)
        V = torch.zeros_like(X, device=self.device)  # (B, d)
        t = 0

        # 记录每条轨迹的历史最优
        best_X = X.clone()                   # (B, d)
        best_R = self._batch_reward(X)       # (B,)

        for iteration in range(1, num_steps + 1):
            t += 1
            # 1) 并行有限差分估计梯度
            H = make_shifted_h1(X, self.h_step, self.fixed_vars)  # (B*(1+2du), d)
            rew_flat, _, _ = least_sq_std_rew_and_c_th(
                H, self.z, self.zbar, COEFFS,
                N_lsq=self.N_lsq,
                kmax=self.kmax,
                n_states_rew=self.n_states_rew,
                device=self.device
            )
            reshaped = rew_flat.view(B, 1 + 2 * du)  # (B, 1+2du)

            G = torch.zeros_like(X, device=self.device)  # (B, d)
            # 利用向量化更新所有自由维度
            plus_indices = [1 + 2 * i for i in range(du)]
            minus_indices = [idx + 1 for idx in plus_indices]
            # 把 reshaped[:, plus_indices] 和 reshaped[:, minus_indices] 分别提取出来
            plus_vals = reshaped[:, plus_indices]   # (B, du)
            minus_vals = reshaped[:, minus_indices] # (B, du)
            grads = (plus_vals - minus_vals) / (2 * self.h_step)  # (B, du)
            # 将 grads 填回 G
            for i_dim, dim in enumerate(self.unfixed):
                G[:, dim] = grads[:, i_dim]
            # fixed_vars 的列保持 0

            # 2) Adam 更新
            M = self.beta1 * M + (1 - self.beta1) * G
            V = self.beta2 * V + (1 - self.beta2) * (G * G)

            m_hat = M / (1 - self.beta1 ** t)
            v_hat = V / (1 - self.beta2 ** t)

            step = lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            X = X + step
            X = torch.max(torch.min(X, x_hi), x_lo)

            # 3) 更新历史最优
            R_new = self._batch_reward(X)   # (B,)
            improved = R_new > best_R       # (B,) 布尔
            if improved.any():
                best_R[improved] = R_new[improved]
                best_X[improved, :] = X[improved, :]

        return best_X, best_R

    def _batch_reward(self, X: torch.Tensor) -> torch.Tensor:
        """
        批量计算 X (B, d) 的 reward (B,)，调用 least_sq_std_rew_and_c_th。
        """
        B, d = X.shape
        H_full = make_shifted_h1(X, self.h_step, self.fixed_vars)  # (B*(1+2du), d)
        rew_full, _, _ = least_sq_std_rew_and_c_th(
            H_full, self.z, self.zbar, COEFFS,
            N_lsq=self.N_lsq,
            kmax=self.kmax,
            n_states_rew=self.n_states_rew,
            device=self.device
        )
        du = len(self.unfixed)
        rew_mat = rew_full.view(B, 1 + 2 * du)  # (B, 1+2du)
        return rew_mat[:, 0]  # 中心点的 reward


class ParallelSpectrumOptimizer:
    """
    集成 FiniteDiffAdam 和 SpectrumBuffer 的并行搜索器。
    - 多条 trajectory 并行运行 Adam+有限差分。
    - 每条 trajectory 在指定步数结束后，将它的历史最优点插入 SpectrumBuffer。
    - 然后为该轨迹重新随机初始化一个 spectrum，进入下一轮迭代（wave）。
    - 循环执行多个 wave，直至达到 num_waves。
    """

    def __init__(
        self,
        d: int,
        h_step_obs: float,
        fixed_vars: List[int],
        z: torch.Tensor,
        zbar: torch.Tensor,
        N_lsq: int = 20,
        kmax: int = 10,
        n_states_rew: int = 3,
        device: str = "cuda",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        buffer_threshold: float = 1e-2,
        use_sobol: bool = True,
        sobol_seed: Optional[int] = None
    ):
        """
        Args:
            d: 状态空间维度 (spectrum 长度)
            h_step_obs: 有限差分步长 ε
            fixed_vars: 哪些维度固定不动
            z, zbar: 用于 least_sq_std_rew_and_c_th
            N_lsq, kmax, n_states_rew: least_sq_std_rew_and_c_th 超参数
            device: 'cuda' 或 'cpu'
            beta1, beta2, eps: Adam 超参数
            buffer_threshold: buffer 中相似度阈值
            use_sobol: 是否使用 Sobol 采样初始化
            sobol_seed: Sobol 随机种子
        """
        if COEFFS is None:
            raise RuntimeError("全局 COEFFS 未定义，请先设置 COEFFS 后再使用 ParallelSpectrumOptimizer。")

        self.d = d
        self.device = device if torch.cuda.is_available() else "cpu"
        self.h_step_obs = h_step_obs
        self.fixed_vars = fixed_vars.copy()
        self.z = z.to(self.device)
        self.zbar = zbar.to(self.device)
        self.N_lsq = N_lsq
        self.kmax = kmax
        self.n_states_rew = n_states_rew
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # 存储“好”的 spectrum
        self.buffer = SpectrumBuffer(threshold=buffer_threshold)

        # Sobol 采样（用于初始化）
        self.use_sobol = use_sobol
        if self.use_sobol:
            from Adam.Sobol import SobolSampler
            self.sobol = SobolSampler(dim=d, scramble=True, seed=sobol_seed)

    def _random_init(self, B: int, x_lo: torch.Tensor, x_hi: torch.Tensor) -> torch.Tensor:
        """
        返回 B 个 (d,) 随机初始点：如果 use_sobol=True，使用 Sobol 采样，否则均匀随机。
        """

        if self.use_sobol:
            pts_np = self.sobol.sample(n=B, lo=x_lo.cpu().numpy(), hi=x_hi.cpu().numpy())
            return torch.from_numpy(pts_np).to(self.device).type(torch.float64)
        else:
            return (x_lo + (x_hi - x_lo) * torch.rand((B, self.d), device=self.device, dtype=torch.float64))

    def run(
        self,
        B: int,
        x_lo: torch.Tensor,
        x_hi: torch.Tensor,
        lr: float,
        steps_per_wave: int,
        num_waves: int
    ):
        """
        多轮并行 Adam+有限差分搜索：
          for wave in range(num_waves):
             1) 随机初始化 B 条 spectrum → X0 (B, d)
             2) 新建 FiniteDiffAdam，调用 optimize(X0, x_lo, x_hi, lr, steps_per_wave)
             3) 得到 best_X (B, d), best_R (B,)
             4) 将每个 best_X[b], best_R[b] 插入 SpectrumBuffer
        """
        x_lo = x_lo.to(self.device).type(torch.float64)
        x_hi = x_hi.to(self.device).type(torch.float64)

        for wave in range(num_waves):
            # 1) 随机初始化
            X0 = self._random_init(B, x_lo, x_hi)  # (B, d)

            # 2) 实例化 FiniteDiffAdam
            adam_opt = FiniteDiffAdam(
                h_step_obs=self.h_step_obs,
                fixed_vars=self.fixed_vars,
                z=self.z, zbar=self.zbar,
                N_lsq=self.N_lsq, kmax=self.kmax, n_states_rew=self.n_states_rew,
                device=self.device, beta1=self.beta1, beta2=self.beta2, eps=self.eps
            )

            # 3) 运行 optimize
            best_X, best_R = adam_opt.optimize(
                X0=X0,
                x_lo=x_lo,
                x_hi=x_hi,
                lr=lr,
                num_steps=steps_per_wave
            )

            # 4) 插入 buffer
            for b in range(B):
                spec_b = best_X[b]       # torch.Tensor (d,), 在 GPU
                rew_b = float(best_R[b].item())
                self.buffer.insert(spec_b, rew_b)

    def get_buffer(self) -> Tuple[List[torch.Tensor], List[float]]:
        """
        返回 buffer 中所有 spectrum 列表（torch.Tensor, 在 GPU）及对应的 rewards 列表。
        """
        return self.buffer.get_all(), self.buffer.get_rewards()

