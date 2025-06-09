import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .env import least_sq_std_rew, least_sq_std_rew_W_func
from .utils import plot_r_gauss_z, generate_random_points, generate_gaussian_complex_points
from ..z_sampling.z_weight_net import WNet

# Global dtype setting.
dtype = torch.float64
torch.set_default_dtype(dtype)

###############################################################################
# Replay Buffer with Preallocated Tensors (GPU optimized)
###############################################################################
class GPUReplayBuffer:
    """
    A replay buffer that preallocates tensors on the target device.
    This minimizes CPU-GPU transfers and avoids per-sample unsqueeze calls.
    """
    def __init__(self, capacity=1_000_000, device=torch.device("cpu")):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.pos = 0
        # Buffers will be allocated when the first batch is pushed.
        self.ds_buffer = None
        self.s_buffer = None
        self.a_buffer = None
        self.r_buffer = None
        self.ns_buffer = None
        self.d_buffer = None

    def push(self, ds, s, a, r, ns, d):
        """Stores a batch of transitions."""
        batch_size = s.shape[0]
        if self.s_buffer is None:
            ds_shape = ds.shape[1:]
            s_shape = s.shape[1:]
            a_shape = a.shape[1:]
            r_shape = r.shape[1:]
            ns_shape = ns.shape[1:]
            d_shape = d.shape[1:]
            self.ds_buffer = torch.empty((self.capacity, *ds_shape), device=self.device, dtype=s.dtype)
            self.s_buffer = torch.empty((self.capacity, *s_shape), device=self.device, dtype=s.dtype)
            self.a_buffer = torch.empty((self.capacity, *a_shape), device=self.device, dtype=a.dtype)
            self.r_buffer = torch.empty((self.capacity, *r_shape), device=self.device, dtype=r.dtype)
            self.ns_buffer = torch.empty((self.capacity, *ns_shape), device=self.device, dtype=ns.dtype)
            self.d_buffer = torch.empty((self.capacity, *d_shape), device=self.device, dtype=d.dtype)
        remain = self.capacity - self.pos
        if batch_size <= remain:
            self.ds_buffer[self.pos:self.pos+batch_size].copy_(ds.to(self.device))
            self.s_buffer[self.pos:self.pos+batch_size].copy_(s.to(self.device))
            self.a_buffer[self.pos:self.pos+batch_size].copy_(a.to(self.device))
            self.r_buffer[self.pos:self.pos+batch_size].copy_(r.to(self.device))
            self.ns_buffer[self.pos:self.pos+batch_size].copy_(ns.to(self.device))
            self.d_buffer[self.pos:self.pos+batch_size].copy_(d.to(self.device))
        else:
            self.ds_buffer[self.pos:self.capacity].copy_(ds[:remain].to(self.device))
            self.s_buffer[self.pos:self.capacity].copy_(s[:remain].to(self.device))
            self.a_buffer[self.pos:self.capacity].copy_(a[:remain].to(self.device))
            self.r_buffer[self.pos:self.pos+remain].copy_(r[:remain].to(self.device))
            self.ns_buffer[self.pos:self.capacity].copy_(ns[:remain].to(self.device))
            self.d_buffer[self.pos:self.capacity].copy_(d[:remain].to(self.device))
            extra = batch_size - remain
            self.ds_buffer[0:extra].copy_(ds[remain:].to(self.device))
            self.s_buffer[0:extra].copy_(s[remain:].to(self.device))
            self.a_buffer[0:extra].copy_(a[remain:].to(self.device))
            self.r_buffer[0:extra].copy_(r[remain:].to(self.device))
            self.ns_buffer[0:extra].copy_(ns[remain:].to(self.device))
            self.d_buffer[0:extra].copy_(d[remain:].to(self.device))
        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """Randomly samples a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        ds = self.ds_buffer[indices]
        s = self.s_buffer[indices]
        a = self.a_buffer[indices]
        r = self.r_buffer[indices]
        ns = self.ns_buffer[indices]
        d = self.d_buffer[indices]
        return ds, s, a, r, ns, d

    def __len__(self):
        return self.size

###############################################################################
# Prioritized Replay Buffer with Preallocated Tensors
###############################################################################
class PrioritizedReplayBuffer:
    """
    Replay buffer with prioritized sampling.
    This buffer uses a preallocated tensor for transitions and stores
    a priority value for each transition.
    """
    def __init__(self, capacity, alpha=0.6, device=torch.device("cpu")):
        self.capacity = capacity
        self.alpha = alpha
        self.device = device
        self.size = 0
        self.pos = 0
        self.ds_buffer = None
        self.s_buffer = None
        self.a_buffer = None
        self.r_buffer = None
        self.ns_buffer = None
        self.d_buffer = None
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=self.device)

    def push(self, ds, s, a, r, ns, d):
        """Stores a batch of transitions with initial priority."""
        batch_size = s.shape[0]
        if self.s_buffer is None:
            ds_shape = ds.shape[1:]
            s_shape = s.shape[1:]
            a_shape = a.shape[1:]
            r_shape = r.shape[1:]
            ns_shape = ns.shape[1:]
            d_shape = d.shape[1:]
            self.ds_buffer = torch.empty((self.capacity, *ds_shape), device=self.device, dtype=s.dtype)
            self.s_buffer = torch.empty((self.capacity, *s_shape), device=self.device, dtype=s.dtype)
            self.a_buffer = torch.empty((self.capacity, *a_shape), device=self.device, dtype=a.dtype)
            self.r_buffer = torch.empty((self.capacity, *r_shape), device=self.device, dtype=r.dtype)
            self.ns_buffer = torch.empty((self.capacity, *ns_shape), device=self.device, dtype=ns.dtype)
            self.d_buffer = torch.empty((self.capacity, *d_shape), device=self.device, dtype=d.dtype)
        remain = self.capacity - self.pos
        if batch_size <= remain:
            self.ds_buffer[self.pos:self.pos+batch_size].copy_(ds.to(self.device))
            self.s_buffer[self.pos:self.pos+batch_size].copy_(s.to(self.device))
            self.a_buffer[self.pos:self.pos+batch_size].copy_(a.to(self.device))
            self.r_buffer[self.pos:self.pos+batch_size].copy_(r.to(self.device))
            self.ns_buffer[self.pos:self.pos+batch_size].copy_(ns.to(self.device))
            self.d_buffer[self.pos:self.pos+batch_size].copy_(d.to(self.device))
            self.priorities[self.pos:self.pos+batch_size].fill_(self.priorities[:batch_size].max().item() if self.size > 0 else 1.0)
        else:
            self.ds_buffer[self.pos:self.capacity].copy_(ds[:remain].to(self.device))
            self.s_buffer[self.pos:self.capacity].copy_(s[:remain].to(self.device))
            self.a_buffer[self.pos:self.pos+remain].copy_(a[:remain].to(self.device))
            self.r_buffer[self.pos:self.pos+remain].copy_(r[:remain].to(self.device))
            self.ns_buffer[self.pos:self.pos+remain].copy_(ns[:remain].to(self.device))
            self.d_buffer[self.pos:self.pos+remain].copy_(d[:remain].to(self.device))
            self.priorities[self.pos:self.capacity].fill_(self.priorities[:self.pos].max().item() if self.size > 0 else 1.0)
            extra = batch_size - remain
            self.ds_buffer[0:extra].copy_(ds[remain:].to(self.device))
            self.s_buffer[0:extra].copy_(s[remain:].to(self.device))
            self.a_buffer[0:extra].copy_(a[remain:].to(self.device))
            self.r_buffer[0:extra].copy_(r[remain:].to(self.device))
            self.ns_buffer[0:extra].copy_(ns[remain:].to(self.device))
            self.d_buffer[0:extra].copy_(d[remain:].to(self.device))
            self.priorities[0:extra].fill_(self.priorities[self.pos - 1].max().item())
        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size, beta=0.4):
        """Samples a batch of transitions along with importance-sampling weights."""
        current_size = self.size
        prios = self.priorities[:current_size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = torch.multinomial(probs, batch_size, replacement=False)
        ds = self.ds_buffer[indices]
        s = self.s_buffer[indices]
        a = self.a_buffer[indices]
        r = self.r_buffer[indices]
        ns = self.ns_buffer[indices]
        d = self.d_buffer[indices]
        total = current_size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.unsqueeze(1)
        return (ds, s, a, r, ns, d), indices, weights

    def update_priorities(self, indices, new_priorities):
        """Updates the priorities for the given indices."""
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        new_prios = new_priorities.clone().detach().to(dtype=torch.float32, device=self.device)
        self.priorities[indices] = new_prios

    def __len__(self):
        return self.size

###############################################################################
# QNetwork & Policy Base Classes
###############################################################################
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Q-network 1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_fc3 = nn.Linear(hidden, 1)
        # Q-network 2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_fc3 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        # Concatenate state and action.
        sa = torch.cat([s, a], dim=-1)
        # Q-network 1 forward pass.
        x1 = F.relu(self.q1_fc1(sa))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_fc3(x1)
        # Q-network 2 forward pass.
        x2 = F.relu(self.q2_fc1(sa))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_fc3(x2)
        return q1, q2

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # Two fully connected layers.
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # Output layers for mean and log_std.
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, s, eps=1e-6):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()  # reparameterization trick
        action = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1 - action.pow(2) + eps)
        logp = logp.sum(dim=-1, keepdim=True)
        return action, logp

# CorrectedGaussianPolicy implements corrected sampling (CSAC).
class CorrectedGaussianPolicy(GaussianPolicy):
    def sample(self, s, eps=1e-6):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()  # unsquashed sample
        action = torch.tanh(u)
        # Compute the corrected log probability.
        logp = dist.log_prob(u) - torch.log(1 - action.pow(2) + eps)
        logp = logp.sum(dim=-1, keepdim=True)
        return action, logp

###############################################################################
# Neural Network Agents
###############################################################################
class BaseSACAgent:
    """
    Standard Soft Actor-Critic (SAC) agent.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch_size=256,
        hidden_size=256,
        replay_capacity=1_000_000,
        alpha_init=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        store_replay_on_gpu=True if torch.cuda.is_available() else False,
        seed=None
    ):
        # Set random seed for reproducibility if provided.
        if seed is not None:
            torch.manual_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        self.target_entropy = -float(action_dim)
        self.replay = GPUReplayBuffer(replay_capacity, device=device)
        self.device = device

        # Initialize Q-networks.
        self.critic = QNetwork(state_dim, action_dim, hidden=hidden_size).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden=hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the policy network.
        self.policy = GaussianPolicy(state_dim, action_dim, hidden=hidden_size).to(device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.log_alpha = torch.tensor(math.log(alpha_init), device=device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, s, evaluate=False):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy(s)
                a = torch.tanh(mean)
            else:
                a, _ = self.policy.sample(s)
        return a

    def push_replay(self, ds, s, a, r, ns, d):
        self.replay.push(ds, s, a, r, ns, d)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        ds, s, a, r, ns, d = self.replay.sample(self.batch_size)
        ds = ds.to(self.device, non_blocking=True).view(-1, 1)
        s = s.to(self.device, non_blocking=True)
        a = a.to(self.device, non_blocking=True)
        r = r.to(self.device, non_blocking=True).view(-1, 1)
        ns = ns.to(self.device, non_blocking=True)
        d = d.to(self.device, non_blocking=True).view(-1, 1)
        with torch.no_grad():
            na, nlogp = self.policy.sample(ns)
            q1_next, q2_next = self.critic_target(ns, na)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * nlogp
            target_q = r + (1 - d) * self.gamma * min_q_next
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        new_a, logp = self.policy.sample(s)
        q1_pi, q2_pi = self.critic(s, new_a)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * logp - min_q_pi).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# CSACAgent: uses corrected sampling for the policy.
class CSACAgent(BaseSACAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the policy with the corrected version.
        self.policy = CorrectedGaussianPolicy(self.state_dim, self.action_dim, hidden=kwargs.get('hidden_size', 256)).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.lr)

# DSACAgent: uses a distributional critic.
class DSACAgent(BaseSACAgent):
    def __init__(self, num_quantiles=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_quantiles = num_quantiles
        # Create distributional Q-networks.
        self.critic = DistributionalQNetwork(self.state_dim, self.action_dim,
                                              self.num_quantiles,
                                              hidden=kwargs.get('hidden_size', 256)).to(self.device)
        self.critic_target = DistributionalQNetwork(self.state_dim, self.action_dim,
                                                     self.num_quantiles,
                                                     hidden=kwargs.get('hidden_size', 256)).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        ds, s, a, r, ns, d = self.replay.sample(self.batch_size)
        ds = ds.to(self.device, non_blocking=True).view(-1, 1)
        s = s.to(self.device, non_blocking=True)
        a = a.to(self.device, non_blocking=True)
        r = r.to(self.device, non_blocking=True).view(-1, 1)
        ns = ns.to(self.device, non_blocking=True)
        d = d.to(self.device, non_blocking=True).view(-1, 1)
        with torch.no_grad():
            na, nlogp = self.policy.sample(ns)
            # Get quantile values and compute their mean.
            q_quantiles_next = self.critic_target(ns, na)
            target_q_next = q_quantiles_next.mean(dim=1, keepdim=True) - self.alpha * nlogp
            target_q = r + (1 - d) * self.gamma * target_q_next
        q_quantiles = self.critic(s, a)
        critic_loss = F.mse_loss(q_quantiles.mean(dim=1, keepdim=True), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        new_a, logp = self.policy.sample(s)
        q1_pi = self.critic(s, new_a).mean(dim=1, keepdim=True)
        policy_loss = (self.alpha * logp - q1_pi).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

# ISAC_DSACAgent: combines DSAC with prioritized replay (ISAC).
class ISAC_DSACAgent(DSACAgent):
    """
    ISAC+DSAC Agent: Uses a distributional critic (DSAC) and a prioritized replay buffer (ISAC).
    """
    def __init__(self, prioritized_alpha=0.6, prioritized_beta=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the replay buffer with a prioritized version.
        self.replay = PrioritizedReplayBuffer(capacity=kwargs.get('replay_capacity', 1_000_000),
                                                alpha=prioritized_alpha,
                                                device=self.device)
        self.prioritized_beta = prioritized_beta

    def update(self):
        if len(self.replay) < self.batch_size:
            return
        # Sample transitions along with their importance-sampling weights.
        (ds, s, a, r, ns, d), indices, weights = self.replay.sample(self.batch_size, beta=self.prioritized_beta)
        ds = ds.to(self.device, non_blocking=True).view(-1, 1)
        s = s.to(self.device, non_blocking=True)
        a = a.to(self.device, non_blocking=True)
        r = r.to(self.device, non_blocking=True).view(-1, 1)
        ns = ns.to(self.device, non_blocking=True)
        d = d.to(self.device, non_blocking=True).view(-1, 1)
        
        weights = weights.to(self.device, non_blocking=True)
        with torch.no_grad():
            na, nlogp = self.policy.sample(ns)
            q_quantiles_next = self.critic_target(ns, na)
            target_q_next = q_quantiles_next.mean(dim=1, keepdim=True) - self.alpha * nlogp
            target_q = r + (1 - d) * self.gamma * target_q_next
        q_quantiles = self.critic(s, a)
        critic_loss = F.mse_loss(q_quantiles.mean(dim=1, keepdim=True), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        new_a, logp = self.policy.sample(s)
        q1_pi = self.critic(s, new_a).mean(dim=1, keepdim=True)
        policy_loss = (self.alpha * logp - q1_pi).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        # Update priorities based on TD error.
        td_error = (q_quantiles.mean(dim=1, keepdim=True) - target_q).abs() + 1e-6
        td_error = td_error.squeeze().detach().cpu()
        self.replay.update_priorities(indices, td_error)

# CSAC_DSACAgent: DSAC with corrected sampling.
class CSAC_DSACAgent(DSACAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace policy with corrected version.
        self.policy = CorrectedGaussianPolicy(self.state_dim, self.action_dim, hidden=kwargs.get('hidden_size', 256)).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.lr)

# CSAC_ISAC_DSACAgent: combines corrected sampling, DSAC, and prioritized replay.
class CSAC_ISAC_DSACAgent(ISAC_DSACAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace policy with corrected version.
        self.policy = CorrectedGaussianPolicy(self.state_dim, self.action_dim, hidden=kwargs.get('hidden_size', 256)).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.lr)

###############################################################################
# Distributional Q-Network for DSAC variants.
###############################################################################
class DistributionalQNetwork(nn.Module):
    """
    A simple distributional Q-network that outputs a set of quantile estimates.
    """
    def __init__(self, state_dim, action_dim, num_quantiles, hidden=256):
        super().__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_quantiles)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        quantiles = self.fc3(x)
        return quantiles

###############################################################################
# Agent Factory
###############################################################################
class SACAgentFactory:
    @staticmethod
    def create(agent_type, **kwargs):
        if agent_type == "SAC":
            return BaseSACAgent(**kwargs)
        elif agent_type == "CSAC":
            return CSACAgent(**kwargs)
        elif agent_type == "DSAC":
            return DSACAgent(**kwargs)
        elif agent_type == "ISAC+DSAC":
            return ISAC_DSACAgent(**kwargs)
        elif agent_type == "CSAC+DSAC":
            return CSAC_DSACAgent(**kwargs)
        elif agent_type == "CSAC+ISAC+DSAC":
            return CSAC_ISAC_DSACAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

###############################################################################
# HPC Batch Env Loop - Random initialization from bound for last action_dim dims
###############################################################################
def train_model_sac_batched(
    agent: BaseSACAgent,
    num_envs=8,
    init_state: torch.Tensor = None,
    spins: torch.Tensor = None,
    dSigma: float = 0.0,
    d_max: float = 9.0,
    bound: torch.Tensor = None,
    max_steps=20000,
    max_episode_len=50,
    start_steps=1000,
    d_step_size=0.01,
    reward_scale=10.0,
    std_z=0.1,
    num_points=400,
    updates_per_step=1,
    N_lsq=20, 
    n_states_rew=2,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    WNet=None,
    # New parameters for linear LR decay and checkpointing:
    lr=3e-4,
    lr_final=1e-4,
    checkpoint_dir=None,
    checkpoint_prefix=None,
    checkpoint_interval=5000  # in training steps
):
    """
    Batched training for SAC with vectorized operations, linear learning rate decay,
    and periodic checkpointing.

    The learning rate decays linearly from lr to lr_final as total_steps evolves from 1 to max_steps.
    Checkpoints are saved at every checkpoint_interval training steps if checkpoint_dir is provided.
    """
    # Ensure init_state is on the target device.
    if init_state is None:
        init_state = torch.zeros(agent.critic.q1_fc1.in_features - agent.policy.fc1.in_features,
                                 dtype=dtype, device=device)
    else:
        init_state = init_state.to(device, dtype=dtype)

    dlr = lr_final - lr

    # Deduce dimensions:
    sdim = init_state.shape[0]                      # total state dimension
    adim = agent.policy.mean_head.out_features       # action dimension
    fixed_dim = sdim - adim                         # dimensions fixed to init_state

    # Ensure bound is on the target device.
    bound = bound.to(device, dtype=dtype)

    # Build initial states using vectorized operations:
    states = torch.empty(num_envs, sdim, device=device, dtype=dtype)
    states[:, :fixed_dim] = init_state[:fixed_dim].unsqueeze(0).expand(num_envs, fixed_dim)
    lo_var = bound[fixed_dim:, 0].unsqueeze(0)  # shape (1, adim)
    hi_var = bound[fixed_dim:, 1].unsqueeze(0)
    states[:, fixed_dim:] = torch.rand((num_envs, adim), device=device, dtype=dtype) * (hi_var - lo_var) + lo_var
    
    ep_steps = torch.zeros(num_envs, device=device, dtype=dtype)
    total_steps = 0

    # Precompute inactive zeros for state update.
    inactive_dim = agent.num_inactive_dim
    inactive_zeros = torch.zeros(num_envs, inactive_dim, device=device, dtype=dtype)

    # Select reward function based on whether WNet is provided.
    rew = least_sq_std_rew if WNet is None else least_sq_std_rew_W_func

    # Precompute full-bound vectors for vectorized out-of-bound checks.
    lo_all = bound[:, 0].unsqueeze(0)  # shape (1, sdim)
    hi_all = bound[:, 1].unsqueeze(0)  # shape (1, sdim)
    # Estimate total iterations (each iteration increases total_steps by num_envs)
    total_iterations = math.ceil(max_steps / num_envs)
    pbar = tqdm(total=total_iterations, unit="iter", desc=f"Training ({agent.__class__.__name__})")

    if checkpoint_dir is not None and os.path.isdir(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)

    # Generate HPC z points on the correct device.
    z_gpu = generate_gaussian_complex_points(num_points, std=std_z, device=device)

    while total_steps < max_steps:
        total_steps += num_envs
        # Generate HPC z points on the correct device.
        # z_gpu = generate_gaussian_complex_points(num_points, std=std_z, device=device)
        # Choose actions: random before start_steps, then policy-based.
        if total_steps < start_steps:
            actions = 2 * torch.rand(num_envs, adim, device=device, dtype=dtype) - 1
        else:
            actions = agent.select_action(states, evaluate=False)

        # Compute next states.
        appended = torch.cat([inactive_zeros, d_step_size * actions], dim=1)
        next_states = states + appended
        # Compute rewards via HPC expansion.
        
        new_hpc,_,_ =  rew(
            next_states, z_gpu, spins, dSigma, d_max if WNet is None else WNet,
            N_lsq=N_lsq, n_states_rew=n_states_rew
        )
        new_hpc  *=reward_scale
        # Replace NaN/Inf rewards with 0.
        new_hpc[torch.isnan(new_hpc) | torch.isinf(new_hpc)] = 0.0

        ep_steps += 1

        # Determine 'done' flags.
        cond1 = ep_steps >= max_episode_len
        #cond_bounds = ((next_states < lo_all) | (next_states > hi_all)).any(dim=1)
        use_bound_check=True
        if use_bound_check:
            cond_bounds = ((next_states < lo_all) | (next_states > hi_all)).any(dim=1)
            # Penalize rewards for out-of-bound transitions.
            new_hpc[cond_bounds] = -torch.abs(new_hpc[cond_bounds] * max_episode_len)
        else:
            cond_bounds = torch.zeros_like(cond1, dtype=bool)

        done_mask_new = (cond1 | cond_bounds).to(dtype=dtype)
        # Store transitions.
        agent.push_replay(torch.tensor(dSigma), states, actions, new_hpc, next_states, done_mask_new)

        # Update current states.
        states = next_states.clone()

        # Reset finished environments using vectorized masking.
        done_mask_bool = done_mask_new > 0.5
        if done_mask_bool.any():
            ep_steps[done_mask_bool] = 0.0
            states[done_mask_bool, :fixed_dim] = init_state[:fixed_dim].unsqueeze(0).expand(done_mask_bool.sum(), fixed_dim)
            var_dim = sdim - fixed_dim
            rand_vals = torch.rand((done_mask_bool.sum(), var_dim), device=device, dtype=dtype)
            lo_reset = bound[fixed_dim:, 0].unsqueeze(0)
            hi_reset = bound[fixed_dim:, 1].unsqueeze(0)
            states[done_mask_bool, fixed_dim:] = rand_vals * (hi_reset - lo_reset) + lo_reset

        # Update the agent if we've passed the start_steps threshold.
        if total_steps > start_steps:
            for _ in range(updates_per_step):
                agent.update()
                # Compute new learning rate via linear decay.
                new_lr = lr + dlr * min(total_steps, max_steps) / max_steps
                # Update learning rate for each optimizer.
                for optimizer in [agent.critic_opt, agent.policy_opt, agent.alpha_opt]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
        # Save a checkpoint if a checkpoint directory is provided.
        if checkpoint_dir is not None and (total_steps % checkpoint_interval) < num_envs:
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_{total_steps}.pth")
            save_sac_agent(agent, checkpoint_path)

        pbar.update(1)
        pbar.refresh()

    pbar.close()

import os
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, Any

# 下面假设以下几个函数/类已经在全局作用域里 import 完毕：
#   - BaseSACAgent
#   - generate_gaussian_complex_points
#   - least_sq_std_rew
#   - least_sq_std_rew_W_func
#   - least_sq_std_rew_Virasoro
#   - least_sq_std_rew_Hybrid
#   - dtype（例如 torch.float64）
#   - save_sac_agent
#   - ……

def train_model_sac_batched(
    agent: "BaseSACAgent",
    reward_fn: Callable[..., torch.Tensor],
    reward_kwargs: Dict[str, Any],
    num_envs: int = 8,
    init_state: torch.Tensor = None,
    spins: torch.Tensor = None,
    dSigma: float = 0.0,
    d_max: float = 9.0,
    bound: torch.Tensor = None,
    max_steps: int = 20000,
    max_episode_len: int = 50,
    start_steps: int = 1000,
    d_step_size: float = 0.01,
    reward_scale: float = 10.0,
    std_z: float = 0.1,
    num_points: int = 400,
    updates_per_step: int = 1,
    N_lsq: int = 20,
    n_states_rew: int = 2,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr: float = 3e-4,
    lr_final: float = 1e-4,
    checkpoint_dir: str = None,
    checkpoint_prefix: str = None,
    checkpoint_interval: int = 5000,
):
    """
    Batched SAC 训练函数，内置“线性衰减 lr”、“周期性存 checkpoint”，
    并且支持传入任意符合约定签名的 reward_fn + reward_kwargs。
    
    reward_fn 的约定签名：
        reward_fn(
            next_states: torch.Tensor,  # shape (num_envs, sdim), dtype float64, device=device
            x: torch.Tensor,            # shape (num_points,), dtype float64
            y: torch.Tensor,            # shape (num_points,), dtype float64
            spins: torch.Tensor,        # shape (N_state,), dtype float64
            dSigma: float,
            d_max: float,
            **reward_kwargs             # 其余依赖于具体 reward_fn 的关键词参数
        ) -> torch.Tensor               # 返回 shape (num_envs,) 的 reward 标量
    """

    # 1) 准备初始状态
    dtype = torch.float64
    if init_state is None:
        # 如果没传 init_state，则让它等于 agent 的网络输入 dimension
        # 这里假定 agent.policy.fc1.in_features 是 action 维度，
        # agent.critic.q1_fc1.in_features - agent.policy.fc1.in_features 即为 fixed_dim
        fixed_dim = agent.critic.q1_fc1.in_features - agent.policy.fc1.in_features
        init_state = torch.zeros(fixed_dim, dtype=dtype, device=device)
    else:
        init_state = init_state.to(device=device, dtype=dtype)

    # 2) 计算一下 lr 衰减量
    dlr = lr_final - lr

    # 3) 从 init_state 推断状态/action 维度
    sdim = init_state.shape[0] + agent.policy.mean_head.out_features
    adim = agent.policy.mean_head.out_features
    fixed_dim = init_state.shape[0]

    # 4) bound 张量
    if bound is None:
        raise ValueError("必须提供 bound, 用于约束 action 对应的 state 分量")
    bound = bound.to(device=device, dtype=dtype)  # shape (sdim, 2)

    # 5) 生成一批并行 env 的初始 states
    states = torch.empty(num_envs, sdim, device=device, dtype=dtype)
    # 前 fixed_dim 分量固定
    states[:, :fixed_dim] = init_state.unsqueeze(0).expand(num_envs, fixed_dim)
    # 后面 action 对应的那几维（adim）用均匀随机初始化
    lo_var = bound[fixed_dim:, 0].unsqueeze(0)  # (1,adim)
    hi_var = bound[fixed_dim:, 1].unsqueeze(0)
    states[:, fixed_dim:] = torch.rand(num_envs, adim, device=device, dtype=dtype) * (hi_var - lo_var) + lo_var

    # 6) 记录各个 env 已经走了多少 step
    ep_steps = torch.zeros(num_envs, device=device, dtype=dtype)
    total_steps = 0

    # 7) precompute 用于更新状态“插 0”的张量
    inactive_dim = agent.num_inactive_dim  # 假设 agent 里有记录多少维度不动
    inactive_zeros = torch.zeros(num_envs, inactive_dim, device=device, dtype=dtype)

    # 8) 用来做 out-of-bound 判断的上下界
    lo_all = bound[:, 0].unsqueeze(0)  # (1, sdim)
    hi_all = bound[:, 1].unsqueeze(0)  # (1, sdim)

    # 9) 生成 z 复数点（HPC expansion）：
    #    这里我们用 generate_gaussian_complex_points 生成 complex128，
    #    然后在具体的 reward_fn 里自行拆成 real/imag
    z_gpu_complex = generate_gaussian_complex_points(num_points, std=std_z, device=device)
    # 后面传入 reward_fn 时，reward_fn 里面要自己写 `.real` 和 `.imag`

    # 10) 进度条 + checkpoint 目录检查
    total_iterations = math.ceil(max_steps / num_envs)
    pbar = tqdm(total=total_iterations, unit="iter", desc=f"Training ({agent.__class__.__name__})")
    if checkpoint_dir is not None and not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 11) 开始主循环
    while total_steps < max_steps:
        total_steps += num_envs

        # 11.1) 先选动作：在 start_steps 之前随机，否则用 policy
        if total_steps < start_steps:
            actions = 2 * torch.rand(num_envs, adim, device=device, dtype=dtype) - 1.0
        else:
            actions = agent.select_action(states, evaluate=False)

        # 11.2) 计算 next_states
        appended = torch.cat([inactive_zeros, d_step_size * actions], dim=1)  # (num_envs, sdim)
        next_states = states + appended  # (num_envs, sdim)

        # 11.3) 计算 reward —— 只需要把 next_states、z_gpu_complex、spins、dSigma、d_max 传给 reward_fn，
        #        其余额外的参数都从 reward_kwargs 里取。
        #        假设 reward_fn 内部会做：
        #          x_vals = z_gpu_complex.real
        #          y_vals = z_gpu_complex.imag
        #          然后调用正确的内核和最小二乘流程。
        #
        #        这里我们只给一个“扁平”的、对 num_envs 并行的 reward tensor：
        reward_vals = reward_fn(
            next_states,               # (num_envs, sdim)
            z_gpu_complex,             # complex128, (num_points,)
            spins,                     # (N_state,)
            dSigma,                    # float
            d_max,                     # float
            **reward_kwargs
        )
        # reward_vals 需要是 torch.Tensor，形状 (num_envs,)；下面 scale 一下
        reward_vals = reward_vals * reward_scale
        reward_vals[torch.isnan(reward_vals) | torch.isinf(reward_vals)] = 0.0

        # 11.4) 更新 ep_steps
        ep_steps += 1.0

        # 11.5) 判断 done：超过 max_episode_len 或者 越界
        cond1 = ep_steps >= max_episode_len
        cond_bounds = ((next_states < lo_all) | (next_states > hi_all)).any(dim=1)
        # 如果越界，就给一个比较大的惩罚
        reward_vals[cond_bounds] = -torch.abs(reward_vals[cond_bounds] * max_episode_len)
        done_mask_new = (cond1 | cond_bounds).to(dtype=dtype)  # float 型 0/1

        # 11.6) 将 transition 存到 replay buffer
        #       这里假设 agent.push_replay 接口是：
        #         push_replay(dSigma_tensor, state, action, reward, next_state, done_mask)
        agent.push_replay(torch.tensor(dSigma, device=device), states, actions, reward_vals, next_states, done_mask_new)

        # 11.7) 更新当前 states
        states = next_states.clone()

        # 11.8) 对于 done 的 env，把它们重置
        done_mask_bool = done_mask_new > 0.5
        if done_mask_bool.any():
            # 重置 ep_steps、fixed 部分复位、可动部分随机
            ep_steps[done_mask_bool] = 0.0
            states[done_mask_bool, :fixed_dim] = init_state.unsqueeze(0).expand(done_mask_bool.sum(), fixed_dim)
            var_dim = sdim - fixed_dim
            rand_vals = torch.rand((done_mask_bool.sum(), var_dim), device=device, dtype=dtype)
            lo_reset = bound[fixed_dim:, 0].unsqueeze(0)
            hi_reset = bound[fixed_dim:, 1].unsqueeze(0)
            states[done_mask_bool, fixed_dim:] = rand_vals * (hi_reset - lo_reset) + lo_reset

        # 11.9) 如果超过 start_steps，就开始更新 agent
        if total_steps > start_steps:
            for _ in range(updates_per_step):
                agent.update()
                # 线性衰减 lr
                new_lr = lr + dlr * min(total_steps, max_steps) / max_steps
                for optimizer in [agent.critic_opt, agent.policy_opt, agent.alpha_opt]:
                    for pg in optimizer.param_groups:
                        pg["lr"] = new_lr

        # 11.10) checkpoint 存盘
        if checkpoint_dir is not None and (total_steps % checkpoint_interval) < num_envs:
            ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_{total_steps}.pth")
            save_sac_agent(agent, ckpt_path)

        pbar.update(1)

    pbar.close()

    return agent


import numpy as np
import torch

def gather_trajectories_sac_batched(
    agent: BaseSACAgent,
    num_envs=8,
    num_episodes=20,
    init_state: torch.Tensor = None,
    spins: torch.Tensor = None,
    dSigma: float = 0.0,
    d_max: float = 9.0,
    bound: torch.Tensor = None,
    n_states_rew=2,
    d_step_size=0.01,
    reward_scale=10.0,
    std_z=0.1,
    num_points=400,
    N_lsq=20, 
    max_episode_len=50,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    WNet=None,
):
    import numpy as np
    import torch

    # Assume a default dtype is defined (or use torch.get_default_dtype())
    dtype = torch.get_default_dtype()

    # Ensure initial state is on the correct device and of the proper dtype.
    s0 = init_state.clone().to(device, dtype=dtype)

    # Deduce dimensions:
    sdim = s0.shape[0]                      # total state dimension
    adim = agent.policy.mean_head.out_features       # action dimension
    fixed_dim = sdim - adim                         # dimensions fixed to init_state

    # 初始化多個環境的狀態
    states = torch.empty(num_envs, sdim, device=device, dtype=dtype)
    states[:, :fixed_dim] = s0[:fixed_dim].unsqueeze(0).expand(num_envs, fixed_dim)
    lo_var = bound[fixed_dim:, 0].unsqueeze(0)      # shape (1, adim)
    hi_var = bound[fixed_dim:, 1].unsqueeze(0)
    states[:, fixed_dim:] = torch.rand((num_envs, adim), device=device, dtype=dtype) * (hi_var - lo_var) + lo_var

    # 預先計算邊界檢查用的向量
    lo_all = bound[:, 0].unsqueeze(0)  # shape (1, sdim)
    hi_all = bound[:, 1].unsqueeze(0)

    # 初始化每個環境的步數與軌跡收集變數
    ep_steps = torch.zeros(num_envs, device=device, dtype=dtype)
    # 每個環境初始狀態的數值軌跡
    trajs = [[states[i].detach().cpu().numpy()] for i in range(num_envs)]
    # 收集 reward 序列（這裡初始可以設定為 0.0）
    reward_trajs = [[0.0] for i in range(num_envs)]
    # 新增：用來收集 c_mean 與 c_std 這兩組變數
    c_mean_trajs = [[] for i in range(num_envs)]
    c_std_trajs = [[] for i in range(num_envs)]

    all_trajectories = []  # 儲存完成 episode 的 state 軌跡
    all_rewards = []       # 儲存完成 episode 的 reward 序列
    all_c_means = []       # 儲存完成 episode 的 c_mean 序列
    all_c_stds = []        # 儲存完成 episode 的 c_std 序列
    episodes_completed = 0

    # 預先計算 inactive zeros（用於 state 更新中固定部分不變）
    inactive_dim = agent.num_inactive_dim
    inactive_zeros = torch.zeros(num_envs, inactive_dim, device=device, dtype=dtype)

    # 選擇 reward function (rew)；它返回三個變數：new_hpc, c_mean, c_std
    rew = least_sq_std_rew if WNet is None else least_sq_std_rew_W_func
    while episodes_completed < num_episodes:
        z_gpu = generate_gaussian_complex_points(num_points, std=std_z, device=device)
        # 選取動作，計算 next_states
        a_gpu = agent.select_action(states, evaluate=True)
        appended = torch.cat([inactive_zeros, d_step_size * a_gpu], dim=1)
        next_states = states + appended
        ep_steps += 1

        # 用 rew 計算 reward 及其它指標
        new_hpc, cur_c_mean, cur_c_std = rew(
            next_states, z_gpu, spins, dSigma, d_max if WNet is None else WNet,
            N_lsq=N_lsq, n_states_rew=n_states_rew
        )

        new_hpc *= reward_scale
        new_hpc[torch.isnan(new_hpc) | torch.isinf(new_hpc)] = 0.0

        # 判斷是否達到終止條件：超過步數上限或狀態超出 bound
        cond1 = ep_steps >= max_episode_len
        cond_bounds = ((next_states < lo_all) | (next_states > hi_all)).any(dim=1)
        done_mask_new = (cond1 | cond_bounds).to(dtype=dtype)

        # 處理未結束環境：更新 state 與收集資料
        not_done = (done_mask_new < 0.5)
        for i in torch.nonzero(not_done, as_tuple=False).flatten().tolist():
            states[i] = next_states[i]
            trajs[i].append(states[i].detach().cpu().numpy())
            reward_trajs[i].append(new_hpc[i].item())
            # 收集 c_mean 與 c_std (轉換成 NumPy 資料)
            c_mean_trajs[i].append(cur_c_mean[i].detach().cpu().numpy())
            c_std_trajs[i].append(cur_c_std[i].detach().cpu().numpy())

        # 處理結束環境：存入結果，並重置該環境
        done_idxs = torch.nonzero(done_mask_new > 0.5, as_tuple=False).flatten().tolist()
        if len(done_idxs) > 0:
            for i in done_idxs:
                all_trajectories.append(np.array(trajs[i]))
                all_rewards.append(np.array(reward_trajs[i]))
                all_c_means.append(np.array(c_mean_trajs[i]))
                all_c_stds.append(np.array(c_std_trajs[i]))
                episodes_completed += 1
                if episodes_completed >= num_episodes:
                    break
                ep_steps[i] = 0.0
                states[i, :fixed_dim] = s0[:fixed_dim] # states[:fixed_dim].unsqueeze(0).expand(1, fixed_dim)
                var_dim = sdim - fixed_dim
                rand_vals = torch.rand((1, var_dim), device=device, dtype=dtype)
                lo_reset = bound[fixed_dim:, 0].unsqueeze(0)
                hi_reset = bound[fixed_dim:, 1].unsqueeze(0)
                states[i, fixed_dim:] = rand_vals * (hi_reset - lo_reset) + lo_reset
                trajs[i] = [states[i].detach().cpu().numpy()]
                reward_trajs[i] = [0.0]
                c_mean_trajs[i] = []
                c_std_trajs[i] = []
        if episodes_completed >= num_episodes:
            break
    return all_trajectories[:num_episodes], all_rewards[:num_episodes], all_c_means[:num_episodes], all_c_stds[:num_episodes]

from typing import Callable, Dict, Any, List, Tuple
class SACBatchedRunner:
    """
    把 train_model_sac_batched 和 gather_trajectories_sac_batched 整合到一个类里。
    通过传入不同的 reward_fn（或 WNet）来切换 reward 模式。

    使用示例：

        # 假设你已经有一个 BaseSACAgent 的实例 agent_ 实例，
        # 和 bound 张量、init_state、spins 等。
        runner = SACBatchedRunner(
            agent=agent_,
            bound=bound_tensor,
            reward_fn=my_reward_function,        # 或者不传，改用 WNet 模式
            reward_kwargs={'foo': 1.23, 'bar': 4.56},
            WNet=None,                           # 如果用 least_sq_std_rew_W_func，可以传入对应网络
            num_envs=8,
            max_steps=20000,
            max_episode_len=50,
            # 下面还有一大堆可选超参数……
        )

        # 1) 训练模型：
        runner.train()

        # 2) 采集轨迹：
        trajs, rewards, c_means, c_stds = runner.gather_trajectories(num_episodes=20)
    """

    def __init__(
        self,
        agent: BaseSACAgent,
        reward_fn: Callable[..., torch.Tensor] = None,
        reward_kwargs: Dict[str, Any] = None,
        num_envs=8,
        h_vals: torch.Tensor = None,
        hb_vals: torch.Tensor = None,
        h_ext: float = 0.0,
        bound: torch.Tensor = None,
        max_steps=20000,
        max_episode_len=50,
        start_steps=1000,
        d_step_size=0.01,
        reward_scale=10.0,
        std_z=0.1,
        num_points=400,
        updates_per_step=1,
        N_lsq=20, 
        n_states_rew=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        WNet=None,
        # New parameters for linear LR decay and checkpointing:
        lr=3e-4,
        lr_final=1e-4,
        checkpoint_dir=None,
        checkpoint_prefix=None,
        checkpoint_interval=5000  # in training steps
    ):
        """
        初始化时把所有可能用到的超参数都收进来。

        Args:
            agent: 已经初始化好的 SAC agent 实例（需要有 select_action, update, push_replay 等接口）。
            bound: 形状为 (sdim, 2) 的张量，用于约束 state 的上下界。
            reward_fn: 如果你采用最简单的 least_sq_std_rew 方式，就把它传进来。
                       如果你想用带 WNet 的版本，就把 reward_fn 设为 None，WNet 设为网络实例。
            reward_kwargs: 传给 reward_fn 的额外参数，字典形式。
            WNet: 如果选用 least_sq_std_rew_W_func 模式，就把 WNet（网络实例）传进来。
                   否则留 None。
            num_envs: 并行环境数量。
            init_state: 初始的 fixed 状态张量（形状为 (fixed_dim,)）。
                        如果为 None，训练时会自动用全 0 来初始化那几维度。
            spins: 用于 reward_fn 的 `spins` 张量（形状为 (N_state,)）。
            dSigma, d_max: reward_fn 还需要的两个浮点参数。
            max_steps, max_episode_len 等: 训练、采集轨迹时的超参数。
            start_steps: 开始训练前纯随机探索的步数阈值。
            d_step_size: 动作到状态的映射尺度（next_state = state + d_step_size * action）。
            reward_scale: 训练、采集时对 reward 的整体缩放因子。
            std_z, num_points: 生成复数点（HPC expansion）时的参数。
            updates_per_step: 每一次采集（env stepping）后，更新多少次网络。
            N_lsq, n_states_rew: 在 least-squares reward 函数里用到的参数。
            lr, lr_final: 线性衰减的初始 lr 和最终 lr。
            checkpoint_dir, checkpoint_prefix, checkpoint_interval: checkpoint 存储相关。
            device: 如果没传，默认用 cuda（若可用）或 cpu。
        """
        # ----- 基本属性 -----
        self.agent = agent
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_envs = num_envs

        # ----- reward 相关 -----
        # 如果 reward_fn 给了，就用 reward_fn；否则认为要用 WNet 模式
        self.reward_fn = reward_fn
        self.reward_kwargs = reward_kwargs or {}
        self.h_vals = h_vals
        self.hb_vals = hb_vals
        self.h_ext = h_ext

        # ----- 状态空间 & 边界 -----
        # init_state: 形状为 (fixed_dim,) 的张量（dtype 会在 train/gather 时改成 float64）
        # bound: 形状为 (sdim, 2) 的张量，用于约束 state 的上下界
        if bound is None:
            raise ValueError("必须提供 bound，用于约束 action 对应的 state 分量")
        self.bound = bound.to(device=self.device, dtype=torch.float64)

        # ----- 训练相关超参数 -----
        self.max_steps = max_steps
        self.max_episode_len = max_episode_len
        self.start_steps = start_steps
        self.d_step_size = d_step_size
        self.reward_scale = reward_scale
        self.std_z = std_z
        self.num_points = num_points
        self.updates_per_step = updates_per_step
        self.N_lsq = N_lsq
        self.n_states_rew = n_states_rew

        # ----- 优化器 lr 相关 -----
        self.lr = lr
        self.lr_final = lr_final
        self.dlr = lr_final - lr

        # ----- checkpoint -----
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_interval = checkpoint_interval

        # ----- 进度条 -----
        self.pbar = None

        # ----- 其他 -----
        # agent 里必须有以下几个属性/方法：
        #   - policy.mean_head.out_features （动作维度 adim）
        #   - critic.q1_fc1.in_features （用来推断 fixed_dim）
        #   - select_action(states, evaluate)      # 输出动作
        #   - push_replay(dSigma_tensor, state, action, reward, next_state, done_mask)
        #   - update()                             # 训练一步
        #   - critic_opt, policy_opt, alpha_opt     # 优化器列表
        #   - num_inactive_dim                      # 数值里默认有这个属性
        #

    def _prepare_initials(self) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """
        统一在 train() 或 gather_trajectories() 开始时，准备以下数据：
          - init_state_tensor: 形状 (fixed_dim,) 的 float64 张量
          - sdim, adim, fixed_dim
          - inactive_dim（从 agent.num_inactive_dim 取）
        """
        dtype = torch.float64

        # 1) 如果没有传 init_state，就根据 agent 的网络结构推断 fixed_dim，并让它全 0
        if self.h_vals is None:
            # 这里用 critic.q1_fc1.in_features - policy.fc1.in_features 来推断 fixed_dim
            fixed_dim = (
                self.agent.critic.q1_fc1.in_features
                - self.agent.policy.fc1.in_features
            )
            init_state_tensor = torch.zeros(fixed_dim, dtype=dtype, device=self.device)
        else:
            init_state_tensor = self.init_state.to(device=self.device, dtype=dtype)
            fixed_dim = init_state_tensor.shape[0]

        # 2) 用 init_state 推断 sdim、adim
        adim = self.agent.policy.mean_head.out_features
        sdim = fixed_dim + adim

        # 3) inactive_dim
        inactive_dim = self.agent.num_inactive_dim

        return init_state_tensor, sdim, adim, fixed_dim, inactive_dim

    def train(self) -> "BaseSACAgent":
        """
        训练主函数，相当于原来写的 train_model_sac_batched。

        返回训练好的 agent 对象（self.agent）。
        """
        # --------------- 1) 准备初始状态、边界等 ---------------
        (
            init_state_tensor,
            sdim,
            adim,
            fixed_dim,
            inactive_dim,
        ) = self._prepare_initials()

        # 全局 dtype 确定为 float64
        dtype = torch.float64

        # --------------- 2) bound 张量 ---------------
        bound = self.bound  # 已经在 __init__ 里转为 float64、放到 self.device

        # --------------- 3) 并行环境的初始 states ---------------
        states = torch.empty(self.num_envs, sdim, device=self.device, dtype=dtype)
        # 固定部分全部等于 init_state_tensor
        states[:, :fixed_dim] = init_state_tensor.unsqueeze(0).expand(self.num_envs, fixed_dim)
        # 自由部分用均匀随机去 boundary 里采样
        lo_var = bound[fixed_dim:, 0].unsqueeze(0)  # (1, adim)
        hi_var = bound[fixed_dim:, 1].unsqueeze(0)
        states[:, fixed_dim:] = (
            torch.rand(self.num_envs, adim, device=self.device, dtype=dtype) * (hi_var - lo_var)
            + lo_var
        )

        # --------------- 4) ep_steps, total_steps ---------------
        ep_steps = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        total_steps = 0

        # --------------- 5) inactive_zeros ---------------
        inactive_zeros = torch.zeros(self.num_envs, inactive_dim, device=self.device, dtype=dtype)

        # --------------- 6) 用来做 out-of-bound 判断的上下界 ---------------
        lo_all = bound[:, 0].unsqueeze(0)  # (1, sdim)
        hi_all = bound[:, 1].unsqueeze(0)  # (1, sdim)

        # --------------- 7) 生成 z 复数点 ---------------
        z_gpu_complex = generate_gaussian_complex_points(
            self.num_points, std=self.std_z, device=self.device
        )

        # --------------- 8) 进度条 + checkpoint 目录检查 ---------------
        total_iterations = math.ceil(self.max_steps / self.num_envs)
        self.pbar = tqdm(
            total=total_iterations,
            unit="iter",
            desc=f"Training ({self.agent.__class__.__name__})",
        )
        if self.checkpoint_dir is not None and not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # --------------- 9) 主循环 ---------------
        while total_steps < self.max_steps:
            total_steps += self.num_envs

            # 9.1) 先选动作：在 start_steps 之前随机，否则用 policy
            if total_steps < self.start_steps:
                actions = 2 * torch.rand(self.num_envs, adim, device=self.device, dtype=dtype) - 1.0
            else:
                actions = self.agent.select_action(states, evaluate=False)

            # 9.2) 计算 next_states
            appended = torch.cat([inactive_zeros, self.d_step_size * actions], dim=1)  # (num_envs, sdim)
            next_states = states + appended  # (num_envs, sdim)

            # 9.3) 计算 reward
            # reward 模式分两种：
            #   如果 self.reward_fn 不为 None，就认为用纯 least_sq_std_rew
            #   否则用 least_sq_std_rew_W_func，并把 WNet 传进去
            if self.reward_fn is not None:
                reward_vals = self.reward_fn(
                    next_states,               # (num_envs, sdim)
                    z_gpu_complex,             # complex128, (num_points,)
                    self.spins,                # (N_state,)
                    self.dSigma,
                    self.d_max,
                    **self.reward_kwargs
                )
            else:
                # WNet 模式：least_sq_std_rew_W_func(next_states, z_gpu, spins, dSigma, d_max=N_max, WNet, N_lsq, n_states_rew)
                reward_vals = least_sq_std_rew_W_func(
                    next_states,
                    z_gpu_complex,
                    self.spins,
                    self.dSigma,
                    self.WNet,
                    N_lsq=self.N_lsq,
                    n_states_rew=self.n_states_rew,
                    **self.reward_kwargs
                )
            # reward_vals 形状 (num_envs,)
            reward_vals = reward_vals * self.reward_scale
            reward_vals[torch.isnan(reward_vals) | torch.isinf(reward_vals)] = 0.0

            # 9.4) 更新 ep_steps
            ep_steps += 1.0

            # 9.5) 判断 done：超过 max_episode_len 或者 越界
            cond1 = ep_steps >= self.max_episode_len
            cond_bounds = ((next_states < lo_all) | (next_states > hi_all)).any(dim=1)
            reward_vals[cond_bounds] = -torch.abs(reward_vals[cond_bounds] * self.max_episode_len)
            done_mask_new = (cond1 | cond_bounds).to(dtype=dtype)  # float 型 0/1

            # 9.6) 存入 replay buffer
            self.agent.push_replay(
                torch.tensor(self.dSigma, device=self.device),
                states,
                actions,
                reward_vals,
                next_states,
                done_mask_new,
            )

            # 9.7) 更新当前 states
            states = next_states.clone()

            # 9.8) 对于 done 的 env，把它们重置
            done_mask_bool = done_mask_new > 0.5
            if done_mask_bool.any():
                # 重置 ep_steps、fixed 部分复位、可动部分随机
                ep_steps[done_mask_bool] = 0.0
                states[done_mask_bool, :fixed_dim] = init_state_tensor.unsqueeze(0).expand(
                    done_mask_bool.sum(), fixed_dim
                )
                var_dim = sdim - fixed_dim
                rand_vals = torch.rand((done_mask_bool.sum(), var_dim), device=self.device, dtype=dtype)
                lo_reset = bound[fixed_dim:, 0].unsqueeze(0)
                hi_reset = bound[fixed_dim:, 1].unsqueeze(0)
                states[done_mask_bool, fixed_dim:] = rand_vals * (hi_reset - lo_reset) + lo_reset

            # 9.9) 如果超过 start_steps，就开始更新 agent
            if total_steps > self.start_steps:
                for _ in range(self.updates_per_step):
                    self.agent.update()
                    # 线性衰减 lr
                    new_lr = self.lr + self.dlr * min(total_steps, self.max_steps) / self.max_steps
                    for optimizer in [self.agent.critic_opt, self.agent.policy_opt, self.agent.alpha_opt]:
                        for pg in optimizer.param_groups:
                            pg["lr"] = new_lr

            # 9.10) checkpoint 存盘
            if (
                self.checkpoint_dir is not None
                and (total_steps % self.checkpoint_interval) < self.num_envs
            ):
                ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f"{self.checkpoint_prefix}_{total_steps}.pth",
                )
                save_sac_agent(self.agent, ckpt_path)

            self.pbar.update(1)

        self.pbar.close()
        return self.agent

    def gather_trajectories(
        self,
        num_episodes: int = 20,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        采集轨迹的方法，相当于原来的 gather_trajectories_sac_batched。
        返回：
            all_trajectories: List[np.ndarray]  # 每个元素形状 (T_i, sdim)，T_i 是该 episode 的时长
            all_rewards: List[np.ndarray]       # 每个元素形状 (T_i,)
            all_c_means: List[np.ndarray]       # 如果是 least_sq_std_rew_W_func 模式，会返回 c_mean 序列；否则列表为空
            all_c_stds: List[np.ndarray]        # 同上，c_std 序列
        """
        # --------------- 1) 准备初始状态、边界等 ---------------
        (
            init_state_tensor,
            sdim,
            adim,  # 这里 adim 用不到，因为从 sdim - fixed_dim 可得
            fixed_dim,
            inactive_dim,
        ) = self._prepare_initials()

        dtype = torch.get_default_dtype()

        # --------------- 2) 初始化并行环境的 states ---------------
        states = torch.empty(self.num_envs, sdim, device=self.device, dtype=dtype)
        states[:, :fixed_dim] = init_state_tensor[:fixed_dim].unsqueeze(0).expand(self.num_envs, fixed_dim)
        lo_var = self.bound[fixed_dim:, 0].unsqueeze(0)  # shape (1, adim)
        hi_var = self.bound[fixed_dim:, 1].unsqueeze(0)
        states[:, fixed_dim:] = (
            torch.rand(self.num_envs, adim, device=self.device, dtype=dtype) * (hi_var - lo_var)
            + lo_var
        )

        # --------------- 3) 边界检测向量 ---------------
        lo_all = self.bound[:, 0].unsqueeze(0)  # shape (1, sdim)
        hi_all = self.bound[:, 1].unsqueeze(0)

        # --------------- 4) 初始化 ep_steps、轨迹收集容器 ---------------
        ep_steps = torch.zeros(self.num_envs, device=self.device, dtype=dtype)

        trajs: List[List[np.ndarray]] = [
            [states[i].detach().cpu().numpy()] for i in range(self.num_envs)
        ]
        reward_trajs: List[List[float]] = [[0.0] for _ in range(self.num_envs)]
        c_mean_trajs: List[List[np.ndarray]] = [[] for _ in range(self.num_envs)]
        c_std_trajs: List[List[np.ndarray]] = [[] for _ in range(self.num_envs)]

        all_trajectories: List[np.ndarray] = []
        all_rewards: List[np.ndarray] = []
        all_c_means: List[np.ndarray] = []
        all_c_stds: List[np.ndarray] = []
        episodes_completed = 0

        # --------------- 5) inactive_zeros ---------------
        inactive_zeros = torch.zeros(self.num_envs, inactive_dim, device=self.device, dtype=dtype)

        # --------------- 6) 主循环：采集 num_episodes 个 episode ---------------
        while episodes_completed < num_episodes:
            # (a) 重新生成新的 z 复数点
            z_gpu = generate_gaussian_complex_points(
                self.num_points, std=self.std_z, device=self.device
            )

            # (b) 选动作并计算 next_states
            a_gpu = self.agent.select_action(states, evaluate=True)
            appended = torch.cat([inactive_zeros, self.d_step_size * a_gpu], dim=1)
            next_states = states + appended
            ep_steps += 1

            # (c) 计算 reward 及其他指标
            if self.reward_fn is not None:
                new_hpc, cur_c_mean, cur_c_std = self.reward_fn(
                    next_states,
                    z_gpu,
                    self.spins,
                    self.dSigma,
                    self.d_max,
                    N_lsq=self.N_lsq,
                    n_states_rew=self.n_states_rew,
                    **self.reward_kwargs,
                )
            else:
                new_hpc, cur_c_mean, cur_c_std = least_sq_std_rew_W_func(
                    next_states,
                    z_gpu,
                    self.spins,
                    self.dSigma,
                    self.WNet,
                    N_lsq=self.N_lsq,
                    n_states_rew=self.n_states_rew,
                    **self.reward_kwargs,
                )

            # 归一化并处理 NaN/inf
            new_hpc = new_hpc * self.reward_scale
            new_hpc[torch.isnan(new_hpc) | torch.isinf(new_hpc)] = 0.0

            # (d) 终止条件判断
            cond1 = ep_steps >= self.max_episode_len
            cond_bounds = ((next_states < lo_all) | (next_states > hi_all)).any(dim=1)
            done_mask_new = (cond1 | cond_bounds).to(dtype=dtype)

            # (e) 对尚未 done 的 env，直接把 next_states 写回，并收集数据
            not_done = (done_mask_new < 0.5)
            for i in torch.nonzero(not_done, as_tuple=False).flatten().tolist():
                states[i] = next_states[i]
                trajs[i].append(states[i].detach().cpu().numpy())
                reward_trajs[i].append(new_hpc[i].item())
                c_mean_trajs[i].append(cur_c_mean[i].detach().cpu().numpy())
                c_std_trajs[i].append(cur_c_std[i].detach().cpu().numpy())

            # (f) 对已经 done 的 env，保存已完成的轨迹，并重置该 env
            done_idxs = torch.nonzero(done_mask_new > 0.5, as_tuple=False).flatten().tolist()
            if len(done_idxs) > 0:
                for i in done_idxs:
                    all_trajectories.append(np.array(trajs[i]))
                    all_rewards.append(np.array(reward_trajs[i]))
                    all_c_means.append(np.array(c_mean_trajs[i]))
                    all_c_stds.append(np.array(c_std_trajs[i]))
                    episodes_completed += 1
                    if episodes_completed >= num_episodes:
                        break

                    # 重置该环境
                    ep_steps[i] = 0.0
                    states[i, :fixed_dim] = init_state_tensor[:fixed_dim]
                    var_dim = sdim - fixed_dim
                    rand_vals = torch.rand((1, var_dim), device=self.device, dtype=dtype)
                    lo_reset = self.bound[fixed_dim:, 0].unsqueeze(0)
                    hi_reset = self.bound[fixed_dim:, 1].unsqueeze(0)
                    states[i, fixed_dim:] = rand_vals * (hi_reset - lo_reset) + lo_reset

                    # 重置收集器列表
                    trajs[i] = [states[i].detach().cpu().numpy()]
                    reward_trajs[i] = [0.0]
                    c_mean_trajs[i] = []
                    c_std_trajs[i] = []

            if episodes_completed >= num_episodes:
                break

        return (
            all_trajectories[:num_episodes],
            all_rewards[:num_episodes],
            all_c_means[:num_episodes],
            all_c_stds[:num_episodes],
        )


def save_sac_agent(agent, filepath):
    torch.save({
        'critic': agent.critic.state_dict(),
        'critic_target': agent.critic_target.state_dict(),
        'critic_opt': agent.critic_opt.state_dict(),
        'policy': agent.policy.state_dict(),
        'policy_opt': agent.policy_opt.state_dict(),
        'log_alpha': agent.log_alpha,
        'alpha_opt': agent.alpha_opt.state_dict()
    }, filepath)


def load_sac_agent(agent, filepath, weights_only=True):
    checkpoint = torch.load(filepath, map_location=agent.device, weights_only=weights_only)
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.critic_target.load_state_dict(checkpoint['critic_target'])
    agent.critic_opt.load_state_dict(checkpoint['critic_opt'])
    agent.policy.load_state_dict(checkpoint['policy'])
    agent.policy_opt.load_state_dict(checkpoint['policy_opt'])
    agent.log_alpha = checkpoint['log_alpha']
    agent.alpha_opt.load_state_dict(checkpoint['alpha_opt'])

