import os
import time
import torch

from .SAC import SACAgentFactory, train_model_sac_batched, gather_trajectories_sac_batched, gather_trajectories_sac_batched_v1
from .utils import plot_r, plot_r_gauss_z,generate_gaussian_complex_points

from ..z_sampling.z_weight_net import WNet
from .env import least_sq_std_rew, least_sq_std_rew_W_func

###############################################################################
# Train_and_Predict Wrapper
###############################################################################
class Train_and_Predict:
    """
    Wrapper to train and predict using a SAC agent.
    This class handles environment simulation, training, and prediction.
    """
    def __init__(
        self,
        theory,
        scope,
        spins,
        dSigma,
        d_max,
        init_state,
        bound,
        device=None,#torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        state_dim=None,
        action_dim=None,
        max_steps=int(1e6),
        max_episode_len=50,
        start_steps=10000,
        d_step_size=0.1,
        reward_scale=10.0,
        num_envs=1,
        updates_per_step=1,
        std_z=0.1,
        num_points=400,
        w_name="",
        gamma=0.99,
        tau=0.005,
        batch_size=2048,
        hidden_size=256,
        replay_capacity=1_000_000,
        alpha_init=0.1,
        verbose=1,
        agent_type="SAC",
        seed=None,
        n_states_rew=2,
        lr=1e-3,
        lr_final=1e-4,
        checkpoint_dir=None,
        checkpoint_prefix=None,
        checkpoint_interval=5000,
    ):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if not isinstance(device, torch.device) else device

        self.theory = theory
        self.scope = scope
        self.spins = torch.tensor(spins, device=device)
        self.dSigma = dSigma
        self.d_max = d_max
        self.init_state = torch.tensor(init_state, device=device)
        self.bound = torch.tensor(bound, device=device)
        self.max_steps = max_steps
        self.max_episode_len = max_episode_len
        self.start_steps = start_steps
        self.d_step_size = d_step_size
        self.reward_scale = reward_scale
        self.num_envs = num_envs
        self.updates_per_step = updates_per_step
        self.std_z = std_z
        self.num_points = num_points
        self.device = device

        self.n_states_rew=n_states_rew
        self.lr=lr
        self.lr_final=lr_final
        self.checkpoint_dir=checkpoint_dir
        self.checkpoint_prefix=checkpoint_prefix
        self.checkpoint_interval=checkpoint_interval

        self.state_dim = len(init_state) if state_dim is None else state_dim
        self.action_dim = len(init_state) - 1 if action_dim is None else action_dim

        self.w_name = w_name
        self.WNet = None
        if w_name:
            try:
                self.WNet = WNet().to(device)
                self.WNet.load_state_dict(torch.load(f"../z_sampling/{w_name}", weights_only=True))
                
            except:
                self.WNet = None

        self.agent = SACAgentFactory.create(
            agent_type=agent_type,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            gamma=gamma,
            tau=tau,
            lr=lr,
            batch_size=batch_size,
            hidden_size=hidden_size,
            replay_capacity=replay_capacity,
            alpha_init=alpha_init,
            device=device,
            seed=seed  # pass seed for reproducible initialization
        )
        # Set num_inactive_dim used in HPC env simulation.
        self.agent.num_inactive_dim = self.state_dim - self.action_dim
        self.verbose = verbose
        self.agent_type = agent_type

    def calc_rew_c(self,d_values, N_lsq=20, n_states_rew=2,num_points=400,std=.1):
        zs= generate_gaussian_complex_points(num_points,std, device=self.device)
        if self.WNet is None:
            return least_sq_std_rew(d_values, zs, self.spins, self.dSigma, self.d_max, N_lsq, n_states_rew)
        return least_sq_std_rew_W_func(d_values, zs, self.spins, self.dSigma, self.WNet, N_lsq, n_states_rew)

    def train(self):
        if self.verbose:
            print("Start training:")
            start_time = time.time()
        train_model_sac_batched(            
            self.agent,
            num_envs=self.num_envs,
            init_state=self.init_state,
            spins=self.spins,
            dSigma=self.dSigma,
            d_max=self.d_max,
            bound=self.bound,
            max_steps=self.max_steps,
            max_episode_len=self.max_episode_len,
            start_steps=self.start_steps,
            d_step_size=self.d_step_size,
            reward_scale=self.reward_scale,
            std_z=self.std_z,
            num_points=self.num_points,
            updates_per_step=self.updates_per_step,
            device=self.device,
            WNet=self.WNet,
            N_lsq=20, 
            n_states_rew=self.n_states_rew,
            lr=self.lr,
            lr_final=self.lr_final,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_prefix=self.checkpoint_prefix,
            checkpoint_interval=self.checkpoint_interval,
        )
        if self.verbose:
            end_time = time.time()
            print(f'Finished training in {end_time - start_time:.2f} sec (~ {(end_time - start_time) // 60} min).')
            print('----------------------------------------------')

    def predict(self, num_episodes=20, potential=True):
        if potential:
            trajectories, rews, c_means, c_stds = gather_trajectories_sac_batched_v1(
                    self.agent,
                    num_envs=self.num_envs,
                    num_episodes=num_episodes,
                    init_state=self.init_state,
                    spins=self.spins,
                    dSigma=self.dSigma,
                    d_max=self.d_max,
                    bound=self.bound,
                    n_states_rew=self.n_states_rew,
                    d_step_size=self.d_step_size,
                    reward_scale=self.reward_scale,
                    std_z=self.std_z,
                    num_points=self.num_points,
                    N_lsq=20, 
                    max_episode_len=self.max_episode_len,
                    device=self.device,
                    )
            return trajectories,rews,c_means,c_stds
        
        trajectories = gather_trajectories_sac_batched(
            self.agent,
            num_envs=self.num_envs,
            num_episodes=num_episodes,
            init_state=self.init_state,
            bound=self.bound,
            d_step_size=self.d_step_size,
            max_episode_len=self.max_episode_len,
            device=self.device,
        )

        rews=[]
        c_means=[]
        c_stds=[]
        for tr in trajectories:
            rew,c_mean,c_std=self.calc_rew_c(torch.tensor(tr,dtype=torch.float64,device=self.device))
            rews.append(rew)
            c_means.append(c_mean)
            c_stds.append(c_std)
        return trajectories,rews,c_means,c_stds

    def predict_and_plot(self, num_episodes=20, ii=1, jj=2, save_trajectories_dir="", model_name=""):
        trajectories,rews,c_means,c_stds = self.predict(num_episodes=num_episodes)
       
        if save_trajectories_dir:
            assert os.path.isdir(save_trajectories_dir)
            torch.save(trajectories, f'{save_trajectories_dir}/delta_trajectories_of_{model_name}.pt')
            torch.save(c_means, f'{save_trajectories_dir}/c_mean_trajectories_of_{model_name}.pt')
            torch.save(rews, f'{save_trajectories_dir}/rew_trajectories_of_{model_name}.pt')
            torch.save(c_stds, f'{save_trajectories_dir}/c_std_trajectories_of_{model_name}.pt')
                                          
        fig1, max_r, max_deltas = plot_r_gauss_z(
            fixed_deltas=self.init_state.cpu().numpy(),
            variable_indices=(ii, jj),
            variable_ranges=[
                [self.bound[ii][0].cpu(), self.bound[ii][1].cpu(), 0.051],
                [self.bound[jj][0].cpu(), self.bound[jj][1].cpu(), 0.051]
            ],
            spins=self.spins,
            dSigma=self.dSigma,
            d_max=self.d_max,
            num_points=400,
            N_lsq=20,
            std=self.std_z,
            WNet=self.WNet,
            device=self.device
        )
        ax1 = fig1.gca()
        ax1.set_title(f"{self.scope} search of {self.theory}:\nMaximum residual r: {max_r}\nUnfixed deltas for maximum r: {max_deltas}")
        first_leg = True
        for traj in trajectories:
            ax1.scatter(traj[0][ii], traj[0][jj], color='green', s=100, marker='o', label='Start' if first_leg else None)
            ax1.scatter(traj[-1][ii], traj[-1][jj], color='black', s=100, marker='X', label='End' if first_leg else None)
            if first_leg:
                ax1.legend()
                first_leg = False
        return fig1

    def predict_and_plot_trajectories(self, num_episodes=20, ii=1, jj=2, save_trajectories_dir="", agent_name="", model_name=""):
        trajectories,rews,c_means,c_stds = self.predict(num_episodes=num_episodes)
        if save_trajectories_dir:
            assert os.path.isdir(save_trajectories_dir)
            torch.save(trajectories, f'{save_trajectories_dir}/{agent_name}_delta_trajectories_of_{model_name}.pt')
            torch.save(c_means, f'{save_trajectories_dir}/{agent_name}_c_mean_trajectories_of_{model_name}.pt')
            torch.save(rews, f'{save_trajectories_dir}/{agent_name}_rew_trajectories_of_{model_name}.pt')
            torch.save(c_stds, f'{save_trajectories_dir}/{agent_name}_c_std_trajectories_of_{model_name}.pt')
                
        fig1, max_r, max_deltas = plot_r_gauss_z(
            fixed_deltas=self.init_state.cpu().numpy(),
            variable_indices=(ii, jj),
            variable_ranges=[
                [self.bound[ii][0].cpu(), self.bound[ii][1].cpu(), 0.051],
                [self.bound[jj][0].cpu(), self.bound[jj][1].cpu(), 0.051]
            ],
            spins=self.spins,
            dSigma=self.dSigma,
            d_max=self.d_max,
            num_points=400,
            N_lsq=20,
            std=self.std_z,
            WNet=self.WNet,
            device=self.device
        )
        ax1 = fig1.gca()
        ax1.set_title(f"{self.scope} search of {self.theory}:\nMaximum residual r: {max_r}\nUnfixed deltas for maximum r: {max_deltas}")
        first_leg = True
        for i, traj in enumerate(trajectories):
            x_coords = [state[ii] for state in traj]
            y_coords = [state[jj] for state in traj]
            ax1.plot(x_coords, y_coords, color='black', linewidth=1, alpha=0.5, label="Trajectory" if i==0 else None)
            ax1.scatter(x_coords, y_coords, color='black', s=30, alpha=0.7, label="Points" if i==0 else None)
            ax1.scatter(x_coords[0], y_coords[0], color='red', s=30, alpha=0.7, label="Points" if i==0 else None)
            ax1.scatter(x_coords[-1], y_coords[-1], color='red', s=30, alpha=0.7, label="Points" if i==0 else None)
            if first_leg:
                ax1.legend()
                first_leg = False
        return fig1
    
    def predict_and_save_traj(self, num_episodes=20, save_trajectories_dir="", agent_name="", model_name=""):
        trajectories,rews,c_means,c_stds = self.predict(num_episodes=num_episodes, potential=True)
        if save_trajectories_dir:
            assert os.path.isdir(save_trajectories_dir)
            torch.save(trajectories, f'{save_trajectories_dir}/{agent_name}_delta_trajectories_of_{model_name}.pt')
            torch.save(c_means, f'{save_trajectories_dir}/{agent_name}_c_mean_trajectories_of_{model_name}.pt')
            torch.save(rews, f'{save_trajectories_dir}/{agent_name}_rew_trajectories_of_{model_name}.pt')
            torch.save(c_stds, f'{save_trajectories_dir}/{agent_name}_c_std_trajectories_of_{model_name}.pt')

    def plot_trajectories(self, data, ii=1, jj=2):
        trajectories,rews,c_means,c_stds = data
        
        fig1, max_r, max_deltas = plot_r_gauss_z(
            fixed_deltas=self.init_state.cpu().numpy(),
            variable_indices=(ii, jj),
            variable_ranges=[
                [self.bound[ii][0].cpu(), self.bound[ii][1].cpu(), 0.051],
                [self.bound[jj][0].cpu(), self.bound[jj][1].cpu(), 0.051]
            ],
            spins=self.spins,
            dSigma=self.dSigma,
            d_max=self.d_max,
            num_points=400,
            N_lsq=20,
            std=self.std_z,
            WNet=self.WNet,
            device=self.device
        )
        ax1 = fig1.gca()
        ax1.set_title(f"{self.scope} search of {self.theory}:\nMaximum residual r: {max_r}\nUnfixed deltas for maximum r: {max_deltas}")
        first_leg = True
        for i, traj in enumerate(trajectories):
            x_coords = [state[ii] for state in traj]
            y_coords = [state[jj] for state in traj]
            ax1.plot(x_coords, y_coords, color='black', linewidth=1, alpha=0.5, label="Trajectory" if i==0 else None)
            ax1.scatter(x_coords, y_coords, color='black', s=30, alpha=0.7, label="Points" if i==0 else None)
            ax1.scatter(x_coords[0], y_coords[0], color='white', s=40, alpha=0.7, label="Start Points" if i==0 else None)
            ax1.scatter(x_coords[-1], y_coords[-1], color='red', s=40, alpha=0.7, label="End Points" if i==0 else None, zorder=10)
            if first_leg:
                ax1.legend()
                first_leg = False
        return fig1
    
    def plot_potential(self, ii, jj):
        fig1, max_r, max_deltas = plot_r_gauss_z(
            fixed_deltas=self.init_state.cpu().numpy(),
            variable_indices=(ii, jj),
            variable_ranges=[
                [self.bound[ii][0].cpu(), self.bound[ii][1].cpu(), 0.051],
                [self.bound[jj][0].cpu(), self.bound[jj][1].cpu(), 0.051]
            ],
            spins=self.spins,
            dSigma=self.dSigma,
            d_max=self.d_max,
            num_points=400,
            N_lsq=20,
            std=self.std_z,
            WNet=self.WNet,
            device=self.device
        )
        ax1 = fig1.gca()
        ax1.set_title(f"{self.scope} search of {self.theory}:\nMaximum residual r: {max_r}\nUnfixed deltas for maximum r: {max_deltas}")

        return fig1
