import numpy as np
import torch
import matplotlib.pyplot as plt

from .env import least_sq_std_rew, least_sq_std_rew_W_func
# from ..z_sampling.rew_func_cuda import least_sq_std_rew_W_func


def rho(z):
    return z / (1 + torch.sqrt(1 - z))**2

def lambda_z(z):
    return torch.abs(rho(z)) + torch.abs(rho(1 - z))

def generate_random_points(lambda_0=0.42, x_range=[0.51, 1.5], y_range=[0, 1.16], num_points=200, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    valid_points = []
    while len(valid_points) < num_points:
        real_part = torch.rand(num_points, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        imag_part = torch.rand(num_points, device=device) * (y_range[1] - y_range[0]) + y_range[0]
        z_batch = torch.complex(real_part, imag_part)
        mask = lambda_z(z_batch) < lambda_0
        valid_points.append(z_batch[mask])
    valid_points = torch.cat(valid_points)
    return valid_points[:num_points]

def generate_gaussian_complex_points(n, mean=0.5, std=0.1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    real = torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64)
    imag = torch.normal(mean=0.0, std=std, size=(n,),device=device,dtype=torch.float64)

    return torch.complex(real, imag)

def plot_r(
        fixed_deltas, variable_indices, variable_ranges, spins, dSigma, d_max,
        std=0.1,num_points=400, N_lsq=20, n_states_rew=2, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        WNet = None,
):
    rew = least_sq_std_rew if WNet==None else least_sq_std_rew_W_func
    zs_gpu= generate_gaussian_complex_points(num_points,std=std, device=device)

    var1_idx,var2_idx= variable_indices
    var1_range=np.arange(*variable_ranges[0])
    var2_range=np.arange(*variable_ranges[1])
    Var1,Var2=np.meshgrid(var1_range,var2_range,indexing='ij')
    r_matrix=np.zeros_like(Var1,dtype=np.float64)
    d_list=[]

    for i,var1 in enumerate(var1_range):
        for j,var2 in enumerate(var2_range):
            dd=fixed_deltas.copy()
            dd[var1_idx]=var1
            dd[var2_idx]=var2
            d_list.append(torch.tensor(dd,dtype=torch.float64,device=device))

    d_stack= torch.stack(d_list,dim=0)
    r_flat,_,_= rew(
        d_stack, zs_gpu, spins, dSigma, d_max if WNet == None else WNet, N_lsq=N_lsq, n_states_rew=n_states_rew
    )
    r_matrix= r_flat.cpu().numpy().reshape(Var1.shape)

    fig,axis= plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(10)
    cp= axis.contourf(Var1, Var2, r_matrix, levels=50, cmap='viridis')
    fig.colorbar(cp,label='Reward')
    axis.set_xlabel(f"$d_{{{var1_idx+1}}}$")
    axis.set_ylabel(f"$d_{{{var2_idx+1}}}$")
    axis.set_title(f"Reward for $d_{{{var1_idx+1}}}$ vs $d_{{{var2_idx+1}}}$")

    max_r= np.max(r_matrix)
    max_idx= np.unravel_index(np.argmax(r_matrix), r_matrix.shape)
    max_deltas= fixed_deltas.copy()
    max_deltas[var1_idx]=Var1[max_idx]
    max_deltas[var2_idx]=Var2[max_idx]

    # print(f"Maximum residual r: {max_r}")
    # print(f"Unfixed deltas for maximum r: {max_deltas}")
    return fig, max_r, max_deltas

def plot_r_gauss_z(
        fixed_deltas, variable_indices, variable_ranges, spins, dSigma, d_max,
        std=0.1,num_points=400, N_lsq=20, n_states_rew=2, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        WNet = None,
):
    rew = least_sq_std_rew if WNet==None else least_sq_std_rew_W_func
    # zs_gpu= generate_gaussian_complex_points(num_points,std=std, device=device)

    var1_idx,var2_idx= variable_indices
    var1_range=np.arange(*variable_ranges[0])
    var2_range=np.arange(*variable_ranges[1])
    Var1,Var2=np.meshgrid(var1_range,var2_range,indexing='ij')
    r_matrix=np.zeros_like(Var1,dtype=np.float64)
    d_list=[]
    #print(dSigma, variable_ranges)
    #print(var1_range, var2_range)
    for i,var1 in enumerate(var1_range):
        for j,var2 in enumerate(var2_range):
                     
            dd=fixed_deltas.copy()
            if (var1==var2 and spins[[var1_idx]]==spins[[var2_idx]]) :
                var1+=.01
                 
            dd[var1_idx]=var1
            dd[var2_idx]=var2
          
            d_list.append(torch.tensor(dd,dtype=torch.float64,device=device))
            
    d_stack= torch.stack(d_list,dim=0)
    #r_flat, _,_= rew(
    #    d_stack, zs_gpu, spins, dSigma, d_max if WNet == None else WNet, N_lsq=N_lsq, n_states_rew=n_states_rew
    #)
    #r_matrix= r_flat.cpu().numpy().reshape(Var1.shape)
    # 進行多次采樣，然後取平均
    sample_count = 5
    r_flat_samples = []
    for k in range(sample_count):
        # 每次新采樣一組 z_gpu
        z_gpu = generate_gaussian_complex_points(num_points, std=std, device=device)
        r_flat_k, _, _ = rew(
            d_stack, z_gpu, spins, dSigma, d_max if WNet is None else WNet,
            N_lsq=N_lsq, n_states_rew=n_states_rew
        )
        r_flat_samples.append(r_flat_k)
    # 取平均得到最終的 r_flat
    r_flat_avg = torch.stack(r_flat_samples, dim=0).mean(dim=0)
    r_matrix = r_flat_avg.cpu().numpy().reshape(Var1.shape)

    fig,axis= plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(10)
    cp= axis.contourf(Var1, Var2, r_matrix, levels=50, cmap='viridis')
    fig.colorbar(cp,label='Reward')
    axis.set_xlabel(f"$d_{{{var1_idx+1}}}$")
    axis.set_ylabel(f"$d_{{{var2_idx+1}}}$")
    axis.set_title(f"Reward for $d_{{{var1_idx+1}}}$ vs $d_{{{var2_idx+1}}}$")

    axis.axhline(y=(variable_ranges[1][0]+variable_ranges[1][1])/2, color='white', linestyle='--', linewidth=2, label='Midpoint d_{}'.format(var1_idx+1))
    axis.axvline(x=(variable_ranges[0][0]+variable_ranges[0][1])/2, color='white', linestyle='--', linewidth=2, label='Midpoint d_{}'.format(var2_idx+1))

    max_r= np.max(r_matrix)
    max_idx= np.unravel_index(np.argmax(r_matrix), r_matrix.shape)
    max_deltas= fixed_deltas.copy()
    max_deltas[var1_idx]=Var1[max_idx]
    max_deltas[var2_idx]=Var2[max_idx]

    # print(f"Maximum residual r: {max_r}")
    # print(f"Unfixed deltas for maximum r: {max_deltas}")
    return fig, max_r, max_deltas

