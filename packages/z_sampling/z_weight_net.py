import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from .rew_func_cuda import precompute_G_v, least_sq_std_rew_with_W_G, compute_W_v

class WNet(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim1=64, hidden_dim2=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim1,dtype=torch.float64)
        self.fc2 = torch.nn.Linear(hidden_dim1, hidden_dim2,dtype=torch.float64)
        self.fc3 = torch.nn.Linear(hidden_dim2, 1,dtype=torch.float64)

    def forward(self, x,y):
        x = x.unsqueeze(-1) if x.dim() == 1 else x  # Convert (N,) → (N,1)
        y = y.unsqueeze(-1) if y.dim() == 1 else y  # Convert (N,) → (N,1)
        z = torch.cat([x,y], dim=-1)
        x = torch.relu(self.fc1(z))
        w = self.fc2(x)
        w = torch.relu(w)
        w= self.fc3(w)
        # Use F.softmax 
        #w = F.softmax(w,dim=0)
        w=torch.abs(w)
        return w.squeeze(-1) 
    



def plot_r(fixed_deltas, variable_indices, variable_ranges, spins, dSigma,W, x_values, y_values,N_lsq=10,n_states_rew=2):
    """
    Plots the residuals as a contour plot for any two deltas while keeping the others fixed.

    Parameters:
    - fixed_deltas: List of 4 deltas [d1, d2, d3, d4] with placeholders (-1) for the variable deltas.
    - variable_indices: Tuple of two indices (i, j) indicating which deltas to vary.
    - variable_ranges: List of two ranges [[start1, end1, step1], [start2, end2, step2]] for the variables.
    - spins: List of spin values.
    - dSigma: The dSigma parameter for the least squares computation.
    """
    var1_idx, var2_idx = variable_indices
    var1_range = np.arange(*variable_ranges[0])
    var2_range = np.arange(*variable_ranges[1])
    
    # Prepare meshgrid for the two variable deltas
    Var1, Var2 = np.meshgrid(var1_range, var2_range, indexing='ij')
    r_matrix = np.zeros_like(Var1, dtype=np.float64)
    d_arr = []

    # Generate all deltas combinations
    for i, var1 in enumerate(var1_range):
        for j, var2 in enumerate(var2_range):
            deltas = fixed_deltas.copy()
            deltas[var1_idx] = var1
            deltas[var2_idx] = var2
            d_arr.append(deltas)


    
 
    #zs= generate_complex_gaussian(200, .1)
    # Compute residuals using least_sq_rew
    G, v = precompute_G_v(np.array(d_arr), spins, x_values, y_values, dSigma)
    r_matrix_flat = least_sq_std_rew_with_W_G(W,G,v,N_lsq=N_lsq,n_states_rew=n_states_rew)
    
    
    r_matrix = r_matrix_flat.reshape(Var1.shape).detach().cpu().numpy()

    # Plot the results
    plt.figure(figsize=(10, 8))
    cp = plt.contourf(Var1, Var2, r_matrix, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Residual r')
    plt.xlabel(f'$d_{var1_idx + 1}$')
    plt.ylabel(f'$d_{var2_idx + 1}$')
    plt.title(f'Residual $r$ for $d_{var1_idx + 1}$ vs $d_{var2_idx + 1}$')
    plt.show()
   
    max_r = np.max(r_matrix)
    max_idx = np.unravel_index(np.argmax(r_matrix), r_matrix.shape)
    max_deltas = fixed_deltas.copy()
    max_deltas[var1_idx] = Var1[max_idx]
    max_deltas[var2_idx] = Var2[max_idx]

    print(f"Maximum residual r: {max_r}")
    print(f"Unfixed deltas for maximum r: {max_deltas}")
