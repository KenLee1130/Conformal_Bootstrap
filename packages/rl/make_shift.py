import torch

def make_shifted_h1(states: torch.Tensor,
                   h_step_obs: float,
                   fixed_vars: list[int] = []) -> torch.Tensor:
    """
    states     : (num_envs, num_h)
    h_step_obs : scalar step size for the shift
    fixed_vars : list of indices to NOT shift

    returns h : (num_envs, 2*num_unfixed, num_h)
                where h[i, 2*j   ] = states[i] + h_step_obs * e_j (for unfixed j)
                      h[i, 2*j+1] = states[i] - h_step_obs * e_j (for unfixed j)
    """
    B, num_h = states.shape
    device, dtype = states.device, states.dtype

    # Indices to shift
    unfixed_vars = [j for j in range(num_h) if j not in fixed_vars]
    num_unfixed = len(unfixed_vars)

    # 1. Build ±offsets only for unfixed axes -> (2*num_unfixed, num_h)
    eye = torch.eye(num_h, device=device, dtype=dtype)
    offsets = []
    for j in unfixed_vars:
        offsets.append(eye[j])
        offsets.append(-eye[j])
    offsets = torch.stack(offsets, dim=0) * h_step_obs  # (2*num_unfixed, num_h)

    # 2. Expand states to match offsets -> (B, 2*num_unfixed, num_h)
    base = states.unsqueeze(1).repeat(1, offsets.size(0), 1)

    # 3. Add the offsets
    h = base + offsets.unsqueeze(0)  # (B, 2*num_unfixed, num_h)

    # 4. Concatenate original states at the top
    return torch.cat((states, h.reshape(-1, states.size(1))), dim=0)