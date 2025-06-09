import torch 

def generate_gaussian_real_points(n, mean=0.5, std=0.1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    def sample_clipped(n_required):
        samples = []
        while len(samples) < n_required:
            s = torch.normal(mean=mean, std=std, size=(n,),device=device,dtype=torch.float64)
            s = s[(s>=0.0)&(s<=1.0)]
            samples.append(s)
        return torch.cat(samples)[:n_required]
    
    x = sample_clipped(n)
    y = sample_clipped(n)
    return x,y