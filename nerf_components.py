import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth
        
        self.register_buffer('frequencies', 2.0 ** torch.linspace(0.0, bandwidth - 1, bandwidth) * np.pi)

    def forward(self, x):
        out = [x]
        for frequency in self.frequencies:
            out.append(torch.sin(x * frequency)) 
            out.append(torch.cos(x * frequency))
        return torch.cat(out, dim=-1)


def get_rays(H, W, focal, c2w):
    
    device = c2w.device
    
    u = torch.arange(0, W, 1, device=device).float()
    v = torch.arange(0, H, 1, device=device).float()
    X, Y = torch.meshgrid(u, v, indexing='xy')
    
    directions = torch.stack([(X-W*.5)/focal, -(Y-H*.5)/focal, -torch.ones_like(X)], -1)
    
    
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], -1)
    
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d

def sampler_coarse(vector_o, vector_d, t_near, t_far, no_of_samples: int, jitter=True):
    device = vector_o.device 
    
    t_values = torch.linspace(t_near, t_far, no_of_samples, device=device)
    batch_shape = vector_o.shape[:-1]
    t_values = t_values.expand(batch_shape + (no_of_samples,))

    midpoint_of_bins = 0.5 * (t_values[..., 1:] + t_values[..., :-1])
    upper_boundary_of_bins = torch.cat([midpoint_of_bins, t_values[..., -1:]], -1)
    lower_boundary_of_bins = torch.cat([t_values[..., :1], midpoint_of_bins], -1)

    if jitter:
        t_rand = torch.rand(t_values.shape, device=device)
    else:
        t_rand = 0.5 

    mus = lower_boundary_of_bins + (upper_boundary_of_bins - lower_boundary_of_bins) * t_rand
    
    
    sampled_points = vector_o[..., None, :] + mus[..., :, None] * vector_d[..., None, :]
    
    return sampled_points, mus

def sampler_fine(vector_o, vector_d, weights, mus_coarse):
    device = vector_o.device 
    
    total_weight = torch.sum(weights, -1)
    weight_distribution = weights / (total_weight[..., None] + 1e-5)

    mus_with_weights = (weight_distribution > 1e-5) * mus_coarse
    mask = mus_with_weights > 0

    search_for_min = mus_with_weights.clone()
    search_for_min[~mask] = float('inf')
    min_values = torch.min(search_for_min, dim=1).values[..., None]

    search_for_max = mus_with_weights.clone()
    search_for_max[~mask] = float('-inf')
    max_values = torch.max(search_for_max, dim=1).values[..., None]

    hit_nothing = torch.isinf(min_values)
    
    
    min_values = torch.where(hit_nothing, torch.tensor(2.0, device=device), min_values)
    max_values = torch.where(hit_nothing, torch.tensor(6.0, device=device), max_values)

    N_rays = vector_o.shape[0]
    z_fine = min_values + (max_values - min_values) * torch.rand(N_rays, 128, device=device)

    final_concatenated_mus = torch.sort(torch.cat((mus_coarse, z_fine), dim=-1))[0]
    sampled_points_fine = (vector_o[..., None, :] + final_concatenated_mus[..., :, None] * vector_d[..., None, :])

    return sampled_points_fine, final_concatenated_mus

def raw2outputs(raw, z_vals, rays_d, White_background=True):
    device = raw.device 
    
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    
    
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[..., :1].shape)], -1)
    
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    sigma = F.relu(raw[..., 3])

    alpha = 1.0 - torch.exp(-sigma * dists)
    weights_parts = torch.cumprod(1.0 - alpha + 1e-10, -1)
    transmittance = torch.cat([torch.ones_like(weights_parts[..., :1]), weights_parts[..., :-1]], -1)

    weights = transmittance * alpha
    acc_map = torch.sum(weights, -1)

    rgb_map = torch.sum(weights[..., None] * rgb, -2)

    if White_background:
        transparency = 1.0 - acc_map[..., None]
        rgb_map = rgb_map + transparency

    depth_map = torch.sum(weights * z_vals, -1)
    return rgb_map, depth_map, acc_map, weights