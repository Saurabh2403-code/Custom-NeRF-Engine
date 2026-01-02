import torch
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import os
import json

from nerf_config import NerfConfig
from nerf_model import NeRF_ViewDependent
from nerf_components import (
    PositionalEncoding, sampler_coarse, sampler_fine, raw2outputs, get_rays
)

def render_video():
    cfg = NerfConfig()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Rendering on: {device}")

   
    dim_pos = 3 + 2 * 3 * cfg.L_pos
    dim_dir = 3 + 2 * 3 * cfg.L_dir
    model = NeRF_ViewDependent(embedding_dim_pos=dim_pos, embedding_dim_dir=dim_dir).to(device)
    
    ckpt_path = os.path.join(cfg.save_path, "nerf_final.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Weights not found at {ckpt_path}. Run train.py first!")
        return

    print(f"Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    encoder_pos = PositionalEncoding(cfg.L_pos).to(device)
    encoder_dir = PositionalEncoding(cfg.L_dir).to(device)


    json_path = os.path.join(cfg.dataset_path, 'transforms_train.json')
    with open(json_path, 'r') as fp:
        meta = json.load(fp)
    

    test_img = imageio.imread(os.path.join(cfg.dataset_path, meta['frames'][0]['file_path'] + '.png'))
    H, W = test_img.shape[:2]
    if cfg.half_res:
        H, W = H // 2, W // 2
        
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    print(f"Detected Resolution: {H}x{W} | Focal: {focal:.2f}")

    def pose_spherical(theta, phi, radius):
        trans_t = lambda t : np.array([[1,0,0,0], [0,1,0,0], [0,0,1,t], [0,0,0,1]], dtype=float)
        rot_phi = lambda phi : np.array([[1,0,0,0], [0,np.cos(phi),-np.sin(phi),0], [0,np.sin(phi),np.cos(phi),0], [0,0,0,1]], dtype=float)
        rot_theta = lambda th : np.array([[np.cos(th),0,-np.sin(th),0], [0,1,0,0], [np.sin(th),0,np.cos(th),0], [0,0,0,1]], dtype=float)
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return torch.from_numpy(c2w).float()

    frames = []
    print("Generating frames...")
    
    for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
        c2w = pose_spherical(th, -30., 4.0)
        with torch.no_grad():
            rays_o, rays_d = get_rays(H, W, focal, c2w)
            rays_o = rays_o.reshape(-1, 3).to(device)
            rays_d = rays_d.reshape(-1, 3).to(device)
            
            chunk_size = cfg.chunk_size 
            all_rgb = []
            
            for i in range(0, rays_o.shape[0], chunk_size):
                batch_o = rays_o[i:i+chunk_size]
                batch_d = rays_d[i:i+chunk_size]
                
                pts_c, mus_c = sampler_coarse(batch_o, batch_d, 2.0, 6.0, cfg.n_coarse, jitter=False)
                enc_pts_c = encoder_pos(pts_c.reshape(-1, 3))
                d_exp_c = batch_d.unsqueeze(1).expand(-1, cfg.n_coarse, -1).reshape(-1, 3)
                enc_dir_c = encoder_dir(d_exp_c)
                
                raw_c = model(enc_pts_c, enc_dir_c).reshape(-1, cfg.n_coarse, 4)
                _, _, _, weights_c = raw2outputs(raw_c, mus_c, batch_d, cfg.white_bkgd)
                
                pts_f, mus_f = sampler_fine(batch_o, batch_d, weights_c, mus_c)
                enc_pts_f = encoder_pos(pts_f.reshape(-1, 3))
                n_total = mus_f.shape[1]
                d_exp_f = batch_d.unsqueeze(1).expand(-1, n_total, -1).reshape(-1, 3)
                enc_dir_f = encoder_dir(d_exp_f)
                
                raw_f = model(enc_pts_f, enc_dir_f).reshape(-1, n_total, 4)
                rgb_f, _, _, _ = raw2outputs(raw_f, mus_f, batch_d, cfg.white_bkgd)
                all_rgb.append(rgb_f.cpu())

            img = torch.cat(all_rgb, 0).reshape(H, W, 3).numpy()
            frames.append((255*np.clip(img,0,1)).astype(np.uint8))

    video_path = os.path.join(cfg.video_path, 'lego_360_50k.gif')
    os.makedirs(cfg.video_path, exist_ok=True)
    imageio.mimwrite(video_path, frames, fps=30, quality=7)
    print(f"Done! Video saved to {video_path}")

if __name__ == "__main__":
    render_video()
