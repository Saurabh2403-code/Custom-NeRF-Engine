import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import glob # Needed for finding latest checkpoint

# --- IMPORT MODULES ---
from nerf_config import NerfConfig
from nerf_model import NeRF_ViewDependent 
from nerf_components import (
    PositionalEncoding, 
    sampler_coarse, 
    sampler_fine, 
    raw2outputs,
    get_rays
)
from nerf_data import load_blender_data

def train():
    # 1. SETUP
    cfg = NerfConfig()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Training on: {device}")
    
    os.makedirs(cfg.save_path, exist_ok=True)

    # 2. LOAD DATA
    images, poses, focal = load_blender_data(cfg.dataset_path, half_res=cfg.half_res)
    images = images.to(device)
    poses = poses.to(device)
    focal = torch.tensor(focal).float().to(device)
    
    # 3. PRE-CALCULATE RAYS
    print("Generating Rays for all images...")
    
    all_rays_o = []
    all_rays_d = []
    all_pixels = []
    
    H, W = images[0].shape[:2]
    
    for i in range(images.shape[0]):
        ro, rd = get_rays(H, W, focal, poses[i])
        all_rays_o.append(ro.reshape(-1, 3))
        all_rays_d.append(rd.reshape(-1, 3))
        all_pixels.append(images[i].reshape(-1, 3))
        
    all_rays_o = torch.cat(all_rays_o, 0)
    all_rays_d = torch.cat(all_rays_d, 0)
    all_pixels = torch.cat(all_pixels, 0)
    
    print(f"Total Rays: {all_rays_o.shape[0]}")

    # 4. INITIALIZE MODEL
    dim_pos = 3 + 2 * 3 * cfg.L_pos
    dim_dir = 3 + 2 * 3 * cfg.L_dir
    
    model = NeRF_ViewDependent(embedding_dim_pos=dim_pos, embedding_dim_dir=dim_dir).to(device)
    
    encoder_pos = PositionalEncoding(cfg.L_pos).to(device)
    encoder_dir = PositionalEncoding(cfg.L_dir).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    loss_fn = nn.MSELoss()

    # --- AUTO RESUME LOGIC ---
    start_epoch = 0
    # Check for existing checkpoints
    checkpoints = glob.glob(os.path.join(cfg.save_path, "nerf_weights_*.pth"))
    
    if os.path.exists(os.path.join(cfg.save_path, "nerf_final.pth")):
        # Resume from final
        print("Resuming from nerf_final.pth...")
        model.load_state_dict(torch.load(os.path.join(cfg.save_path, "nerf_final.pth")))
        # We don't know the exact epoch, so we just append more training
        start_epoch = 0 
    elif checkpoints:
        # Resume from latest numbered checkpoint
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        print(f"Resuming from {latest_ckpt}...")
        model.load_state_dict(torch.load(latest_ckpt))
        # Extract epoch number from filename "nerf_weights_1000.pth"
        try:
            start_epoch = int(latest_ckpt.split('_')[-1].split('.')[0])
            print(f"Resuming at Epoch {start_epoch}")
        except:
            start_epoch = 0
    else:
        print("No checkpoint found. Starting from scratch.")

    # 5. TRAINING LOOP
    print(f"Starting training loop...")
    
    # We train for cfg.epochs MORE iterations from wherever we started
    # Or, if you want total iterations to cap at cfg.epochs, change range to:
    # range(start_epoch, cfg.epochs)
    
    for i in tqdm(range(start_epoch, cfg.epochs + start_epoch)):
        
        # --- Batching ---
        idxs = torch.randint(0, all_rays_o.shape[0], (cfg.batch_size,))
        batch_o = all_rays_o[idxs]
        batch_d = all_rays_d[idxs]
        target = all_pixels[idxs]
        
        # --- Coarse Pass ---
        pts_c, mus_c = sampler_coarse(batch_o, batch_d, 2.0, 6.0, cfg.n_coarse)
        
        flat_pts_c = pts_c.reshape(-1, 3)
        enc_pts_c = encoder_pos(flat_pts_c)
        
        d_exp_c = batch_d.unsqueeze(1).expand(-1, cfg.n_coarse, -1).reshape(-1, 3)
        enc_dir_c = encoder_dir(d_exp_c)
        
        raw_c = model(enc_pts_c, enc_dir_c).reshape(cfg.batch_size, cfg.n_coarse, 4)
        rgb_c, _, _, weights_c = raw2outputs(raw_c, mus_c, batch_d, cfg.white_bkgd)
        
        # --- Fine Pass ---
        pts_f, mus_f = sampler_fine(batch_o, batch_d, weights_c, mus_c)
        
        flat_pts_f = pts_f.reshape(-1, 3)
        enc_pts_f = encoder_pos(flat_pts_f)
        
        n_total = mus_f.shape[1] 
        d_exp_f = batch_d.unsqueeze(1).expand(-1, n_total, -1).reshape(-1, 3)
        enc_dir_f = encoder_dir(d_exp_f)
        
        raw_f = model(enc_pts_f, enc_dir_f).reshape(cfg.batch_size, n_total, 4)
        rgb_f, _, _, _ = raw2outputs(raw_f, mus_f, batch_d, cfg.white_bkgd)
        
        # --- Loss ---
        loss = loss_fn(rgb_c, target) + loss_fn(rgb_f, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # --- Logging & Saving ---
        if i % 100 == 0:
            tqdm.write(f"Iter {i} | Loss: {loss.item():.5f}")
            
        if i % 1000 == 0 and i > 0:
            path = os.path.join(cfg.save_path, f"nerf_weights_{i}.pth")
            torch.save(model.state_dict(), path)
            tqdm.write(f"Saved checkpoint to {path}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(cfg.save_path, "nerf_final.pth"))
    print("Training Complete.")

if __name__ == "__main__":
    train()