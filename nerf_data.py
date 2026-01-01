import os
import json
import numpy as np
import imageio.v2 as imageio
import torch

def load_blender_data(basedir, half_res=False):
    print(f"Loading Data from {basedir}...")
    
    # Ensure basedir exists
    if not os.path.exists(basedir):
        raise FileNotFoundError(f"Dataset not found at {basedir}. Please check the path in nerf_config.py")

    json_path = os.path.join(basedir, 'transforms_train.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, 'r') as fp:
        meta = json.load(fp)

    all_imgs = []
    all_poses = []
    
    print(f"Found {len(meta['frames'])} frames. Processing...")
    
    for frame in meta['frames']:
        fname = os.path.join(basedir, frame['file_path'] + '.png')
        
        if not os.path.exists(fname):
            print(f"Warning: Image {fname} not found. Skipping.")
            continue

        img = imageio.imread(fname)
        pose = np.array(frame['transform_matrix'])
        
        img = (np.array(img) / 255.).astype(np.float32)
        
        if img.shape[2] == 4:
            img = img[..., :3] * img[..., 3:] + (1. - img[..., 3:])
            
        if half_res:
            img = img[::2, ::2]
            
        all_imgs.append(img)
        all_poses.append(pose)

    if len(all_imgs) == 0:
        raise RuntimeError("No images were loaded! Check your data path.")

    imgs = torch.from_numpy(np.stack(all_imgs)).float()
    poses = torch.from_numpy(np.stack(all_poses)).float()
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    print(f"Loaded {imgs.shape[0]} images at Resolution: {H}x{W}")
    
    return imgs, poses, focal