import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from PIL import Image
import os

# Import your modules
from nerf2d_model import networkv1
from nerf2d_config import NeRF2D_config

# Initialize Config
cfg = NeRF2D_config()

# Ensure the save directory exists
if cfg.save_path and not os.path.exists(cfg.save_path):
    os.makedirs(cfg.save_path)
    print(f"Created checkpoint directory: {cfg.save_path}")

# --- Data Preparation (Same as before) ---
image = Image.open(cfg.dataset_path).resize((100, 100))
image_array = np.array(image) / 255.0 
image_combined = image_array.reshape(-1, 3)
color_target = torch.tensor(image_combined).type(torch.float32)

x = torch.arange(0, 1, 0.01)
y = torch.arange(0, 1, 0.01)
X, Y = np.meshgrid(x, y)
co_ordinates = torch.from_numpy(np.stack([X.flatten(), Y.flatten()], axis=-1)).type(torch.float32)

# --- Model Setup ---
torch.manual_seed(42)
model = networkv1()
optimizer = optim.SGD(params=model.parameters(), lr=cfg.learning_rate)
loss_fn = nn.L1Loss()

def save_checkpoint(epoch, final=False):
    if final:
        filename = f"{cfg.model_namev1}_final.pth"
        print("saved_full_model")
    else:
        filename = f"{cfg.model_namev1}_epoch_{epoch}.pth"
    
    save_full_path = os.path.join(cfg.save_path, filename)
    
    # It is often good practice to save optimizer state as well if you plan to resume training
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_fn
    }
    
    torch.save(checkpoint, save_full_path)
def train():
    for epoch in range(cfg.epochs):
        model.train()

        color_pred = model(co_ordinates)
        loss = loss_fn(color_pred, color_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%1000==0:
            print(loss)
        if epoch % 5000 == 0:
            save_checkpoint(epoch)
            
    save_checkpoint(cfg.epochs, final=True)
    print("Training Complete.")

if __name__ == "__main__":
    train()

