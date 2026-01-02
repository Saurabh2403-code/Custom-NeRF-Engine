from dataclasses import dataclass

@dataclass
class NerfConfig:
    batch_size: int = 1024
    epochs: int = 50000
    learning_rate: float = 5e-4
    

    L_pos: int = 6
    L_dir: int = 4
    

    n_coarse: int = 64
    n_fine: int = 128

    half_res: bool = True
    white_bkgd: bool = True
    chunk_size: int = 4096 
    
    dataset_path: str = 'nerf_synthetic/lego'  
    save_path: str = './checkpoints' 
    video_path: str = './videos'
