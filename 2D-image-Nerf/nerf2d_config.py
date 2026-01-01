from dataclasses import dataclass
@dataclass
class NeRF2D_config:
    #hyperparameter
    epochs:int=10000
    learning_rate:float=0.1
    L_pos:int=10

    dataset_path:str ='/Users/saurabhgiri/Downloads/butterfly.jpeg'
    save_path:str='./checkpoints'
    model_namev1:str='nerf_2d_model_without_positional_encoding'
    model_namev2:str='nerf_2d_model_with_positional_encoding'