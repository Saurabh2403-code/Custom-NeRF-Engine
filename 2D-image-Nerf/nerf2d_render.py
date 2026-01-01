import torch
from nerf2d_model import networkv1,networkv2
import numpy as np
from PIL import Image
from nerf2d_config import NeRF2D_config
from nerf2d_components import PositionalEncoding

encoder=PositionalEncoding(10)
cfg=NeRF2D_config()

x = torch.arange(0, 1, 0.01)
y = torch.arange(0, 1, 0.01)
X, Y = np.meshgrid(x, y)
co_ordinates = torch.from_numpy(np.stack([X.flatten(), Y.flatten()], axis=-1)).type(torch.float32)
fourier_co_ordinates=encoder(co_ordinates)


image = Image.open(cfg.dataset_path).resize((100, 100))


def renderv1():
    model=networkv1()
    checkpoint = torch.load('./checkpoints/nerf_2d_model_without_positional_encoding_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.inference_mode():
        image_learnt=model(co_ordinates).reshape(100,100,3)
    image_learnt=image_learnt.detach().numpy()
    image_learnt = (image_learnt * 255).astype(np.uint8)

    learnt_image = Image.fromarray(image_learnt)
    error=image_learnt-image
    error=Image.fromarray(error)
    error_name='error_without_positional_encoding.png'
    error.save(error_name)
    save_name = "Image Learnt Without Positional Encoding.png"
    learnt_image.save(save_name)
def renderv2():
    model=networkv2()
    checkpoint=torch.load('./checkpoints/nerf_2d_model_with_positional_encoding_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.inference_mode():
        image_learnt=model(fourier_co_ordinates).reshape(100,100,3)
        image_learnt=image_learnt.detach().numpy()
    image_learnt = (image_learnt * 255).astype(np.uint8)

    learnt_image = Image.fromarray(image_learnt)
    error=image_learnt-image
    error=Image.fromarray(error)
    error_name='error_with_positional_encoding.png'
    error.save(error_name)
    save_name = "Image Learnt With Positional Encoding.png"
    learnt_image.save(save_name)

if __name__=='__main__':
    renderv2()

