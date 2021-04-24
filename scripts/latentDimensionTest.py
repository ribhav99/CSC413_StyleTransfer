import torch
import torch.nn as nn
from GeneratorAndDiscriminator import Generator
from dataloader import get_data_loader
from tqdm import tqdm
import skimage.io as io
from skimage import img_as_ubyte
import numpy as np


def test(args, device):
    full_data = get_data_loader(args)

    g = Generator(args).to(device)

    print("--Generator architecture--")
    print(g)

    print("Loading Models...")
    models = torch.load(args.model_path)
    g.load_state_dict(models['g'])
    print("Successfully Loaded Models...")

    total_g_loss = 0.0
    total_data = 0

    N = 50

    x = torch.normal(0, 1, size=(
        N, args.channel_list[0], args.image_dim[0], args.image_dim[1])).to(device)
    fake_y = g(x).cpu().detach().numpy()

    for i in range(N):
        img2 = np.moveaxis(fake_y[i], 0, -1)
        io.imsave(
            f'../results/fake_y{i}.png', img_as_ubyte(img2))

    print(img2)
