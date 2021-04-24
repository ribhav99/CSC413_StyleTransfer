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

    g = Generator(args).to(device)  # x -> y

    print("--Generator architecture--")
    print(g)

    print("Loading Models...")
    models = torch.load(args.model_path)
    g.load_state_dict(models['g'])
    print("Successfully Loaded Models...")

    total_g_loss = 0.0
    total_data = 0

    with torch.no_grad():
        for batch_num, data in tqdm(enumerate(full_data)):

            y, x = data[0].to(device), data[1].to(
                device)  # x is cartoon, y is human
            total_data += x.shape[0]

            fake_y = g(x).cpu().numpy()

            for i in range(args.batch_size):
                img2 = np.moveaxis(fake_y[i], 0, -1)
                io.imsave(
                    f'../results/fake_y{i}{batch_num}.png', img_as_ubyte(img2))
            del x
            del y
            del fake_y
