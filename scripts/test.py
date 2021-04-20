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

    g_x_y = Generator(args).to(device)  # x -> y
    g_y_x = Generator(args).to(device)  # y -> x
    print("--Generator architecture--")
    print(g_y_x)

    print("Loading Models...")
    models = torch.load(args.model_path)
    g_x_y.load_state_dict(models['g_x_y'])
    g_y_x.load_state_dict(models['g_y_x'])
    print("Successfully Loaded Models...")

    # d_x.train()
    # d_y.train()
    # g_x_y.train()
    # g_y_x.train()

    total_d_loss = 0.0
    total_g_x_y_loss = 0.0
    total_g_y_x_loss = 0.0
    total_data = 0

    with torch.no_grad():
        for batch_num, data in tqdm(enumerate(full_data)):

            y, x = data[0].to(device), data[1].to(
                device)  # x is cartoon, y is human
            total_data += x.shape[0]

            fake_x = g_y_x(y).cpu().numpy()
            fake_y = g_x_y(x).cpu().numpy()

            for i in range(args.batch_size):
                img1 = np.moveaxis(fake_x[i], 0, -1)
                io.imsave(
                    f'../results/fake_x{i}{batch_num}.png', img_as_ubyte(img1))

                img2 = np.moveaxis(fake_y[i], 0, -1)
                io.imsave(
                    f'../results/fake_y{i}{batch_num}.png', img_as_ubyte(img2))
            del x
            del y
            del fake_x
            del fake_y
