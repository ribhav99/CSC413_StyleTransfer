import torch
import torch.nn as nn
import torch.optim as optim
from Generator import Generator
from Discriminator import Discriminator
from dataloader import get_data_loader
from tqdm import tqdm, trange
from datetime import datetime


def train(args, device):
    full_data = get_data_loader(args)

    d_x = Discriminator(args).to(device)  # x is horse
    d_y = Discriminator(args).to(device)  # y is zebra
    print("--Discriminator architecture--")
    print(d_x)
    g_x_y = Generator(args).to(device)  # x -> y
    g_y_x = Generator(args).to(device)  # y -> x
    print("--Generator architecture--")
    print(g_y_x)

    optimiser_d_x = optim.Adam(d_x.parameters(), lr=args.dis_learning_rate)
    optimiser_d_y = optim.Adam(d_y.parameters(), lr=args.dis_learning_rate)
    optimiser_g_x_y = optim.Adam(g_x_y.parameters(), lr=args.gen_learning_rate)
    optimiser_g_y_x = optim.Adam(g_y_x.parameters(), lr=args.gen_learning_rate)

    def cycle_loss(real, reconstructed):
        loss = (torch.abs(real - reconstructed)).mean()
        return args.lambda_cycle * loss

    d_x.train()
    d_y.train()
    g_x_y.train()
    g_y_x.train()

    if args.buffer_train:
        if args.gray:
            dimensions = (1, 128, 128)
        else:
            dimensions = (3, 128, 128)
        bufferX = torch.zeros(
            10, dimensions[0], dimensions[1], dimensions[2]).to(device)
        bufferY = torch.zeros(
            10, dimensions[0], dimensions[1], dimensions[2]).to(device)
        samplesX = torch.zeros(
            10, dimensions[0], dimensions[1], dimensions[2]).to(device)
        samplesY = torch.zeros(
            10, dimensions[0], dimensions[1], dimensions[2]).to(device)
        sampling = args.batch_size//2
    print("Start Training....")
    for epoch in trange(args.num_epochs):
        total_d_loss = 0.0
        total_g_x_y_loss = 0.0
        total_g_y_x_loss = 0.0
        total_data = 0
        if args.decay:
            dis_lr = args.dis_learning_rate - \
                ((args.dis_learning_rate / 100) * (epoch + 40))
            gen_lr = args.gen_learning_rate - \
                ((args.gen_learning_rate / 100) * (epoch + 40))

            for l in range(len(optimiser_d_x.param_groups)):
                optimiser_d_x.param_groups[l]['lr'] = dis_lr

            for l in range(len(optimiser_d_y.param_groups)):
                optimiser_d_y.param_groups[l]['lr'] = dis_lr

            for l in range(len(optimiser_g_x_y.param_groups)):
                optimiser_g_x_y.param_groups[l]['lr'] = gen_lr

            for l in range(len(optimiser_g_y_x.param_groups)):
                optimiser_g_y_x.param_groups[l]['lr'] = gen_lr

        for batch_num, data in enumerate(full_data):

            y, x = data[0].to(device), data[1].to(
                device)  # x is cartoon, y is human
            total_data += x.shape[0]

            optimiser_g_x_y.zero_grad()
            optimiser_g_y_x.zero_grad()
            optimiser_d_x.zero_grad()
            optimiser_d_y.zero_grad()

            fake_x = g_y_x(y)
            fake_y = g_x_y(x)

            if args.buffer_train:
                if batch_num == 0 and epoch == 0:
                    bufferX = fake_x[:10].clone()
                    bufferY = fake_y[:10].clone()

                perm = torch.randperm(bufferX.size(0))
                idx = perm[:sampling]
                samplesX[:sampling] = bufferX[idx]
                bufferX[idx] = fake_x[sampling:]
                samplesX[sampling:] = fake_x[:sampling]

                samplesY[:sampling] = bufferY[idx]
                bufferY[idx] = fake_y[sampling:]
                samplesY[sampling:] = fake_y[:sampling]

                d_x_loss = ((d_x(x) - 1) ** 2).mean() + \
                    (d_x(samplesX.detach())**2).mean()
                d_y_loss = ((d_y(y) - 1) ** 2).mean() + \
                    (d_y(samplesY.detach())**2).mean()

            else:
                d_x_loss = ((d_x(x) - 1) ** 2).mean() + \
                    (d_x(fake_x.detach())**2).mean()
                d_y_loss = ((d_y(y) - 1) ** 2).mean() + \
                    (d_y(fake_y.detach())**2).mean()

                d_x_loss /= 2
                d_y_loss /= 2

            d_x_loss.backward()
            d_y_loss.backward()

            optimiser_d_x.step()
            optimiser_d_y.step()

            loss_g_y_x = ((d_x(fake_x) - 1)**2).mean() + \
                cycle_loss(y, g_x_y(fake_x))
            loss_g_x_y = ((d_y(fake_y) - 1)**2).mean() + \
                cycle_loss(x, g_y_x(fake_y))

            loss_g_y_x.backward()
            loss_g_x_y.backward()
            optimiser_g_x_y.step()
            optimiser_g_y_x.step()
            total_d_loss += d_x_loss.item() + d_y_loss.item()
            total_g_x_y_loss += loss_g_x_y.item()
            total_g_y_x_loss += loss_g_y_x.item()

            del x
            del y

        avg_d_loss = total_d_loss / total_data
        avg_g_x_y_loss = total_g_x_y_loss / total_data
        avg_g_y_x_loss = total_g_y_x_loss / total_data
        print("Avg Discriminator Loss: {}".format(avg_d_loss))
        print("Avg Horse to Zebra Loss: {}".format(avg_g_x_y_loss))
        print("Avg Zebra to Horse Loss: {}".format(avg_g_y_x_loss))
