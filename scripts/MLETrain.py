import torch
import torch.nn as nn
import torch.optim as optim
from GeneratorAndDiscriminator import Generator, Discriminator
from dataloader import get_data_loader
from tqdm import tqdm, trange
from datetime import datetime


def train(args, device):
    full_data = get_data_loader(args)

    time = datetime.now()

    d = Discriminator(args).to(device)
    print("--Discriminator architecture--")
    print(d)
    g = Generator(args).to(device)
    print("--Generator architecture--")
    print(g)

    optimiser_d = optim.Adam(d.parameters(), lr=args.dis_learning_rate)
    optimiser_g = optim.Adam(g.parameters(), lr=args.gen_learning_rate)

    klDiv = nn.KLDivLoss()

    if args.load_models:
        print("Loading Models...")
        models = torch.load(args.model_path)
        d.load_state_dict(models['d'])

        g.load_state_dict(models['g'])
        optimiser_d.load_state_dict(models['optimiser_d'])

        optimiser_g.load_state_dict(models['optimiser_g'])
        print("Successfully Loaded Models...")

    d.train()
    g.train()

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
    for epoch in trange(args.starting_epoch, args.num_epochs+1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_data = 0

        if args.decay and epoch > 100:
            dis_lr = args.dis_learning_rate - \
                ((args.dis_learning_rate / 100) * (200 - epoch))
            gen_lr = args.gen_learning_rate - \
                ((args.gen_learning_rate / 100) * (200 - epoch))

            for l in range(len(optimiser_d.param_groups)):
                optimiser_d.param_groups[l]['lr'] = dis_lr

            for l in range(len(optimiser_g.param_groups)):
                optimiser_g.param_groups[l]['lr'] = gen_lr

        for batch_num, data in enumerate(full_data):

            y, x = data[0].to(device), data[1].to(
                device)
            total_data += x.shape[0]

            optimiser_g.zero_grad()
            optimiser_d.zero_grad()

            fake_y = g(x)

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

                d_loss = ((d(x) - 1) ** 2).mean() + \
                    (d(samplesX.detach())**2).mean()

            else:
                d_loss = ((d(x) - 1) ** 2).mean() + \
                    (d(fake_y.detach())**2).mean()

                d_loss /= 2

            d_loss.backward()

            optimiser_d.step()

            loss_g = ((d(fake_y) - 1)**2).mean() + \
                klDiv(y, fake_y)

            loss_g.backward()
            optimiser_g.step()
            total_d_loss += d_loss.item()
            total_g_loss += loss_g.item()

            del x
            del y

        if epoch % args.save_epoch == 0 or epoch == args.num_epochs:
            torch.save({"d": d.state_dict(), "g": g.state_dict(), "optimiser_d": optimiser_d.state_dict(
            ), "optimiser_g": optimiser_g.state_dict()}, args.save_path + 'model{}{}.pt'.format(epoch, time))

        avg_d_loss = total_d_loss / total_data
        avg_g_loss = total_g_loss / total_data

        with open(args.save_path + f'discrimLoss{time}.txt', 'a') as f:
            f.write("Avg Discriminator Loss: {}".format(avg_d_loss))

        with open(args.save_path + f'genx_yLoss{time}.txt', 'a') as f:
            f.write("Avg X to Y Loss: {}".format(avg_g_loss))

    with open(args.save_path + f'model{time}.txt', 'w') as f:
        f.write(str(args))
