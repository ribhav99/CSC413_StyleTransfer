import torch
import torch.nn as nn
import torch.optim as optim
from GeneratorAndDiscriminator import Generator, Discriminator
from dataloader import get_data_loader
from tqdm import tqdm, trange
from datetime import datetime
from torch.autograd import Variable


def train(args, device):
    full_data = get_data_loader(args)

    time = datetime.now()

    adversarial_loss = nn.BCELoss()

    d = Discriminator(args).to(device)
    print("--Discriminator architecture--")
    print(d)
    g = Generator(args).to(device)
    print("--Generator architecture--")
    print(g)

    optimiser_d = optim.Adam(d.parameters(), lr=args.dis_learning_rate)
    optimiser_g = optim.Adam(g.parameters(), lr=args.gen_learning_rate)

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

            y = ((data + 1)/2.0).to(device)
            x = torch.normal(5, 10, size=(
                data.shape[0], args.channel_list[0], args.image_dim[0], args.image_dim[1])).to(device)
            total_data += y.shape[0]

            valid = Variable(torch.Tensor(data.shape[0], 1).fill_(
                1.0), requires_grad=False).to(device)
            fake = Variable(torch.Tensor(data.shape[0], 1).fill_(
                0.0), requires_grad=False).to(device)

            optimiser_d.zero_grad()
            optimiser_g.zero_grad()

            fake_y = g(x)

            real_loss = adversarial_loss(d(y), valid)
            fake_loss = adversarial_loss(d(fake_y.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimiser_d.step()

            loss_g = adversarial_loss(d(fake_y), valid)

            loss_g.backward()
            optimiser_g.step()
            total_d_loss += d_loss.item()
            total_g_loss += loss_g.item()

            del x
            del y
            del valid
            del fake
            del fake_y

        if epoch % args.save_epoch == 0 or epoch == args.num_epochs:
            torch.save({"d": d.state_dict(), "g": g.state_dict(), "optimiser_d": optimiser_d.state_dict(
            ), "optimiser_g": optimiser_g.state_dict()}, args.save_path + 'model{}{}.pt'.format(epoch, time))

        avg_d_loss = total_d_loss / total_data
        avg_g_loss = total_g_loss / total_data

        with open(args.save_path + f'discrimLoss{time}.txt', 'a') as f:
            f.write("Avg Discriminator Loss: {}\n".format(avg_d_loss))

        with open(args.save_path + f'genx_yLoss{time}.txt', 'a') as f:
            f.write("Avg X to Y Loss: {}\n".format(avg_g_loss))

    with open(args.save_path + f'model{time}.txt', 'w') as f:
        f.write(str(args))
