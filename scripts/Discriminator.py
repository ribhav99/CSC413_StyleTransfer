import torch
import torch.nn as nn
from init_helper import gen_downconv, get_down_conv_block


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        channel_list = [3, 16, 32, 64, 128, 256]
        if args.gray:
            channel_list[0] = 1
        act_fn = args.act_fn_dis
        norm_type = args.norm_type
        conv_layers = gen_downconv(channel_list, act_fn, norm_type)
        conv_layers += get_down_conv_block(256,
                                           1, 1, act_fn, norm_type, False, 1)
        self.conv = nn.Sequential(*conv_layers)

        fake_data = torch.ones(
            args.batch_size, args.image_dim[2], args.image_dim[0], args.image_dim[1])
        parsed_fake = self.conv(fake_data)
        parsed_fake = parsed_fake.view(args.batch_size, -1)

        self.linear = nn.Sequential(
            nn.Linear(parsed_fake.shape[1], 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.linear(x)
        return x
