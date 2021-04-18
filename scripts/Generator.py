import torch
import torch.nn as nn
from init_helper import gen_upconv, get_up_conv_block


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        model = []
        for i in range(len(args.channel_list) - 1):
            input_channel = args.channel_list[i]
            output_channel = args.channel_list[i+1]
            model.append(nn.Conv2d(input_channel, output_channel,
                                   kernel_size=args.kernel, stride=args.stride, padding=(args.kernel - 1)//2))

            model.append(make_norm(output_channel, args.norm_type))
            model.append(make_activation(args.act_fn_gen))

        model += [ResidualBlock(args.channel_list[-1], args.norm_type)
                  for _ in range(args.num_res)]

        model += gen_upconv(args.channel_list, args.act_fn_gen, args.norm_type,
                            dropout=False, conv2T=args.Conv2T)
        self.go = nn.Sequential(*model)

    def forward(self, x):
        x = self.go(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, norm_type, dropout=False):
        super(ResidualBlock, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1), make_norm(in_channel, norm_type), make_activation('relu'))
        self.second = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1), make_norm(in_channel, norm_type), make_activation('relu'))

    def forward(self, x):
        residual = x
        output = self.first(x)
        output = self.second(output)
        final = residual + output
        return final


def make_norm(channel, norm_type):
    if norm_type == 'instance':
        return nn.InstanceNorm2d(channel)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(channel)
    else:
        return nn.Identity()


def make_activation(activation):
    if activation == 'lrelu':
        return nn.LeakyReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU()
