import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        channel_list = list(args.channel_list)

        model = []
        for i in range(len(channel_list) - 1):
            input_channel = channel_list[i]
            output_channel = channel_list[i+1]
            model.append(nn.Conv2d(input_channel, output_channel,
                                   kernel_size=args.kernel, stride=args.stride, padding=(args.kernel - 1)//2))

            model.append(make_norm(output_channel, args.norm_type))
            model.append(make_activation(args.act_fn_gen))

        model += [ResidualBlock(channel_list[-1], args.norm_type)
                  for _ in range(args.num_res)]

        # _ = channel_list.pop(0)
        # channel_list.insert(0, args.image_dim[2])
        channel_list.reverse()
        for i in range(len(channel_list) - 1):
            input_channel = channel_list[i]
            output_channel = channel_list[i+1]
            if args.conv2T:
                model.append(nn.ConvTranspose2d(input_channel, output_channel,
                                                kernel_size=args.kernel, stride=args.stride, padding=(args.kernel - 1)//2))
            else:
                model.append(nn.Upsample(scale_factor=2,
                                         mode='bilinear', align_corners=False))
                model.append(nn.ReflectionPad2d(1))
                model.append(nn.Conv2d(input_channel, output_channel,
                                       kernel_size=3, stride=1, padding=0))

            model.append(make_norm(output_channel, args.norm_type))
            if i == len(channel_list) - 2:
                model.append(make_activation('sigmoid'))
            else:
                model.append(make_activation(args.act_fn_gen))

        self.go = nn.Sequential(*model)

    def forward(self, x):
        x = self.go(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        channel_list = list(args.channel_list)
        act_fn = args.act_fn_dis
        norm_type = args.norm_type

        # conv_layers = gen_downconv(channel_list, act_fn, norm_type)
        model = []
        for i in range(len(channel_list) - 1):
            input_channel = channel_list[i]
            output_channel = channel_list[i+1]
            model.append(nn.Conv2d(input_channel, output_channel,
                                   kernel_size=args.kernel, stride=args.stride, padding=(args.kernel - 1)//2))

            model.append(make_norm(output_channel, args.norm_type))
            model.append(make_activation(args.act_fn_gen))

        model.append(
            nn.Conv2d(channel_list[-1], 1, kernel_size=1, stride=1, padding=0))
        model.append(make_norm(1, args.norm_type))
        model.append(make_activation(args.act_fn_gen))

        self.conv = nn.Sequential(*model)

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
