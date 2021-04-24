import argparse
from attrdict import AttrDict
import train
import test
import torch
import latentDimensionTrain
import latentDimensionTest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f"Device: {device}")


run_args = AttrDict()
args_dict = {
    'dis_learning_rate': 0.0002,
    'gen_learning_rate': 0.0002,
    'batch_size': 2048,
    'num_epochs': 5,
    'starting_epoch': 1,
    'channel_list': [1, 32, 64, 128, 256],
    'image_dim': (28, 28, 1),
    'kernel': 4,
    'stride': 2,
    'x': "../data/trainMNIST/",
    'y': "../data/testZebra/",
    'save_path': '../models/',
    'act_fn_gen': 'relu',
    'act_fn_dis': 'lrelu',
    'norm_type': 'instance',
    'num_res': 3,
    'lambda_cycle': 10,
    'gray': False,
    'conv2T': False,
    'buffer_train': False,
    'decay': True,
    'train': False,
    'load_models': False,
    'model_path': '../models/model52021-04-24 13:04:26.582314.pt',
    'save_epoch': 10,
    'LatentDims': True
}

run_args.update(args_dict)
if run_args.train:
    if run_args.LatentDims:
        latentDimensionTrain.train(run_args, device)
    else:
        train.train(run_args, device)
else:
    if run_args.LatentDims:
        latentDimensionTest.test(run_args, device)
    else:
        test.test(run_args, device)
