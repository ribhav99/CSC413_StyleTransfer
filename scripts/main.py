import argparse
from attrdict import AttrDict
import train
import test
import torch
import MLETrain
import MLETest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(f"Device: {device}")


run_args = AttrDict()
args_dict = {
    'dis_learning_rate': 0.0002,
    'gen_learning_rate': 0.0002,
    'batch_size': 16,
    'num_epochs': 200,
    'starting_epoch': 1,
    'channel_list': [3, 32, 64, 128, 256],  # if gray, channel_list[0] = 1
    'image_dim': (256, 256, 3),  # 1 if gray, 3 if coloured
    'kernel': 4,
    'stride': 2,
    'x': "../data/testHorse/",
    'y': "../data/testZebra/",
    'save_path': '../models/',
    'act_fn_gen': 'relu',
    'act_fn_dis': 'lrelu',
    'norm_type': 'instance',
    'num_res': 9,
    'lambda_cycle': 10,
    'gray': False,
    'conv2T': False,
    'buffer_train': False,
    'decay': True,
    'train': False,
    'load_models': False,
    'model_path': '../models/model2002021-04-24 10:26:40.453567.pt',
    'save_epoch': 10,
    'MLE': True
}

run_args.update(args_dict)
if run_args.train:
    if run_args.MLE:
        MLETrain.train(run_args, device)
    else:
        train.train(run_args, device)
else:
    if run_args.MLE:
        MLETest.test(run_args, device)
    else:
        test.test(run_args, device)
