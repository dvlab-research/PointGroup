'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
from yaml import Loader
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    ### ddp
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('-nr', '--node_rank', type=int, default=0, help='ranking within the nodes')

    args = parser.parse_args()
    assert args.config is not None
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    setattr(args, 'exp_path', os.path.join('exp', args.dataset, args.model_name, args.config.split('/')[-1][:-5]))

    return args



