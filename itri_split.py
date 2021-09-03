import os
import argparse
import random
import glob
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def split_itri(data_dir, val=0.2, test=None, random_seed=69):
    random.seed(random_seed)
    ann_dir = 'TXT'
    all_files = glob.glob(os.path.join(data_dir, ann_dir) + '/*.txt')
    all_files = [os.path.basename(f)[:-4] for f in all_files]
    val_files = random.sample(all_files, int(val*len(all_files)))
    train_files = [f for f in all_files if f not in val_files]
    with open(os.path.join(data_dir, 'SPLIT', 'train.txt'), 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)
    with open(os.path.join(data_dir, 'SPLIT', 'val.txt'), 'w') as f:
        for item in val_files:
            f.write("%s\n" % item)


def parse_args():
    parser = argparse.ArgumentParser(description='Split ITRI dataset')
    parser.add_argument('--datadir', help="data dir", default="data/itri", type=str)
    parser.add_argument('--verbose', help="verbose", action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    split_itri(args.datadir)