"""
Main file to call for train or test:
1. load all options for train or test
2. select gpu or cpu to train or test
"""
from __future__ import print_function  # help to use print() in python 2.x
import argparse
import os
import torch.backends.cudnn as cudnn

import train
import test
from network import *



parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='BlindDeblurGAN', help='BlindDeblurGAN')
parser.add_argument('--epoch', dest='epoch', type=int, default=15, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=360, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=1, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=50.0, help='weight on L1 term in objective')
parser.add_argument('--dark_channel_lambda', dest='dark_channel_lambda', type=float, default=1e4, help='weight on Dark Channel loss in objective')
parser.add_argument('--Perpetual_lambda', dest='Perpetual_lambda', type=float, default=0, help='weight on Perpetual term in objective')
parser.add_argument('--H', dest='H', default=720, type=int, help='Test size H')
parser.add_argument('--W', dest='W', default=1280, type=int, help='Test size W')
parser.add_argument('--sn', type=bool, default=True, help='using spectral norm')
parser.add_argument('--gpu', '-g', default=1, type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--seed', type=int, default=123, help='random seed to use, Default=123')

args = parser.parse_args()

print(args)

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if args.gpu > -1 or not torch.cuda.is_available():
    cuda_id = -1
else:
    cuda_id = args.gpu

print(args.gpu)
print(cuda_id)
exit()
cudnn.benchmark = True

torch.manual_seed(args.seed)


with torch.cuda.device(cuda_id):
    print("Current device is: {}".format(cuda_id))
    if args.phase == 'train':
        train.train(args)

    elif args.phase == 'test':
        if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir)
        test.test(args)
    else:
        raise Exception("Phase should be 'train" or 'test')

