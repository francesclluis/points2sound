
import argparse


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--id', default='',
                            help="a name for identifying the model")
        parser.add_argument('--num_mix', default=2, type=int,
                            help="max sources in the mixture")
        parser.add_argument('--arch_sound', default='demucs',
                            help="architecture of net_sound")
        parser.add_argument('--arch_vision', default='sparseresnet18',
                            help="architecture of net_vision")
        parser.add_argument('--weights_sound', default='',
                            help="weights to finetune net_sound")
        parser.add_argument('--weights_vision', default='',
                            help="weights to finetune net_vision")
        parser.add_argument('--visual_feature_size', default=32, type=int,
                            help='extracted visual feature size')
        parser.add_argument('--loss', default='l1',
                            help="loss function to use")
        parser.add_argument('--num_gpus', default=1, type=int,
                            help='number of gpus to use')
        parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
        parser.add_argument('--workers', default=32, type=int,
                            help='number of data loading workers')
        parser.add_argument('--num_val', default=-1, type=int,
                            help='number of samples to evalutate')
        parser.add_argument('--num_vis', default=40, type=int,
                            help='number of samples to evalutate')
        parser.add_argument('--audLen', default=131072, type=int,
                            help='sound length in samples')
        parser.add_argument('--audRate', default=44100, type=int,
                            help='sound sampling rate')
        parser.add_argument('--voxel_size', default=0.02, type=float,
                            help='size of volume element')
        parser.add_argument('--rgbs_feature', default=0, type=int,
                            help='use rgbs in point feature vector')
        parser.add_argument('--seed', default=1234, type=int,
                            help='manual seed')
        parser.add_argument('--ckpt', default='./ckpt',
                            help='folder to output checkpoints')
        parser.add_argument('--disp_iter', type=int, default=20,
                            help='frequency to display')
        parser.add_argument('--eval_epoch', type=int, default=1,
                            help='frequency to evaluate')

        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser

        parser.add_argument('--mode', default='train',
                            help="train/eval")
        parser.add_argument('--list_train',
                            default='data/train.csv')
        parser.add_argument('--list_val',
                            default='data/val.csv')
        parser.add_argument('--dup_trainset', default=100, type=int,
                            help='duplicate so that one epoch has more iters')
        parser.add_argument('--num_epoch', default=100, type=int,
                            help='epochs to train for')
        parser.add_argument('--lr_vision', default=1e-4, type=float, help='LR')
        parser.add_argument('--lr_sound', default=1e-4, type=float, help='LR')
        parser.add_argument('--beta1', default=0.9, type=float,
                            help='momentum for sgd, beta1 for adam')
        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def parse_train_arguments(self):
        self.add_train_arguments()
        args = self.parser.parse_args()
        self.print_arguments(args)
        return args
