
import torch
from .audio_net import Demucs, UNet
from .minkowski_net import SparseResNet18


class ModelBuilder():

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)

    def build_sound(self, arch='demucs', visual_feature_size=64, weights=''):

        if arch == 'unet':
            net_sound = UNet(visual_feature_size=visual_feature_size)
        elif arch == 'demucs':
            net_sound = Demucs(visual_feature_size=visual_feature_size)
        else:
            raise Exception('Architecture undefined!')

        net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    def build_vision(self, arch='sparseresnet18', visual_feature_size=64, in_dim=3, weights=''):

        if arch == 'sparseresnet18':
            net = SparseResNet18(visual_feature_size=visual_feature_size, in_channels=in_dim)
        else:
            raise Exception('Architecture undefined!')
        if len(weights) > 0:
            print('Loading weights for net_frame')
            net.load_state_dict(torch.load(weights))
        return net

    def build_criterion(self, arch):
        if arch == 'l1':
            loss = torch.nn.L1Loss()
        elif arch == 'l2':
            loss = torch.nn.MSELoss()
        else:
            raise Exception('Loss undefined!')
        return loss
