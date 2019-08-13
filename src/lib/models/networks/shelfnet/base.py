###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn


# from .mbv2 import MobileNet2
from .peleenet import PeleeNet
from .resnet import resnet18


__all__ = ['BaseNet']


class BaseNet(nn.Module):
    def __init__(self, backbone):
        super(BaseNet, self).__init__()
        self.bbon = backbone

        if self.bbon == 'pelee':
            self.pretrained = PeleeNet().features
        elif self.bbon == 'res18':
            self.pretrained = resnet18(pretrained=True, dilated=False, norm_layer=nn.BatchNorm2d)

    def base_forward(self, x):
        if self.bbon == 'pelee':
            c1 = x = self.pretrained.stemblock(x)

            x = self.pretrained.denseblock1(x)
            x = self.pretrained.transition1(x)
            c2 = x = self.pretrained.transition1_pool(x)

            x = self.pretrained.denseblock2(x)
            x = self.pretrained.transition2(x)
            c3 = x = self.pretrained.transition2_pool(x)

            x = self.pretrained.denseblock3(x)
            x = self.pretrained.transition3(x)
            x = self.pretrained.transition3_pool(x)

            x = self.pretrained.denseblock4(x)
            c4 = x = self.pretrained.transition4(x)

        elif self.bbon == 'res18':
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)

        return c1, c2, c3, c4
