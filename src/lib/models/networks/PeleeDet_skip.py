# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
from itertools import chain

import torch
import torch.nn as nn
from .peleenet import PeleeNet
from collections import OrderedDict

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class PeleeDet_Skip(nn.Module):
    def __init__(self, heads, head_conv):
        self.heads = heads
        self.deconv_with_bias = False

        super(PeleeDet_Skip, self).__init__()

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False)
        )

        self.bbon = PeleeNet()
        self.inplanes = 704

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            # fc = DCN(self.inplanes, planes,
            #          kernel_size=(3, 3), stride=1,
            #          padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv2d(self.inplanes, planes,
                           kernel_size=3, stride=1,
                           padding=1, dilation=1, bias=False)
            fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = x = self.bbon.features.stemblock(x)

        x = self.bbon.features.denseblock1(x)
        x = self.bbon.features.transition1(x)
        f2 = x = self.bbon.features.transition1_pool(x)

        x = self.bbon.features.denseblock2(x)
        x = self.bbon.features.transition2(x)
        f3 = x = self.bbon.features.transition2_pool(x)

        x = self.bbon.features.denseblock3(x)
        x = self.bbon.features.transition3(x)
        x = self.bbon.features.transition3_pool(x)

        x = self.bbon.features.denseblock4(x)
        f4 = x = self.bbon.features.transition4(x)

        f1 = self.skip_connection(f1)

        x = self.deconv_layers[0](f4)
        x = self.deconv_layers[1](x)
        x = self.deconv_layers[2](x)
        x = self.deconv_layers[3](x)
        x = self.deconv_layers[4](x)
        x = self.deconv_layers[5](x)
        x = x + f3

        x = self.deconv_layers[6](x)
        x = self.deconv_layers[7](x)
        x = self.deconv_layers[8](x)
        x = self.deconv_layers[9](x)
        x = self.deconv_layers[10](x)
        x = self.deconv_layers[11](x)
        x = x + f2

        x = self.deconv_layers[12](x)
        x = self.deconv_layers[13](x)
        x = self.deconv_layers[14](x)
        x = self.deconv_layers[15](x)
        x = self.deconv_layers[16](x)
        x = self.deconv_layers[17](x)
        x = x + f1

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers):
        if 1:
            checkpoint = torch.load('../models/peleenet_acc7208.pth.tar')
            new_state = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module'):
                    k = k[7:]
                new_state[k] = v

            print('=> loading pretrained model PeleeNet')
            self.load_state_dict(new_state, strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def freeze_params(self, freeze):
        deconv_params = self.deconv_layers.parameters()
        head_params = chain(*[self.__getattr__(h).parameters() for h in self.heads.keys()])
        if freeze == 'backbone':
            for p in self.parameters():
                p.requires_grad = False
            for p in chain(deconv_params, head_params):
                p.requires_grad = True
        elif freeze == 'all':
            for p in self.parameters():
                p.requires_grad = False
            for p in head_params:
                p.requires_grad = True


def get_pelee_det_skip(num_layers, heads, head_conv=256, freeze=None):
    model = PeleeDet_Skip(heads, head_conv=head_conv)
    model.init_weights(num_layers)
    model.freeze_params(freeze)
    return model
