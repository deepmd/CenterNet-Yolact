###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNet
from .LadderNetv66_small import Decoder, LadderBlock


__all__ = ['LadderNet']


class LadderNet(BaseNet):
    def __init__(self, num_in_channels, num_out_channels, backbone='pelee'):
        super(LadderNet, self).__init__(backbone)

        self.head = LadderHead(num_in_channels, num_out_channels)

    def forward(self, x):
        features = self.base_forward(x)
        x = self.head(features)
        return x


class LadderHead(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super(LadderHead, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=num_in_channels[0],
        #                        out_channels=num_out_channels[0],
        #                        kernel_size=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=num_in_channels[1],
                               out_channels=num_out_channels[1],
                               kernel_size=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=num_in_channels[2],
                               out_channels=num_out_channels[2],
                               kernel_size=1, bias=False)

        # self.conv4 = nn.Conv2d(in_channels=num_in_channels[3],
        #                        out_channels=num_out_channels[3],
        #                        kernel_size=1, bias=False)

        # self.bn1 = nn.BatchNorm2d(num_out_channels[0])
        self.bn2 = nn.BatchNorm2d(num_out_channels[1])
        self.bn3 = nn.BatchNorm2d(num_out_channels[2])
        # self.bn4 = nn.BatchNorm2d(num_out_channels[3])

        self.decoder = Decoder(planes=[num_out_channels[0],
                                       num_out_channels[1],
                                       num_out_channels[2],
                                       num_out_channels[3]], layers=4)

        self.ladder = LadderBlock(planes=[num_out_channels[0],
                                          num_out_channels[1],
                                          num_out_channels[2],
                                          num_out_channels[3]], layers=4)

    def forward(self, x):
        x1, x2, x3, x4 = x

        # out1 = self.conv1(x1)
        # out1 = self.bn1(out1)
        # out1 = F.relu(out1)
        out1 = x1

        out2 = self.conv2(x2)
        out2 = self.bn2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(x3)
        out3 = self.bn3(out3)
        out3 = F.relu(out3)

        # out4 = self.conv4(x4)
        # out4 = self.bn4(out4)
        # out4 = F.relu(out4)
        out4 = x4

        out = self.decoder([out1, out2, out3, out4])
        out = self.ladder(out)

        return out[-1]

