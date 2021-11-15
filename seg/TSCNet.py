import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models

from seg import deform_conv
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv2_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=3, stride=2, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv2_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=2, stride=2, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv2_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class TSCNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TSCNet, self).__init__()

        self.conv1 = DoubleConv2_1(in_ch, 32)
        self.conv2 = DoubleConv2_2(32, 64)
        # for p in self.parameters():
        #     p.requires_grad = False
        # self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv2_3(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv2_3(128, 256)
        # for p in self.parameters():
        #     p.requires_grad = False

        self.pool4 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        x = self.avgpool(p4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TSCUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TSCUNet, self).__init__()

        self.conv1 = DoubleConv2_1(in_ch, 32)
        # self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv2_2(32, 64)
        # self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv2_3(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv2_3(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        # self.conv5 = DoubleConv(256, 512)
        # self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(96, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(32, 16)
        self.conv10 = nn.Conv2d(16, out_ch, 1)


    def forward(self, x):
        #print(x.shape)
        c1 = self.conv1(x)
        # p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(c1)
        # p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)

        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        # merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(up_9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

class TSCWNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TSCWNet, self).__init__()

        self.conv1 = DoubleConv2_1(in_ch, 32)
        # self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv2_2(32, 64)
        # for p in self.parameters():
        #     p.requires_grad = False
        # self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv2_3(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv2_3(128, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(96, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(32, 16)
        self.conv10 = nn.Conv2d(16, 1, 1)
        #
        # for p in self.parameters():
        #     p.requires_grad = False

        self.conv1_1 = DoubleConv2_1(4, 32)
        self.conv2_2 = DoubleConv2_2(32, 64)
        self.conv3_3 = DoubleConv2_3(64, 128)
        self.pool3_3 = nn.MaxPool2d(2)
        self.conv4_4 = DoubleConv2_3(256, 512)

        self.pool4 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        original=x

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        # p4 = self.pool4(c4)

        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        # merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(up_9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)

        merge1 = torch.cat([original, out], dim=1)
        c1_1 = self.conv1_1(merge1)
        c2_2 = self.conv2_2(c1_1)
        c3_3 = self.conv3_3(c2_2)
        c3_3 = self.pool3_3(c3_3)
        merge4 = torch.cat([c3_3, p3], dim=1)
        c4_4 = self.conv4_4(merge4)
        c4_4 = self.pool3_3(c4_4)

        x = self.avgpool(c4_4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
