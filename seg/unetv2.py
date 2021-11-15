import torch.nn as nn
import torch

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

###3层卷积
class Unetv1_1(nn.Module):
    def __init__(self, in_ch):
        super(Unetv1_1, self).__init__()

        self.conv1 = DoubleConv2_1(in_ch, 32)
        self.conv2 = DoubleConv2_2(32, 64)
        self.conv3 = DoubleConv2_3(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

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
        # c4 = self.conv4(p3)
        # p4 = self.pool4(c4)

        x = self.avgpool(p3)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
###5层卷积
class Unetv1_2(nn.Module):
    def __init__(self, in_ch):
        super(Unetv1_2, self).__init__()

        self.conv1 = DoubleConv2_1(in_ch, 32)
        # self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv2_2(32, 64)

        # self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv2_3(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv2_3(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv2_3(256, 512)
        self.pool5 = nn.MaxPool2d(2)

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
        #print(x.shape)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)
        p5 = self.pool5(c5)

        x = self.avgpool(p5)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
###无空洞
class DoubleConv3_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=3, stride=1, padding=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv3_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=2, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv3_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

####无步长
class Unetv1_3(nn.Module):
    def __init__(self, in_ch):
        super(Unetv1_3, self).__init__()

        self.conv1 = DoubleConv3_1(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv3_2(32, 64)

        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv3_3(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv3_3(128, 256)
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
        # print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        x = self.avgpool(p4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DoubleConv4_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv4_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv4_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv4_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv4_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv4_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
##无空洞
class Unetv1_4(nn.Module):
    def __init__(self, in_ch):
        super(Unetv1_4, self).__init__()

        self.conv1 = DoubleConv3_1(in_ch, 32)
        # self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv3_2(32, 64)

        # self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv3_3(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv3_3(128, 256)
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
        # print(x.shape)
        c1 = self.conv1(x)
        # p1 = self.pool1(c1)
        c2 = self.conv2(c1)
        # p2 = self.pool2(c2)
        c3 = self.conv3(c2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        x = self.avgpool(p4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x