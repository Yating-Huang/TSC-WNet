import torch.nn as nn
import torchvision.models as model
import torch

class vgg16(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(vgg16, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )

        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class mynet(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(mynet, self).__init__()

        self.features = nn.Sequential(
            #####  1  ######
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=1, padding=5),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            #####  2  ######
            nn.Conv2d(64, 128, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            #####  3  ######
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            #####  4  ######
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # # #####  5  ######
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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

class mynet1(nn.Module):
    def __init__(self,  input_channels=3,output_channels=3):
        super(mynet1, self).__init__()

        self.conv1 = DoubleConv(input_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.pool5 = nn.MaxPool2d(2)

        # self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.conv6 = DoubleConv(512, 256)
        # self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.conv7 = DoubleConv(256, 128)
        # self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.conv8 = DoubleConv(128, 64)
        # self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # self.conv9 = DoubleConv(64, 32)
        # self.conv10 = nn.Conv2d(32, out_ch, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_channels)
        )

    def forward(self, x):
        #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        #print(p4.shape)
        c5 = self.conv5(p4)
        p5 = self.pool5(c5)
        # up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6 = self.conv6(merge6)
        # up_7 = self.up7(c6)
        # merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)
        # up_8 = self.up8(c7)
        # merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        # up_9 = self.up9(c8)
        # merge9 = torch.cat([up_9, c1], dim=1)
        # c9 = self.conv9(merge9)
        # c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        x = self.avgpool(p5)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class mynet2(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(mynet2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding=3),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, dilation=3, padding=3),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=3),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1,  padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=3, padding=3),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=5, padding=5),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )

        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class mynet3(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(mynet3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=5, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=3),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=3, padding=3),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=5, padding=5),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )

        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
###步长保留最好结果###
class vgg16_1(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(vgg16_1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2,  dilation=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1,  padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )

        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
###卷积核###
class vgg16_2(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(vgg16_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )

        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
####最好结果0.8287292817679558 loss-3 10 0.1####
class vgg16_3(nn.Module):
    def __init__(self, input_channels=3,output_channels=3):
        super(vgg16_3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, dilation=3, stride=2, padding=1),  #
            nn.ReLU(inplace=True),
            # nn.Conv2d(input_channels, 64, kernel_size=3, dilation=3,  padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 32, kernel_size=3, dilation=2, stride=2, padding=1),  #
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # for p in self.parameters():
            #     p.requires_grad = False,

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )

        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(16*7*7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, output_channels)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        # y = x
        # print(x.shape)
        x = self.avgpool(x)
        # y = x


        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

class DoubleConv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, dilation=3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, dilation=2, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

    def forward(self, input):
        return self.conv(input)
class vgg16_4(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(vgg16_4, self).__init__()
        self.x1 = DoubleConv1(input_channels, 64)
        self.x2 = DoubleConv2(64, 32)

        self.x3 = DoubleConv3(32, 16)
        for p in self.parameters():
            p.requires_grad = False
        self.x4 = DoubleConv3(16, 16)
        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, output_channels)
        )

    def forward(self, x):
        # x = self.features(x)
        x1_1 = self.x1(x)

        x2_1 = self.x2(x1_1)

        x3_1 = self.x3(x2_1)

        x4_1 = self.x4(x3_1)
        x = self.avgpool(x4_1)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class vgg16_5(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(vgg16_5, self).__init__()
        self.x1 = DoubleConv1(input_channels, 64)
        self.x11= BasicBlock(64, 64)
        self.x2 = DoubleConv2(64, 32)
        self.x22 = BasicBlock(32, 32)

        self.x3 = DoubleConv3(32, 16)
        self.x33 = BasicBlock(16, 16)
        self.x4 = DoubleConv3(16, 16)
        self.x44 = BasicBlock(16, 16)

        self.x333 = Bottleneck(32, 32)
        self.x444 = Bottleneck(32, 32)

        self.Att1 = ChannelSELayer(64)
        self.Att2 = ChannelSELayer(32)
        self.Att3 = ChannelSELayer(16)
        self.Att4 = ChannelSELayer(16)
        # 原输入是224*224，5次max-pooling就是7*7，这层是为了模型的适用性
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, output_channels)
        )

    def forward(self, x):
        # x = self.features(x)
        x1_1 = self.x1(x)
        # x1_1 = self.x11(x1_1)
        x1_1 = self.Att1(x1_1)

        x2_1 = self.x2(x1_1)
        # x2_1 = self.x22(x2_1)
        # x2_1 = self.Att2(x2_1)

        x3_1 = self.x3(x2_1)
        # x3_1 = self.x33(x3_1)
        # x3_1 = self.x333(x2_1)
        # x4_1 = self.x444(x3_1)
        # x3_1 = self.Att3(x3_1)

        x4_1 = self.x4(x3_1)
        # x4_1 = self.x44(x3_1)
        # x4_1 = self.Att4(x4_1)

        x = self.avgpool(x4_1)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
