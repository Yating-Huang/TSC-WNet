import torch
import torch.nn as nn
from .utils import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # nn.BatchNorm2d(64),  ##add
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512), #4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # y = x

        x = self.avgpool(x)
        # y = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # print(y.shape)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

class AlexNet_1(nn.Module):

    def __init__(self, num_classes=3):
        super(AlexNet_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # nn.LocalResponseNorm(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.lr = nn.ReLU(inplace=True)
        self.x3_1 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.x3_2 = nn.Conv2d(576, 384, kernel_size=3, padding=1)

        self.x4_1 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.x4_2 = nn.Conv2d(640, 256, kernel_size=3, padding=1)

        self.x5_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.x5_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.mp5 = nn.MaxPool2d(kernel_size=3, stride=2)


        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512), #4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        #  y = x
        residual =x  #192
        x = self.x3_1(x) #384
        x = self.lr(x)
        x = self.x3_2(torch.cat([residual, x], 1)) # 576>384
        x = self.lr(x)

        residual = x  # 384
        x = self.x4_1(x)  # 256
        x = self.lr(x)
        x = self.x4_2(torch.cat([residual, x], 1)) # 640---> 256
        x = self.lr(x)

        residual = x  # 256
        x = self.x5_1(x)  # 256
        x = self.lr(x)
        x = self.x5_2(torch.cat([residual, x], 1))  # 512>256
        x = self.lr(x)

        x = self.mp5(x)

        x = self.avgpool(x)
        # y = x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # print(y.shape)
        return x
