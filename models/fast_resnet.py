import torch
from torch import nn


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True, bias=True):
    """
    Definition of a conv+BN block with residual connection
    :param channels_in: (int) input channel dimension
    :param channels_out: (int) output channel dimension
    :param kernel_size: (int) size of convolutional kernel
    :param stride: (int) stride of convolutional kernel
    :param padding: (int) padding of conv
    :param groups: (int) groups of conv
    :param bn: (int) whether to apply batchnorm
    :param activation: (int) whether to add ReLU activation
    :param bias: (int) whether to add bias for convolution
    :return: module
    """
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class FastResNet(nn.Module):
    
    def __init__(self, num_class=10, sphere_projected=False, last_relu=True, bias=False, dropout=None):
        """
        Fast version of resnet adapted from the blog: https://myrtle.ai/learn/how-to-train-your-resnet/
        :param num_class: (int) number of classes
        :param sphere_projected: (bool) whether to project to unit sphere
        :param last_relu: (bool) whether apply ReLU at the end
        :param bias: (bool) add bias to fully connected or not
        :param dropout: (None or float) add dropout before FC layer
        """
        super(FastResNet, self).__init__()
        self.encoder = torch.nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            # torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(
                conv_bn(128, 128),
                conv_bn(128, 128),
            )),

            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(                             # try from here
                conv_bn(256, 256),
                conv_bn(256, 256),
            )),

            conv_bn(256, 128, kernel_size=3, stride=1, padding=0, activation=last_relu),

            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten())
        self.dropout = None if dropout is None else nn.Dropout(p=dropout)
        self.fc = torch.nn.Linear(128, num_class, bias=bias)
        self.sphere = sphere_projected
    
    def forward(self, x):
        x = self.encoder(x)
        if hasattr(self, 'dropout') and self.dropout is not None:
            x = self.dropout(x)
        if self.sphere:
            norms = torch.norm(x, dim=1, keepdim=True)
            x = x / norms
        x = self.fc(x)
        return x
