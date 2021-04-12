import torch.nn as nn
from torch.nn import functional as F


class DRModel(nn.Module):
    """
    Model that was used as fifth place solution
    for diabetic retinopathy dataset. Used from
    https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/
    """

    def __init__(self, num_channel=3, dropout_rate=0.0, num_classes=2):
        """
        Model for diabetic retinopathy dataset
        :param num_channel: (int) number of channel of input images
        :param dropout_rate: (0.0 < float < 1.0) dropout rate
        :param num_classes: (int) number of classes to predict
        """
        super(DRModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 32, kernel_size=7, stride=2,
                               padding=3, padding_mode='same', bias=True)
        self.leakyrelu = nn.LeakyReLU(0.5, inplace=True)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=3)

        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, padding_mode='same', bias=True)
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                padding=1, padding_mode='same', bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(256)
        self.bn11 = nn.BatchNorm2d(256)
        self.bn12 = nn.BatchNorm2d(256)
        self.bn13 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leakyrelu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.leakyrelu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.leakyrelu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.leakyrelu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.leakyrelu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.leakyrelu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.leakyrelu(x)

        x = self.maxpool2d(x)
        x = x.view(batch_size, -1)

        x = self.dropout(x)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.fc2(x)

        return x
