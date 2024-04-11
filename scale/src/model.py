import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, BN: bool = True,
                 Act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.bias = False if BN else True
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=self.bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = Act

    def forward(self, x):
        x = self.conv(x)
        if not self.bias:
            x = self.bn(x)
        return self.relu(x)


class InceptionA(nn.Module):
    def __init__(self, in_channels, c1x1_out, c3x3_in, c3x3_out, c5x5_in, c5x5_out):
        super().__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=c1x1_out, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c3x3_in, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=c3x3_in, out_channels=c3x3_out, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c5x5_in, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=c5x5_in, out_channels=c5x5_out, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=c5x5_out, out_channels=c5x5_out, kernel_size=3, stride=1, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size: int = 7):
        super().__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=224, kernel_size=(1, kernel_size), stride=1,
                      padding=(0, kernel_size // 2)),
            ConvBlock(in_channels=224, out_channels=256, kernel_size=(kernel_size, 1), stride=1,
                      padding=(kernel_size // 2, 0))
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=192, kernel_size=(1, kernel_size), stride=1,
                      padding=(0, kernel_size // 2)),
            ConvBlock(in_channels=192, out_channels=224, kernel_size=(kernel_size, 1), stride=1,
                      padding=(kernel_size // 2, 0)),
            ConvBlock(in_channels=224, out_channels=224, kernel_size=(1, kernel_size), stride=1,
                      padding=(0, kernel_size // 2)),
            ConvBlock(in_channels=224, out_channels=256, kernel_size=(kernel_size, 1), stride=1,
                      padding=(kernel_size // 2, 0))
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.branch2 = ConvBlock(in_channels=in_channels, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.branch2_1 = ConvBlock(in_channels=384, out_channels=256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_2 = ConvBlock(in_channels=384, out_channels=256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=384, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=384, out_channels=448, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            ConvBlock(in_channels=448, out_channels=512, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )
        self.branch3_1 = ConvBlock(in_channels=512, out_channels=256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch3_2 = ConvBlock(in_channels=512, out_channels=256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x21 = self.branch2_1(x2)
        x22 = self.branch2_2(x2)
        x3 = self.branch3(x)
        x31 = self.branch3_1(x3)
        x32 = self.branch3_2(x3)
        x4 = self.branch4(x)
        x = torch.cat([x1, x21, x22, x31, x32, x4], dim=1)
        return x


class ReductionA(nn.Module):
    def __init__(self, in_channels, c3x3_out, c5x5_in, c5x5_out):
        super().__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=c3x3_out, kernel_size=3, stride=2, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c5x5_in, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=c5x5_in, out_channels=c5x5_out, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=c5x5_out, out_channels=c5x5_out, kernel_size=3, stride=2, padding=0),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=192, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=0)
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=256, out_channels=320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=320, out_channels=320, kernel_size=3, stride=2, padding=0),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class InceptionStem(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4_branch1 = ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.conv4_branch2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv5_branch1 = nn.Sequential(
            ConvBlock(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0),
        )
        self.conv5_branch2 = nn.Sequential(
            ConvBlock(in_channels=160, out_channels=64, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=0),
        )
        self.conv6_branch1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv6_branch2 = ConvBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        print('# Conv1 output shape:', x.shape)
        x = self.conv2(x)
        print('# Conv2 output shape:', x.shape)
        x = self.conv3(x)
        print('# Conv3 output shape:', x.shape)
        x = torch.cat([self.conv4_branch1(x), self.conv4_branch2(x)], dim=1)
        print('# Conv4 output shape:', x.shape)
        x = torch.cat([self.conv5_branch1(x), self.conv5_branch2(x)], dim=1)
        print('# Conv5 output shape:', x.shape)
        x = torch.cat([self.conv6_branch1(x), self.conv6_branch2(x)], dim=1)
        print('# Conv6 output shape:', x.shape)
        return x


class InceptionV4(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        # self.stem: begin
        self.stem = InceptionStem(in_channels)
        # self.stem: end
        self.inceptionA = self.make_times(
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            repeat=4)
        self.reductionA = ReductionA(in_channels=384, c3x3_out=320, c5x5_in=320, c5x5_out=320)
        self.inceptionB = self.make_times(InceptionB(in_channels=1024, kernel_size=7), repeat=7)
        self.reductionB = ReductionB(in_channels=1024)
        self.inceptionC = self.make_times(InceptionC(in_channels=1536), repeat=3)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.8),
            nn.Linear(in_features=1536, out_features=96),
            nn.Linear(in_features=96, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def make_times(self, block, repeat):
        layer = [block] * repeat
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionA(x)
        print('# InceptionA output shape:', x.shape)
        x = self.reductionA(x)
        print('# ReductionA output shape:', x.shape)
        x = self.inceptionB(x)
        print('# InceptionB output shape:', x.shape)
        x = self.reductionB(x)
        print('# ReductionB output shape:', x.shape)
        x = self.inceptionC(x)
        print('# InceptionC output shape:', x.shape)
        x = self.globalpool(x)
        print('# GlobalPool output shape:', x.shape)
        x = self.classifier(x)
        print('# Classifier output shape:', x.shape)
        return x


class Branch2_Dx(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_conv1 = nn.ConvTranspose2d(in_channels=384, out_channels=128, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=0)
        self.de_conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.de_conv1(x)  # 70 * 70 * 128
        x = self.conv2(x)  # 64 * 64 * 128
        x = self.de_conv3(x)  # 128 * 128 * 32
        x = self.conv4(x)  # 128 * 128 * 6  Minutiae Map
        return x


class Branch2_Mx(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=384, out_channels=768, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=768, out_channels=896, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=896, out_channels=1024, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.branch2_fc = nn.Linear(in_features=1024, out_features=96)
        self.branch2_classifier = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(in_features=96, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)   # 35 * 35 * 768
        x = self.conv2(x)   # 18 * 18 * 768
        x = self.conv3(x)   # 18 * 18 * 896
        x = self.conv4(x)   # 9 * 9 * 1024
        x = self.conv5(x)   # 9 * 9 * 1024
        x = self.pool1(x)   # 1 * 1 * 1024
        x = x.view(x.size(0), -1)
        r2 = self.branch2_fc(x)   # minutiae representation
        class2 = self.branch2_classifier(r2)
        return r2, class2


inputs = torch.randn(4, 3, 299, 299)
cnn = InceptionV4(in_channels=3, n_classes=96)
outputs = cnn(inputs)

# inputs = torch.randn(4, 384, 35, 35)
# cnn = Branch2_Mx(n_classes=80)
# outputs = cnn(inputs)
