import torch
import torch.nn as nn
import torch.nn.functional as F
from LocNet import LocNet
from model import InceptionStem, InceptionA, InceptionB, InceptionC, \
    ReductionA, ReductionB, Branch2_Dx, Branch2_Mx


def make_times(block, repeat):
    layer = [block] * repeat
    return nn.Sequential(*layer)


class DeepPrint(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.loc_net = LocNet(in_channels=in_channels)
        self.base_net = InceptionStem(in_channels=in_channels)
        self.branch1 = nn.Sequential(
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            ReductionA(in_channels=384, c3x3_out=320, c5x5_in=320, c5x5_out=320),
            InceptionB(in_channels=1024, kernel_size=7),
            InceptionB(in_channels=1024, kernel_size=7),
            InceptionB(in_channels=1024, kernel_size=7),
            InceptionB(in_channels=1024, kernel_size=7),
            InceptionB(in_channels=1024, kernel_size=7),
            InceptionB(in_channels=1024, kernel_size=7),
            InceptionB(in_channels=1024, kernel_size=7),
            ReductionB(in_channels=1024),
            InceptionC(in_channels=1536),
            InceptionC(in_channels=1536),
            InceptionC(in_channels=1536)
        )   # inception v4
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.branch1_fc = nn.Linear(in_features=1536, out_features=96)
        self.branch1_classifier = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(in_features=96, out_features=n_classes),
            nn.Softmax(dim=1)
        )

        self.branch2_ex = nn.Sequential(
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96),
            InceptionA(in_channels=384, c1x1_out=96, c3x3_in=64, c3x3_out=96, c5x5_in=64, c5x5_out=96)
        )  # 6x inceptionA
        self.branch2_dx = Branch2_Dx()
        self.branch2_mx = Branch2_Mx(n_classes=80)

    def branch1_logit(self, x):
        # adaptiveAvgPoolWidth = feature_map.shape[2]
        # x = F.avg_pool2d(feature_map, kernel_size=adaptiveAvgPoolWidth)
        # x = x.view(x.size(0), -1)     # [N, 1536]
        # x = nn.Dropout(p=0.8)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)             # [N, 1536]
        r1 = self.branch1_fc(x)               # [96], texture representation
        class1 = self.branch1_classifier(r1)
        return r1, class1

    def aligned(self, origin_img):
        b, c, h, w = origin_img.size()
        transforms = self.loc_net(origin_img)    # x, y, θ
        x, y, theta = transforms[:, 0], transforms[:, 1], transforms[:, 2]
        theta_rad = torch.deg2rad(theta)
        cos_theta, sin_theta = torch.cos(theta_rad), torch.sin(theta_rad)
        delta = 285 / 448
        affine_matrix = torch.stack([
            delta*cos_theta, -delta*sin_theta, x,
            delta*sin_theta, delta*cos_theta, y
        ], dim=-1).view(-1, 2, 3)
        affine_grid_points = F.affine_grid(affine_matrix, torch.Size((affine_matrix.size(0), c, h, w)))
        aligned_img = F.grid_sample(origin_img, affine_grid_points)
        return aligned_img

    def forward(self, origin_img):
        aligned_img = self.aligned(origin_img)
        F_map = self.base_net(aligned_img)   # Inception v4 stem
        x = self.branch1(F_map)  # complete inception v4  完整的inception v4提取的特征图
        r1, class1 = self.branch1_logit(x)
        M_map = self.branch2_ex(F_map)
        r2, class2 = self.branch2_mx(M_map)
        minutiae_map = self.branch2_dx(M_map)
        return r1, r2, class1, class2, minutiae_map


img = torch.randn(1, 1, 299, 299)
deepPrint = DeepPrint(in_channels=1, n_classes=80)
outputs = deepPrint(img)


