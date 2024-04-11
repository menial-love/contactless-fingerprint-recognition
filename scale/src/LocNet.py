import torch
import torch.nn as nn
import torch.nn.functional as F


class LocNet(nn.Module):
    def __init__(self, in_channels):
        super(LocNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8*8*64, 64)
        # Fully connected 2 (Output layer assuming 3 classes for x, y, θ)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, origin_img):
        resized_img = F.interpolate(origin_img, size=(128, 128), mode='bilinear', align_corners=False)
        x = self.conv1(resized_img)
        x = self.pool1(F.relu(x))
        x = self.conv2(x)
        x = self.pool2(F.relu(x))
        x = self.conv3(x)
        x = self.pool3(F.relu(x))
        x = self.conv4(x)
        x = self.pool4(F.relu(x))
        x = x.view(-1, 8*8*64)
        x = F.relu(self.fc1(x))
        transforms = self.fc2(x)
        return transforms


# net = LocNet()
# inputs = torch.randn(12, 1, 128, 128)
# transforms = net(inputs)
# x = transforms[:, 0]
# y = transforms[:, 1]
# theta = transforms[:, 2]
# theta_rad = torch.deg2rad(theta)
# cos_theta = torch.cos(theta_rad)
# sin_theta = torch.sin(theta_rad)
# delta = 285 / 448
# # affine_matrices = torch.eye(3).repeat(12, 1, 1)
# # affine_matrices[:, 0, 0] = delta * cos_theta
# # affine_matrices[:, 0, 1] = -delta * sin_theta
# # affine_matrices[:, 0, 2] = transforms[:, 0]
# # affine_matrices[:, 1, 0] = delta * sin_theta
# # affine_matrices[:, 1, 1] = delta * cos_theta
# # affine_matrices[:, 1, 2] = transforms[:, 1]
# affine_matrices = torch.stack([
#     delta*cos_theta, -delta*sin_theta, x,
#     delta*sin_theta, delta*cos_theta, y
# ], dim=-1).view(-1, 2, 3)
# affine_grid_points = F.affine_grid(affine_matrices, torch.Size((affine_matrices.size(0), 1, 128, 128)))
# rois = F.grid_sample(inputs, affine_grid_points)
# print(rois)
