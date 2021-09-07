import torch
import torch.nn as nn
import torch.nn.functional as F

import models.modules.conv


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            models.modules.conv.SeparableConv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            models.modules.conv.SeparableConv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPwDSConv(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPPwDSConv, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            models.modules.conv.SeparableConv2d(len(self.convs) * out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
