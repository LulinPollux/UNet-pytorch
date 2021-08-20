import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import models.backbone
import models.modules
import utils


class Proposed(nn.Module):
    def __init__(self, backbone_type: str, output_stride: int, num_classes: int):
        super(Proposed, self).__init__()
        assert output_stride in (8, 16)
        self.low_level_feature = []

        # Backbone
        self.backbone = models.backbone.efficientnet.efficientnetv2(backbone_type, output_stride, pretrained=True)
        self.backbone.stages[0].register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))
        self.backbone.stages[1].register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))
        if output_stride == 16:
            self.backbone.stages[2].register_forward_hook(utils.hooks.get_feature_maps(self.low_level_feature))

        # ASPP
        if output_stride == 16:
            atrous_rates = (6, 12, 18)
        elif output_stride == 8:
            atrous_rates = (12, 24, 36)
        else:
            raise NotImplementedError('Wrong output_stride.')
        if backbone_type == 'small':
            self.aspp = models.modules.aspp.ASPPwDSConv(256, atrous_rates, 256)
        elif backbone_type == 'medium':
            self.aspp = models.modules.aspp.ASPPwDSConv(512, atrous_rates, 256)
        elif backbone_type == 'large':
            self.aspp = models.modules.aspp.ASPPwDSConv(640, atrous_rates, 256)
        else:
            raise NotImplementedError('Wrong backbone_type.')

        # Decoder
        self.decoder = Decoder(backbone_type, output_stride, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]

        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, self.low_level_feature)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


class Decoder(nn.Module):
    def __init__(self, backbone_type: str, output_stride: int, num_classes: int):
        super(Decoder, self).__init__()
        assert output_stride in (8, 16)

        self.compress_low_level_feature1 = None
        if backbone_type == 'small':
            if output_stride == 16:
                self.compress_low_level_feature1 = self.make_compressor(64, 64)
            self.compress_low_level_feature2 = self.make_compressor(48, 32)
            self.compress_low_level_feature3 = self.make_compressor(24, 16)
        elif backbone_type == 'medium':
            if output_stride == 16:
                self.compress_low_level_feature1 = self.make_compressor(80, 64)
            self.compress_low_level_feature2 = self.make_compressor(48, 32)
            self.compress_low_level_feature3 = self.make_compressor(24, 16)
        elif backbone_type == 'large':
            if output_stride == 16:
                self.compress_low_level_feature1 = self.make_compressor(96, 64)
            self.compress_low_level_feature2 = self.make_compressor(64, 32)
            self.compress_low_level_feature3 = self.make_compressor(32, 16)
        else:
            raise NotImplementedError('Wrong backbone_type.')

        self.decode1 = None
        if output_stride == 16:
            self.decode1 = self.make_decoder(256 + 64, 256, 256)
        self.decode2 = self.make_decoder(256 + 32, 256, 128)
        self.decode3 = self.make_decoder(128 + 16, 128, 128)

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, low_level_feature: list[torch.Tensor]) -> torch.Tensor:
        if self.compress_low_level_feature1 is not None and self.decode1 is not None:
            low_level_feature1 = self.compress_low_level_feature1(low_level_feature.pop())
            x = F.interpolate(x, size=low_level_feature1.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat((x, low_level_feature1), dim=1)
            x = self.decode1(x)

        low_level_feature2 = self.compress_low_level_feature2(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature2), dim=1)
        x = self.decode2(x)

        low_level_feature3 = self.compress_low_level_feature3(low_level_feature.pop())
        x = F.interpolate(x, size=low_level_feature3.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feature3), dim=1)
        x = self.decode3(x)

        x = self.classifier(x)
        return x

    def make_compressor(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            models.modules.attention.SpatialAttention(kernel_size=5),
        )

    def make_decoder(self, in_channels: int, mid_channels: int, out_channels: int):
        return nn.Sequential(
            models.modules.conv.SeparableConv2d(in_channels, mid_channels, kernel_size=3,
                                                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            models.modules.conv.SeparableConv2d(mid_channels, out_channels, kernel_size=3,
                                                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Proposed('small', output_stride=16, num_classes=19).to(device)
    models.test.test_model(model, (1, 3, 512, 1024), '../runs', device)
