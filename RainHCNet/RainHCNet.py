import torch
from torch import nn
import torch.nn.functional as F
from HCS_Attention import HCSA

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvDS_Up(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS_Up(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS_Up(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        x = torch.cat([x2, x1], dim=1)
        x= self.conv(x)
        return x


class RainHCNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernels_per_layer=2, bilinear=True):
        super(RainHCNet, self).__init__()
        self.in_channel = in_channel
        self.bilinear = bilinear
        self.inc = DoubleConvDS(self.in_channel, 64, kernels_per_layer=kernels_per_layer)
        
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
                
        self.stage = HCSA(in_channels=512, hidden_dimension=512,layers=2, downscaling_factor=1, num_heads=24, head_dim=32,window_size=9, relative_pos_embedding=True, h_w=[36, 36])#w 9 h_w 36
        
        self.conv=nn.Conv2d(512, out_channel, kernel_size=3, padding=1)
        self.conv1=nn.Conv2d(256, out_channel, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(128, out_channel, kernel_size=3, padding=1)
        self.conv3=nn.Conv2d(64, out_channel, kernel_size=3, padding=1)

        self.up1 = UpDS(1024, 256, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 128, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)   

        self.out_conv = nn.Conv2d(5 * out_channel, out_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        _, _, h, w = x.shape
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        side_outputs = []
        x5 = self.stage(x4)
        x5_o = F.interpolate(self.conv(x5), size=[h, w], mode='bilinear', align_corners=False)
        side_outputs.insert(0, x5_o)
        
        x = self.up1(x5, x4)
        x_o = F.interpolate(self.conv1(x), size=[h, w], mode='bilinear', align_corners=False)
        side_outputs.insert(0, x_o)
        
        x = self.up2(x, x3)
        x_o = F.interpolate(self.conv2(x), size=[h, w], mode='bilinear', align_corners=False)
        side_outputs.insert(0, x_o)
        
        x = self.up3(x, x2)
        x_o = F.interpolate(self.conv3(x), size=[h, w], mode='bilinear', align_corners=False)
        side_outputs.insert(0, x_o)
        
        x = self.up4(x, x1)
        x_o = F.interpolate(self.conv3(x), size=[h, w], mode='bilinear', align_corners=False)
        side_outputs.insert(0, x_o)
        
        out = self.out_conv(torch.concat(side_outputs, dim=1))
        
        return [out]+ side_outputs

