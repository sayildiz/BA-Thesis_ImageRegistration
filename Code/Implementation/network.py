import torch
import torch.nn as nn
from skimage.exposure import match_histograms
from torch.distributions.normal import Normal

from layers import SpatialTransformer

def hist_match(img, ref_img):
    """
    Match histogram of ref_img onto img
    Parameters:
        img: Tensor (C, H, W)
            img which contrast will change
        ref_img: Tensor (C, H, W)
            reference img which histogram is applied to img
    Returns:
        matched_Image: Tensor (N, C, H, W)
    """
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    ref_img = ref_img.permute(1, 2, 0).detach().cpu().numpy()
    img_matched = match_histograms(img, ref_img, channel_axis=-1)
    return torch.from_numpy(img_matched.transpose((2, 0, 1))).unsqueeze(0)


class VoxelMorph(nn.Module):
    """
    VoxelMorph Network
    Parameters:
        imagedim: image dimensions as (height, width) for spatial transformer (mandatory)
        isgrey: for Unet to decide on input channels of 3+3 for RGB or 1+1 for greyscale images
    """

    def __init__(self,
                 imagedim=None,
                 isgrey=False,
                 cpu=False):
        super(VoxelMorph, self).__init__()

        if imagedim is None:
            raise ValueError("Image Dim must be defined like (Height, Width), must be divisible by 16 because of Unet")
        self.cpu = cpu
        self.unet = Unet(isgrey=isgrey)
        self.regressionLayer = RegressionLayer()
        self.spatialTransformer = SpatialTransformer(imagedim)

    def forward(self, moving, fixed, register=False):

        # match contrast histogram from he onto phh3
        if self.cpu:
            mov_hist_matched = hist_match(moving[0], fixed[0])
        else:
            mov_hist_matched = hist_match(moving[0], fixed[0]).cuda()  # comment out cuda() for cpu register/train

        # Localization/Unet
        x = torch.cat([mov_hist_matched, fixed], dim=1)
        x = self.unet(x)

        # Displacement Field generation
        displacement_field = self.regressionLayer(x)

        # Sampling Grid Generation -> Sampling -> registered image
        if register:
            registered = self.spatialTransformer(moving, displacement_field)
        else:
            registered = self.spatialTransformer(mov_hist_matched, displacement_field)

        return registered, displacement_field


class Unet(nn.Module):
    """
    Unet architecture like in Voxelmorph GitHub
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

        Parameters:
            isgrey: number of input channels matching two RGB or Grey images concatinated

        Returns:
            (16, H, W) Tensor with initial input height and width
    """

    def __init__(self, isgrey=False, ):
        super(Unet, self).__init__()

        # Channels for two concat images
        if isgrey:
            self.channels = 2
        else:
            self.channels = 6

        # model definition
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2)

        # Encoder Init
        self.encConv0 = Conv2DBlock(self.channels, 16)  # 1/2
        self.encConv1 = Conv2DBlock(16, 32)  # 1/4
        self.encConv2 = Conv2DBlock(32, 32)  # 1/8
        self.encConv3 = Conv2DBlock(32, 32)  # 1/16

        # Decoder Init
        self.decConv3 = Conv2DBlock(32, 32)  # 1/8
        self.decConv2 = Conv2DBlock(64, 32)  # 1/4
        self.decConv1 = Conv2DBlock(64, 32)  # 1/2
        self.decConv0 = Conv2DBlock(64, 32)  # 1/1 # because of 1. skip connection; in_channel = 32 + 16 = 48

        # Final Convs
        self.final0 = Conv2DBlock(48, 32)
        self.final1 = Conv2DBlock(32, 16)
        self.final2 = Conv2DBlock(16, 16)

    def forward(self, x):
        x_hist = []

        # encoder path
        out = self.encConv0(x)
        x_hist.append(out)
        out = self.down(out)  # 1/2
        out = self.encConv1(out)
        x_hist.append(out)
        out = self.down(out)  # 1/4
        out = self.encConv2(out)
        x_hist.append(out)
        out = self.down(out)  # 1/8
        out = self.encConv3(out)
        x_hist.append(out)
        out = self.down(out)  # 1/16

        # decoder path
        out = self.decConv3(out)
        out = self.up(out)  # 1/8
        out = torch.cat([out, x_hist.pop()], dim=1)  # skip connection
        out = self.decConv2(out)
        out = self.up(out)  # 1/4
        out = torch.cat([out, x_hist.pop()], dim=1)  # skip connection
        out = self.decConv1(out)
        out = self.up(out)  # 1/2
        out = torch.cat([out, x_hist.pop()], dim=1)  # skip connection
        out = self.decConv0(out)
        out = self.up(out)  # 1/1
        out = torch.cat([out, x_hist.pop()], dim=1)  # skip connection

        # final 3 convs
        out = self.final0(out)
        out = self.final1(out)
        out = self.final2(out)

        return out


class Conv2DBlock(nn.Module):
    """
    Convolution followed by leakyrelu(0.2) for 2D images
    inspired by ConvBlock class in https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/networks.py#L290
    """

    def __init__(self, in_channels, out_channels):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.lRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.lRelu(out)
        return out


class RegressionLayer(nn.Module):
    """
    Regression Layer in Spatial Transformer Context; creates Displacement field in Voxelmorph context
    inspired by https://github.com/voxelmorph/voxelmorph/blob/7cf611ab283b22424d82650d6e17d49b986e7981/voxelmorph/torch/networks.py#L210
    """

    def __init__(self):
        super(RegressionLayer, self).__init__()
        self.conv = nn.Conv2d(16, 2, kernel_size=3, padding=1)
        self.conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

    def forward(self, x):
        return self.conv(x)
