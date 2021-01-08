import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import *

def census_transform(img: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Census transform
    
    calculates the census transform of an image of shape [N x C x H x W] with 
    batch size N, number of channels C, height H and width W. If C > 1, the census 
    transform is applied independently on each channel.

    Args:
        img (torch.Tensor): input image as torch.Tensor of shape [H x C x H x W]
        kernel_size (int): [description]

    Raises:
        NotImplementedError: kernel size should be 3 or 5.

    Returns:
        torch.Tensor: census transform of img
    """
 
    assert len(img.size()) == 4
    
    if kernel_size != 3 and kernel_size != 5:
        raise NotImplementedError

    n, c, h, w = img.size()

    # get the center idx of census filter
    census_center = margin = int((kernel_size - 1) / 2)

    # init census container
    census = torch.zeros((n, c, h - kernel_size + 1, w - kernel_size + 1), dtype=(torch.int32), device=(img.device))
    center_points = img[:, :, margin:h - margin, margin:w - margin]
    offsets = [(u, v) for v in range(kernel_size) for u in range(kernel_size) if not u == 1 == v]
    for u, v in offsets:
        census = census * 2 + (img[:, :, v:v + h - kernel_size + 1, u:u + w - kernel_size + 1] >= center_points).int()

    census = torch.nn.functional.pad(census, (margin, margin, margin, margin), mode='constant', value=0)

    return census


class CensusTransform(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self._kernel_size = kernel_size

    def forward(self, x):
        x = census_transform(x, self._kernel_size)
        return x


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YCbCr version of the image with shape :math:`(*, 3, H, W)`.
    
    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """
    
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))

    if len(image.shape) < 3 or image.shape[(-3)] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))

    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]
    delta = 0.5
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(image)))

    if len(image.shape) < 3 or image.shape[(-3)] != 3:
        raise ValueError('Input size must have a shape of (*, 3, H, W). Got {}'.format(image.shape))

    y = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]
    delta = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta
    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted

    return torch.stack([r, g, b], -3)


class RgbToYcbcr(nn.Module):
    """Convert an image from RGB to YCbCr.
    
    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> ycbcr = RgbToYcbcr()
        >>> output = ycbcr(input)  # 2x3x4x5
    """

    def __init__(self):
        super(RgbToYcbcr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_ycbcr(image)


class YcbcrToRgb(nn.Module):
    """
    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = YcbcrToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self):
        super(YcbcrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return ycbcr_to_rgb(image)


def hamming(left: torch.Tensor, right: torch.Tensor, maxdisp: int) -> torch.Tensor:
    """This function compute hamming distance for every single disparity.
    For example, a input (*, 1, H, W) will get the output (*, maxdisp, H, W)

    Args:
        left (torch.Tensor): It is a left image.
        right (torch.Tensor): It is a left image.
        maxdisp (int): the max disparity

    Raises:
        AssertionError: left should be 4-dims 
        AssertionError: right should be 4-dims 
        TypeError: left should be a torch.Tensor
        TypeError: right should be a torch.Tensor
        ValueError: left should be a int32
        ValueError: right should be a int32

    Returns:
        torch.Tensor: a hamming distance.
    """
    if not len(left.size()) == 4:
        raise AssertionError
    if not len(right.size()) == 4:
        raise AssertionError
        
    if not isinstance(left, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(left)))
    if not isinstance(right, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(right)))
    if left.dtype != torch.int32:
        raise ValueError('Input dtype must be torch.int32. Got {}'.format(left.dtype))
    if right.dtype != torch.int32:
        raise ValueError('Input dtype must be torch.int32. Got {}'.format(right.dtype))

    n, c, h, w = left.size()
    hamming_list = list()
    for d in range(maxdisp):
        left_valid, right_valid = left[:, :, :, d:w], right[:, :, :, 0:w - d]
        hamming = torch.zeros((left_valid.shape), dtype=(torch.int32), device=(left_valid.device))
        mask = torch.ones((left_valid.shape), dtype=(torch.int32), device=(left_valid.device))
        left_valid = left_valid.__xor__(right_valid)
        for i in range(23, -1, -1):
            hamming = hamming.add(left_valid.__and__(mask))
            left_valid = left_valid >> 1

        hamming = torch.nn.functional.pad(hamming, (0, d, 0, 0), mode='constant', value=0)
        hamming_list.append(hamming)

    return torch.cat(hamming_list, axis=1).float()


class Hamming(nn.Module):

    def __init__(self, maxdisp):
        super(Hamming, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return hamming(image1, image2, self.maxdisp)


def abs_diff(left: torch.Tensor, right: torch.Tensor, maxdisp: int) -> torch.Tensor:
    """This function compute absolute difference distance for every single disparity.
    
    Other two cost volumes C2 and C3 are populated with the absolute difference of
    the U and V values of the corresponding pixels in the left and right image.

    Args:
        left (torch.Tensor): It is a left image.
        right (torch.Tensor): It is a left image.
        maxdisp (int): the max disparity

    Returns:
        torch.Tensor: the absolute difference.
    
    Example:
        a input (*, 1, H, W) will get the output (*, maxdisp, H, W).
    """
    if not len(left.size()) == 4:
        raise AssertionError

    if not len(right.size()) == 4:
        raise AssertionError
       
    if not isinstance(left, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(left)))
    if not isinstance(right, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(type(right)))
    if left.dtype != torch.float32:
        raise ValueError('Input dtype must be torch.float32. Got {}'.format(left.dtype))
    if right.dtype != torch.float32:
        raise ValueError('Input dtype must be torch.float32. Got {}'.format(right.dtype))
    
    n, c, h, w = left.size()
    abs_diff_list = list()
    for d in range(maxdisp):
        left_valid, right_valid = left[:, :, :, d:w], right[:, :, :, 0:w - d]
        abs_diff = torch.abs(left_valid - right_valid)
        abs_diff = torch.nn.functional.pad(abs_diff, (0, d, 0, 0), mode='constant', value=0)
        abs_diff_list.append(abs_diff)

    return torch.cat(abs_diff_list, axis=1)


class AbsDiff(nn.Module):

    def __init__(self, maxdisp):
        super(AbsDiff, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        return abs_diff(image1, image2, self.maxdisp)


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(mid_channels), 
                                nn.ReLU(inplace=True), 
                                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
                                nn.BatchNorm2d(out_channels), 
                                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
                                    nn.MaxPool2d(2), 
                                    DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        
        return self.conv(x)
    

class UNet(nn.Module):
    """It is re-designed by FDSCS work.

    Args:
        n_channels ([int]): input channels
    """

    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.down1 = Down(in_channels=35, out_channels=32)
        self.down2 = Down(in_channels=32, out_channels=48)
        self.down3 = Down(in_channels=48, out_channels=64)
        self.down4 = Down(in_channels=64, out_channels=80)
        self.inc = DoubleConv(in_channels=80, out_channels=96)
        self.up1 = Up(in_channels=96, out_channels=80)
        self.up2 = Up(in_channels=80, out_channels=64)
        self.up3 = Up(in_channels=64, out_channels=48)
        self.up4 = Up(in_channels=48, out_channels=32)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.inc(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        out = self.up4(x, x1)

        return out


def conv2d_bn(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), 
                nn.BatchNorm2d(out_planes))


def conv2d(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))


class FDSCS(nn.Module):
    """
    https://github.com/ayanc/fdscs
    """

    def __init__(self, maxdisp):
        super(FDSCS, self).__init__()
        self.maxdisp = maxdisp
        self.yuv_to_rgb = YcbcrToRgb()
        self.rgb_to_yuv = RgbToYcbcr()
        self.census = CensusTransform(kernel_size=5)
        self.hamming = Hamming(maxdisp=(self.maxdisp))
        self.abs_diff = AbsDiff(maxdisp=(self.maxdisp))
     
        self.enc0_conv2d_bn = conv2d_bn(in_planes=384, out_planes=192, kernel_size=1, stride=1, padding=0)
        self.enc1_conv2d_bn = conv2d_bn(in_planes=192, out_planes=96, kernel_size=1, stride=1, padding=0)
        self.enc2_conv2d_bn = conv2d_bn(in_planes=96, out_planes=48, kernel_size=1, stride=1, padding=0)
        self.enc3_conv2d_bn = conv2d_bn(in_planes=48, out_planes=32, kernel_size=1, stride=1, padding=0)
        self.cenc0_conv2d_bn = conv2d_bn(in_planes=35, out_planes=32, kernel_size=3, stride=1, padding=(1,1))
        self.cenc1_conv2d_bn = conv2d_bn(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=(1,1))
        self.cenc2_conv2d_bn = conv2d_bn(in_planes=32, out_planes=32, kernel_size=3, stride=1, padding=(1,1))
        
        self.unet = UNet(n_channels=35)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def lowrescv(self, left, right, imsz=None, maxdisp=128):
        # rgb to yuv
        left, right = self.rgb_to_yuv(left), self.rgb_to_yuv(right)

        # doing census and then hamming at the first channal--Y channal.
        Y_cencus = self.hamming(self.census(left[:, 0:1, :, :]), self.census(right[:, 0:1, :, :]))

        # doing absolute diffrence at the U channal an the V channal.
        U_abs_diff = self.abs_diff(left[:, 1:2, :, :], right[:, 1:2, :, :])
        V_abs_diff = self.abs_diff(left[:, 2:3, :, :], right[:, 2:3, :, :])

        # unifiy the scale
        Y_cencus = (Y_cencus - 11.08282948) / 0.1949711
        U_abs_diff = (U_abs_diff - 0.02175535) / 35.91432953
        V_abs_diff = (V_abs_diff - 0.02679042) / 26.79782867

        # gether data to form a complete costs volume
        costs_volume = torch.cat([Y_cencus, U_abs_diff, V_abs_diff], axis=1)

        return costs_volume


    def forward(self, left, right):
        # costs_volume
        costs_volume = self.lowrescv(left, right)

        # costs_signature
        costs_signature = costs_volume
        costs_signature = self.enc0_conv2d_bn(costs_signature)
        costs_signature = self.enc1_conv2d_bn(costs_signature)
        costs_signature = self.enc2_conv2d_bn(costs_signature)
        costs_signature = self.enc3_conv2d_bn(costs_signature)

        costs_signature = torch.cat([left, costs_signature], axis=1)

        # costs_signature
        costs_signature = self.cenc0_conv2d_bn(costs_signature)
        costs_signature = self.cenc1_conv2d_bn(costs_signature)
        costs_signature = self.cenc2_conv2d_bn(costs_signature)

        unet_input = torch.cat([left, costs_signature], axis=1)
        
        # unet
        out = self.unet(unet_input)

        # final conv and up
        out = self.relu(self.out_conv(out) + 128.0)
        out = self.up(out)
        
        return out
