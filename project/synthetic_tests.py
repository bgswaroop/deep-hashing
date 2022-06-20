import math
from typing import Union

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torchvision.transforms import GaussianBlur

from project.DSH_push_pull_train import AddGaussianNoise


class DifferenceOfGaussian(torch.nn.Module):
    def __init__(self, kernel_size, gamma):
        super(DifferenceOfGaussian, self).__init__()
        self.gaussian1 = GaussianBlur(kernel_size=_pair(kernel_size), sigma=gamma)
        self.gaussian2 = GaussianBlur(kernel_size=_pair(kernel_size), sigma=4 * gamma)

    def forward(self, x):
        x_out = self.gaussian2(x) - self.gaussian1(x)
        return x_out


class PushPullConv2DUnit(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: Union[str, _size_2_t] = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            device=None,
            dtype=None):
        super(PushPullConv2DUnit, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if type(kernel_size) is tuple:
            assert kernel_size[0] % 2 == 1, 'First dimension of kernel_size must be odd'
            assert kernel_size[1] % 2 == 1, 'Second dimension of kernel_size must be odd'
        elif type(kernel_size) is int:
            assert kernel_size % 2 == 1, 'kernel_size must be odd'

        # kernel_init = torch.ones((1, 1, 5, 5))


        # self.DoG = DifferenceOfGaussian(kernel_size=(5, 5), gamma=1)
        # thresholded_kernel = torch.heaviside(self.DoG(kernel_init), values=0)
        # normalized_kernel = thresholded_kernel / torch.sum(thresholded_kernel)
        # self.push_kernel_weights = normalized_kernel

        # bias is always set to False in Conv layer

        self.push_kernel_weights = -torch.ones((1, 1, 5, 5))
        self.push_kernel_weights[0, 0, :, :2] = +1
        # self.push_kernel_weights[0, 0, :, 2] = 0

        # self.push_kernel_weights = -torch.ones((1, 1, 5, 5)) * 4 / 21
        # self.push_kernel_weights[0, 0, :2, :2] = +1
        # self.push_kernel_weights[0, 0, :, 2] = 0

        self.push_kernel_size = _pair(kernel_size)
        # self.pull_kernel_size = (self.push_kernel_size[0] * 2 - 1, self.push_kernel_size[1] * 2 - 1)
        self.pull_kernel_size = _pair(kernel_size)

        self.avg = torch.nn.AvgPool2d(
            kernel_size=_pair(kernel_size),
            stride=1,
            padding=tuple([math.floor(x / 2) for x in _pair(kernel_size)])
            # padding=tuple([math.floor(x / 2) for x in (5, 5)])
        )
        # learnable bias
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
            self.bias.data.uniform_(-1, 1)  # random weight initialization
        else:
            self.bias = torch.zeros(out_channels, device=device, dtype=dtype)

    @property
    def weight(self):
        return self.conv.weight

    @weight.setter
    def weight(self, value):
        self.conv.weight = value

    def forward(self, x):
        push_response = F.relu_(
            F.conv2d(input=x, weight=self.push_kernel_weights, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups))
        # pull_conv_kernel = F.interpolate(-self.push_kernel_weights, size=(5, 5), mode='bilinear')
        pull_conv_kernel = -self.push_kernel_weights
        pull_response = F.relu_(F.conv2d(input=x, weight=pull_conv_kernel, stride=self.stride, padding=self.padding,
                                         dilation=self.dilation, groups=self.groups))
        avg_pull_response = self.avg(pull_response)
        # avg_pull_response = pull_response
        x = F.relu_(push_response - 3 * avg_pull_response) + self.bias.view((1, -1, 1, 1))

        k = 0
        fig, ax = plt.subplots(2, 2)
        ax1 = ax[0][0]
        ax2 = ax[0][1]
        ax3 = ax[1][0]
        ax4 = ax[1][1]

        im1 = ax1.imshow(push_response[0, k, :, :].cpu())
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        ax1.set_title('push_response')

        im2 = ax2.imshow(x[0, k, :, :].cpu())
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        ax2.set_title('final_response')

        im3 = ax3.imshow(pull_response[0, k, :, :].cpu())
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')
        ax3.set_title('pull_response')

        im4 = ax4.imshow(F.relu_(avg_pull_response[0, k, :, :].cpu()))
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im4, cax=cax, orientation='vertical')
        ax4.set_title('avg_pull_response')

        plt.tight_layout()
        plt.show()

        return x


def main():
    x = torch.zeros((1, 1, 32, 32))
    # x[0, 0, :, :16] = 1.0
    x[0, 0, :16, :16] = 1.0
    x = AddGaussianNoise(mean=0, std=0.2)(x)

    PushPullConv2DUnit(in_channels=1, out_channels=32, kernel_size=(5, 5), padding='same', bias=False)(x)


if __name__ == '__main__':
    main()
