'''Model layers.'''

import torch.nn as nn


class CausalConv(nn.Module):
    '''
    Causal convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolutional kernel size.
    stride : int
        Stride parameter.
    dilation : int
        Dilation parameter.
    bias : bool
        Switches bias on/off.

    '''

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True
    ):

        super().__init__()

        # initialize conv layer
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias
        )

        # determine (left) padding
        padding_left = (kernel_size - 1) * dilation
        padding_right = 0

        self.pad = (padding_left, padding_right)

    def forward(self, x):
        x = nn.functional.pad(x, self.pad)
        x = self.conv(x)
        return x

