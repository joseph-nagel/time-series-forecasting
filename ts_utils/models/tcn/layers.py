'''TCN model layers.'''

import torch
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
    dilation : int
        Dilation parameter.
    bias : bool
        Switches bias on/off.

    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()

        # initialize conv layer
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # do not use built-in padding
            dilation=dilation,
            bias=bias
        )

        # determine left padding
        self.padding_left = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.padding_left, 0))  # manually pad on the left
        x = self.conv(x)
        return x


class CausalConvBlock(nn.Module):
    '''
    Causal convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolutional kernel size.
    dilation : int
        Dilation parameter.
    bias : bool
        Switches bias on/off.
    weight_norm : bool
        Whether to apply weight normalization.
    activation : str | None
        Name of the activation function.
    dropout : float | None
        Dropout rate.

    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        weight_norm: bool = False,
        activation: str | None = 'relu',
        dropout: float | None = None
    ):
        super().__init__()

        # create causal conv layer
        self.conv = CausalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias
        )

        # apply weight normalization
        if weight_norm:
            self.conv.conv = nn.utils.parametrizations.weight_norm(self.conv.conv)

        # create activation
        if activation is None:
            self.activ = None
        elif activation.lower() == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            self.activ = nn.Tanh()
        elif activation.lower() == 'relu':
            self.activ = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            self.activ = nn.LeakyReLU()
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        # create dropout
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.activ is not None:
            x = self.activ(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x
