'''TCN model.'''

from collections.abc import Sequence

import torch
import torch.nn as nn

from .layers import CausalConvBlock


class TCNModel(nn.Sequential):
    '''
    TCN model that stacks multiple causal convolutional layers.

    Parameters
    ----------
    num_channels : Sequence[int]
        Number of channels.
    kernel_size : int
        Convolutional kernel size.
    bias : bool
        Switches bias on/off.
    weight_norm : bool
        Whether to apply weight normalization.
    activation : str | None
        Name of the activation function.
    activate_last : bool
        Whether to activate the last layer.
    dropout : float | None
        Dropout rate.

    '''

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_size: int,
        bias: bool = True,
        weight_norm: bool = False,
        activation: str | None = 'relu',
        activate_last: bool = False,
        dropout: float | None = None
    ):
        super().__init__()

        # check number of layers
        if len(num_channels) < 2:
            raise ValueError('Number of channels needs at least two entries')

        num_layers = len(num_channels) - 1

        # assemble layers
        layers = []

        for idx, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            is_not_last = (idx < num_layers - 1)

            dilation = 2 ** idx  # set increasing dilation
            activation = activation if (is_not_last or activate_last) else None  # set last activation conditionally
            dropout = dropout if is_not_last else None  # deactivate dropout in last layer

            # create conv layer
            conv_block = CausalConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=bias,
                weight_norm=weight_norm,
                activation=activation,
                dropout=dropout
            )

            layers.append(conv_block)

            # TODO: add final projection layer

        # initialize module
        super().__init__(*layers)

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        '''Extract the last time step as a single-step forecast.'''
        return self(x)[..., -1:]  # (batch, channels, 1)
