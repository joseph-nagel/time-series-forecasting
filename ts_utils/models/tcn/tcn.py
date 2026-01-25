'''TCN model.'''

from collections.abc import Sequence

import torch

from ..base_forecaster import BaseForecaster
from .models import TCNModel


class TCN(BaseForecaster):
    '''
    TCN model.

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
    loss : str
        Loss function.
    lr : float
        Initial learning rate.

    '''

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_size: int,
        bias: bool = True,
        weight_norm: bool = False,
        activation: str | None = 'relu',
        activate_last: bool = False,
        dropout: float | None = None,
        loss: str = 'mse',
        lr: float = 1e-04
    ):

        model = TCNModel(
            num_channels=num_channels,
            kernel_size=kernel_size,
            bias=bias,
            weight_norm=weight_norm,
            activation=activation,
            activate_last=activate_last,
            dropout=dropout
        )

        super().__init__(model=model, loss=loss, lr=lr)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Run TCN model on input sequence.'''
        return self.model(x)  # (batch, channels, steps)

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        '''Forecast next time step.'''
        return self.model.forecast(x)  # (batch, channels, steps=1)

    @torch.inference_mode()
    def forecast_iteratively(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast multiple steps iteratively.'''
        return self.model.forecast_iteratively(x, steps=steps)
