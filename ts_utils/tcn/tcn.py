'''TCN model.'''

from collections.abc import Sequence

from .base import BaseTCN
from .models import TCNModel


class TCN(BaseTCN):
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
