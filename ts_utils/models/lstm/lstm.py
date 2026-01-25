'''LSTM Lightning module.'''

import torch

from ..base_forecaster import BaseForecaster
from .models import LSTMModel


class LSTMLightningModule(BaseForecaster):
    '''
    LSTM Lightning module.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Size of the hidden state.
    num_layers : int
        Number of LSTM layers.
    loss : str
        Loss function.
    lr : float
        Initial learning rate.

    '''

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        loss: str = 'mse',
        lr: float = 1e-04
    ):

        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        super().__init__(model=model, loss=loss, lr=lr)

        self.save_hyperparameters()

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        '''Run LSTM model on input sequence.'''
        return self.model(x, hx=hx, return_hidden=return_hidden)

    def forecast(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        '''Forecast next time step.'''
        return self.model.forecast(x, hx=hx, return_hidden=return_hidden)

    @torch.inference_mode()
    def forecast_iteratively(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast multiple steps iteratively.'''
        return self.model.forecast_iteratively(x, steps=steps)
