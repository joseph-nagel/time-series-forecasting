'''LSTM models.'''

import torch
import torch.nn as nn

from .utils import get_num_weights


class LSTM(nn.Module):
    '''
    Simple LSTM module.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Size of the hidden state.
    num_layers : int
        Number of LSTM layers.

    '''

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # (batch, steps, features) instead of (steps, batch, features)
        )

        # create output projection
        self.fc = nn.Linear(hidden_size, input_size)

    def num_weights(self, trainable: bool | None = None) -> int:
        '''Get number of weights.'''
        return get_num_weights(self, trainable)

    def forward(self, x: torch.Tensor, reset: bool = True) -> torch.Tensor:
        '''Forecast a single step.'''
        batch_size = x.shape[0]

        # initialize hidden state
        if reset:
            self.h = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=x.dtype,
                device=x.device
            )
            self.c = torch.zeros_like(self.h)

        # run LSTM layers
        output, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        # project to output
        h = self.h[-1]
        h = h.view(batch_size, -1)
        out = self.fc(h)
        out = out.unsqueeze(1)

        return out

    @torch.inference_mode()
    def forecast_iteratively(self, seq: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast iteratively.'''
        pred = seq
        preds = []

        for idx in range(steps):
            reset = idx == 0
            pred = self(pred, reset=reset)
            preds.append(pred)

        return torch.cat(preds, dim=1)
