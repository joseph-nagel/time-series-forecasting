'''LSTM models.'''

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
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

        # create LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # (batch, steps, features) instead of (steps, batch, features)
        )

        # create output projection
        self.dense = nn.Linear(hidden_size, input_size)

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        '''Run LSTM model on input sequence.'''

        # run LSTM layers
        out, (h, c) = self.lstm(x, hx=hx)  # (batch, steps, hidden_size)

        # project to output
        out = self.dense(out)  # (batch, steps, input_size)

        if return_hidden:
            return out, (h, c)
        else:
            return out

    def forecast(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        '''Forecast next time step.'''

        # run LSTM layers
        out, (h, c) = self.lstm(x, hx=hx)  # (batch, steps, hidden_size)

        # get last hidden state
        last_hidden = out[:, -1].unsqueeze(1)  # (batch, steps=1, hidden_size)
        assert (last_hidden == h[-1].unsqueeze(1)).all()

        # project to output
        out = self.dense(last_hidden)  # (batch, steps=1, input_size)

        if return_hidden:
            return out, (h, c)
        else:
            return out

    @torch.inference_mode()
    def forecast_iteratively(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast multiple steps iteratively.'''
        hx = None
        preds = []

        for _ in range(steps):
            x, hx = self.forecast(x, hx=hx, return_hidden=True)
            preds.append(x)

        return torch.cat(preds, dim=1)
