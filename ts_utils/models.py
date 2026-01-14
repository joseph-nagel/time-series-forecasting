'''Forecasting models.'''

import torch
import torch.nn as nn


class ForecastingModel(nn.Module):
    '''Forecasting model base class.'''

    def __init__(self):
        super().__init__()

    @property
    def num_weights(self) -> int:
        '''Get number of weights.'''
        return sum([p.numel() for p in self.parameters()])

    @property
    def num_trainable(self) -> int:
        '''Get number of trainable weights.'''
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class LSTM(ForecastingModel):
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, reset: bool = True) -> torch.Tensor:

        batch_size = x.shape[0]

        # hidden state initialization
        if reset:
            self.h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            self.c = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # LSTM prediction
        output, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        # last linear layer
        h = self.h[-1]
        h = h.view(batch_size, -1)
        out = self.fc(h)
        out = out.unsqueeze(1)

        return out

    def forecast(self, seq: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast based on recursive predictions.'''

        pred = seq

        preds = []
        for idx in range(steps):
            reset = idx == 0
            pred = self(pred, reset=reset)
            preds.append(pred)

        preds = torch.cat(preds, dim=1)

        return preds


class TCN(ForecastingModel):
    '''
    TCN wrapper module.

    Parameters
    ----------
    model : PyTorch module
        TCN forecasting model.

    '''

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forecast(self, seq: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast based on iteratively appended sequences.'''

        preds = []

        for _ in range(steps):
            pred = self(seq)

            preds.append(pred[...,-1:])

            if steps > 1:
                seq = torch.cat((seq[...,1:], pred[...,-1:]), dim=-1)

        preds = torch.cat(preds, dim=-1)

        return preds
