'''LSTM models.'''

import torch
import torch.nn as nn


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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, input_size)

    def num_weights(self, trainable: bool | None = None) -> int:
        '''Get number of weights.'''
        # get total number of weights
        if trainable is None:
            return sum([p.numel() for p in self.parameters()])

        # get number of trainable weights
        elif trainable:
            return sum([p.numel() for p in self.parameters() if p.requires_grad])

        # get number of frozen weights
        else:
            return sum([p.numel() for p in self.parameters() if not p.requires_grad])

    def forward(self, x: torch.Tensor, reset: bool = True) -> torch.Tensor:

        batch_size = x.shape[0]

        # initialize hidden state
        if reset:
            self.h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            self.c = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # run LSTM
        output, (self.h, self.c) = self.lstm(x, (self.h, self.c))

        # project to output
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
