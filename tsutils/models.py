'''Forecasting models.'''

import torch
import torch.nn as nn


class LSTM(nn.Module):
    '''Simple LSTM module.'''

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1):

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

    @property
    def num_trainable(self):
        '''Get number of trainable weights.'''
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, x, reset=True):

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

    def forecast(self, seq, steps=1):
        '''Forecast based on recursive predictions.'''

        preds = []

        pred = self(seq, reset=True)
        preds.append(pred)

        for _ in range(steps - 1):
            pred = self(pred, reset=False)
            preds.append(pred)

        preds = torch.cat(preds, dim=1)

        return preds


class TCN(nn.Module):
    '''TCN wrapper module.'''

    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def num_trainable(self):
        '''Get number of trainable weights.'''
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, x):
        return self.model(x)

    def forecast(self, seq, steps=1):
        '''Forecast based on iteratively appended sequences.'''

        preds = []
        for _ in range(steps):
            pred = self(seq)

            preds.append(pred[...,-1:])

            if steps > 1:
                seq = torch.cat((seq[...,1:], pred[...,-1:]), dim=-1)

        preds = torch.cat(preds, dim=-1)

        return preds

