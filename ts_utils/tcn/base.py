'''TCN base module.'''

from collections.abc import Callable

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics import MeanSquaredError, MeanAbsoluteError


class BaseTCN(LightningModule):
    '''
    TCN wrapper module.

    Parameters
    ----------
    model : nn.Module
        TCN model.
    loss : str | Callable
        Loss function.
    lr : float
        Initial learning rate.

    '''

    def __init__(
        self,
        model: nn.Module,
        loss: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = 'mse',
        lr: float = 1e-04
    ):
        super().__init__()

        # set model
        self.model = model

        # set loss function
        if isinstance(loss, str):
            if loss.lower() == 'mse':
                self.criterion = nn.MSELoss(reduction='mean')
            elif loss.lower() == 'mae':
                self.criterion = nn.L1Loss(reduction='mean')
            else:
                raise ValueError(f'Invalid loss function name: {loss}')
        elif isinstance(loss, nn.Module) or callable(loss):
            self.criterion = loss
        else:
            raise ValueError(f'Invalid loss function type: {type(loss)}')

        # set initial learning rate
        self.lr = abs(lr)

        # save hyperparameters
        self.save_hyperparameters(ignore=['model'])

        # set additional metrics
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()

        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Run the model over an input sequence.'''
        return self.model(x)  # (batch, channels, steps)

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        '''Extract the last time step as a single-step forecast.'''
        return self(x)[..., -1:]  # (batch, channels, 1)

    @torch.inference_mode()
    def forecast_iteratively(self, seq: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast iteratively (by appending steps).'''
        preds = []
        for _ in range(steps):
            pred = self.forecast(seq)  # (batch, channels, 1)
            preds.append(pred)
            if steps > 1:
                seq = torch.cat((seq[...,1:], pred), dim=-1)  # (batch, channels, steps)
        return torch.cat(preds, dim=-1)  # (batch, channels, steps)

    def loss(self, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        '''Compute loss.'''
        y_pred = self.forecast(x)
        return self.criterion(y_pred, y_target)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x_batch, y_batch = batch
        y_pred = self.forecast(x_batch)
        self.log_dict({
            'val_loss': self.criterion(y_pred, y_batch),
            'val_mse': self.val_mse(y_pred, y_batch),
            'val_mae': self.val_mae(y_pred, y_batch)
        })

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x_batch, y_batch = batch
        y_pred = self.forecast(x_batch)
        self.log_dict({
            'test_loss': self.criterion(y_pred, y_batch),
            'test_mse': self.test_mse(y_pred, y_batch),
            'test_mae': self.test_mae(y_pred, y_batch)
        })

    # TODO: enable LR scheduling
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
