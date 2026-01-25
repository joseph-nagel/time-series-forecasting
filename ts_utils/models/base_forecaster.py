'''Forecasting base module.'''

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from .utils import get_num_weights


class BaseForecaster(LightningModule, ABC):
    '''
    Forecasting base module.

    Parameters
    ----------
    model : nn.Module
        Forecasting model.
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
        elif callable(loss):
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
        return get_num_weights(self, trainable)

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        '''Run forecast model on input sequence.'''
        raise NotImplementedError()

    @abstractmethod
    def forecast(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        '''Forecast next time step.'''
        raise NotImplementedError()

    @abstractmethod
    def forecast_iteratively(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        '''Forecast multiple steps iteratively.'''
        raise NotImplementedError()

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
    ):
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
    ):
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
