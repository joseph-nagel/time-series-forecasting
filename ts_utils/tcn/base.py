'''TCN base module.'''

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class BaseTCN(LightningModule):
    '''
    TCN wrapper module.

    Parameters
    ----------
    model : nn.Module
        TCN model.
    loss : str
        Loss function.
    lr : float
        Initial learning rate.

    '''

    def __init__(
        self,
        model: nn.Module,
        loss: str = 'mse',
        lr: float = 1e-04
    ):
        super().__init__()

        self.model = model

        # TODO: allow for custom loss functions
        if loss.lower() == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif loss.lower() == 'mae':
            self.criterion = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f'Invalid loss function: {loss}')

        self.lr = abs(lr)

        self.save_hyperparameters(ignore=['model'])

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
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        loss = self.loss(*batch)
        self.log('test_loss', loss)
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
