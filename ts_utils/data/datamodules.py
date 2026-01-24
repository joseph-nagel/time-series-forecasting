'''Datamodules.'''

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .utils import make_sine_cosine
from .datasets import SlidingWindowsDataset


class SineCosineDataModule(LightningDataModule):
    '''
    Sine and cosine datamodule.

    Parameters
    ----------
    num_steps : int
        Number of time steps to generate.
    max_length : float
        Maximum time interval of the series.
    noise_level : float
        Gaussian noise standard deviation.
    random_seed : int
        Random seed.
    val_size : float | int | None
        Size of the validation set.
    test_size : float | int | None
        Size of the test set.
    window_size : int
        Size of the time window.
    mode : {'next', 'shift'}
        Determines whether the next time steps
        or complete shifted windows are returned.
    next_steps : int
        Number of next steps or shift offset.
    time_last : bool
        Switch time from the first to the last axis.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(
        self,
        num_steps: int,
        max_length: float = 100.,
        noise_level: float = 0.1,
        random_seed: int = 42,
        train_size: float | int | None = None,
        val_size: float | int | None = None,
        test_size: float | int | None = None,
        window_size: int = 12,
        mode: str = 'next',
        next_steps: int = 1,
        time_last: bool = True,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()

        # set data params
        self.num_steps = num_steps
        self.max_length = max_length
        self.noise_level = noise_level
        self.random_seed = random_seed

        # set splitting params
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # set sliding window params
        self.window_size = window_size
        self.mode = mode
        self.next_steps = next_steps
        self.time_last = time_last

        # set data loader params
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        '''Generate (or download) data.'''
        data = make_sine_cosine(
            num_steps=self.num_steps,
            max_length=self.max_length,
            noise_level=self.noise_level,
            val_size=None,
            random_seed=self.random_seed
        )
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def setup(self, stage: str) -> None:
        '''Split data into train/val./test sets.'''

        # normalize sizes to fractions
        train_size = abs(self.train_size) if self.train_size is not None else 0
        val_size = abs(self.val_size) if self.val_size is not None else 0
        test_size = abs(self.test_size) if self.test_size is not None else 0

        total_size = train_size + val_size + test_size
        train_size = train_size / total_size
        val_size = val_size / total_size
        test_size = test_size / total_size

        # determine split indices
        num_samples = len(self.data)
        train_end_idx = int(num_samples * train_size)
        val_end_idx = train_end_idx + int(num_samples * val_size)

        # split data
        self.train_data = self.data[:train_end_idx]
        self.val_data = self.data[train_end_idx:val_end_idx]
        self.test_data = self.data[val_end_idx:] if val_end_idx < num_samples else self.val_data

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            self.train_set = SlidingWindowsDataset(
                data=self.train_data,
                window_size=self.window_size,
                mode=self.mode,
                next_steps=self.next_steps,
                time_last=self.time_last
            )
            self.val_set = SlidingWindowsDataset(
                data=self.val_data,
                window_size=self.window_size,
                mode=self.mode,
                next_steps=self.next_steps,
                time_last=self.time_last
            )

        # create test dataset
        elif stage == 'test':
            self.test_set = SlidingWindowsDataset(
                data=self.test_data,
                window_size=self.window_size,
                mode=self.mode,
                next_steps=self.next_steps,
                time_last=self.time_last
            )

    def train_dataloader(self) -> DataLoader:
        '''Create train dataloader.'''
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        '''Create val. dataloader.'''
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        '''Create test dataloader.'''
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )
