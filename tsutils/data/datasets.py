'''Data tools.'''

import torch
from torch.utils.data import Dataset


class SlidingWindows(Dataset):
    '''
    Sliding windows dataset.

    Summary
    -------
    A sliding window mechanism for time series data is implemented.
    The data should be passed as a (time steps, features)-shaped array.

    Parameters
    ----------
    data : array-like
        Data matrix with the shape (time steps, features).
    window_size : int
        Size of the time window.
    mode : {'next', 'shift'}
        Determines whether the next time steps
        or complete shifted windows are returned.
    next_steps : int
        Number of next steps or shift offset.
    time_last : bool
        Switch time from the first to the last axis.

    '''

    def __init__(self,
                 data,
                 window_size,
                 mode='next',
                 next_steps=1,
                 time_last=False):

        data = torch.as_tensor(data, dtype=torch.float32)

        if data.ndim == 1:
            self.data = data.unsqueeze(1) # (time steps, 1)
        elif data.ndim == 2:
            self.data = data # (time steps, features)
        else:
            raise ValueError('Data array needs two dimensions')

        self.window_size = min(abs(window_size), len(self.data))

        if mode in ('next', 'shift'):
            self.mode = mode
        else:
            raise ValueError(f'Invalid mode: {mode}')

        self.next_steps = abs(next_steps)
        self.time_last = time_last

    def __len__(self):
        return len(self.data) - self.window_size + 1 - self.next_steps

    def __getitem__(self, idx):

        # get time window
        x = self.data[idx:idx+self.window_size]

        # get the next time steps
        if self.mode == 'next':
            y = self.data[idx+self.window_size:idx+self.window_size+self.next_steps]

        # get the shifted time series
        elif self.mode =='shift':
            y = self.data[idx+self.next_steps:idx+self.window_size+self.next_steps]

        # shift time axis to the end
        if self.time_last:
            x = x.T
            y = y.T

        return x, y

