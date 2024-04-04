'''Data tools.'''

import torch
from torch.utils.data import Dataset


class SlidingWindows(Dataset):
    '''Sliding windows dataset.'''

    def __init__(self,
                 data,
                 window_size,
                 mode='next',
                 next_steps=1,
                 time_last=False):

        data = torch.as_tensor(data, dtype=torch.float32)

        if data.ndim == 2:
            self.data = data
        else:
            raise ValueError('Data needs two dimensions')

        self.window_size = min(window_size, len(self.data))

        if mode in ('next', 'shift'):
            self.mode = mode
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        self.next_steps = next_steps
        self.time_last = time_last

    def __len__(self):
        return len(self.data) - self.window_size + 1 - self.next_steps

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]

        if self.mode == 'next':
            y = self.data[idx+self.window_size:idx+self.window_size+self.next_steps]
        elif self.mode =='shift':
            y = self.data[idx+self.next_steps:idx+self.window_size+self.next_steps]

        if self.time_last:
            x = x.T
            y = y.T

        return x, y

