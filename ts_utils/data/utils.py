'''Synthetic data generation.'''

import numpy as np
from sklearn.model_selection import train_test_split


def make_sine_cosine(
    num_steps: int,
    max_length: float = 100.,
    noise_level: float = 0.1,
    random_seed: int | None = None,
    val_size: float | int | None = None
):
    '''
    Create bivariate sine/cosine time series data.

    Parameters
    ----------
    num_steps : int
        Number of time steps to generate.
    max_length : float
        Maximum time interval of the series.
    noise_level : float
        Gaussian noise standard deviation.
    random_seed : int | None
        Random seed.
    val_size : float | int | None
        Size of the validation set.

    '''

    # create sine/cosine data
    x_values = np.linspace(0, max_length, num_steps)

    sine = np.sin(x_values)
    cosine = np.cos(x_values)

    # add random noise
    rng = np.random.default_rng(seed=random_seed)

    noisy_sine = sine + rng.normal(loc=0, scale=noise_level, size=num_steps)
    noisy_cosine = cosine + rng.normal(loc=0, scale=noise_level, size=num_steps)

    # stack data into two columns
    data = np.column_stack((noisy_sine, noisy_cosine)).astype('float32')  # (time steps, 2)

    if val_size is None:
        return data

    else:
        # split into train/val. sets
        train_data, val_data = train_test_split(
            data,
            test_size=val_size,
            shuffle=False  # turn off shuffling for time series data
        )

        return train_data, val_data
