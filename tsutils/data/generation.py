'''Synthetic data generation.'''

import numpy as np
from sklearn.model_selection import train_test_split


def make_sine_cosine(num_steps,
                     max_length=100.0,
                     noise_level=0.1,
                     val_size=None):
    '''Create bivariate sine/cosine data.'''

    x_values = np.linspace(0, max_length, num_steps)

    sine = np.sin(x_values)
    cosine = np.cos(x_values)

    noisy_sine = sine + np.random.normal(loc=0, scale=noise_level, size=num_steps)
    noisy_cosine = cosine + np.random.normal(loc=0, scale=noise_level, size=num_steps)

    data = np.column_stack((noisy_sine, noisy_cosine)).astype('float32') # (time steps, 2)

    if val_size is None:
        return data

    else:
        train_data, val_data = train_test_split(
            data,
            test_size=val_size,
            shuffle=False # turn off shuffling for time series data
        )

        return train_data, val_data

