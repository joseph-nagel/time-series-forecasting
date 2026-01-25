'''Model utilities.'''

import torch.nn as nn


def freeze_weights(model: nn.Module) -> None:
    '''Freeze all model weights.'''
    for p in model.parameters():
        p.requires_grad = False


def get_num_weights(model: nn.Module, trainable: bool | None = None) -> int:
    '''Get number of model weights.'''

    # get total number of weights
    if trainable is None:
        return sum([p.numel() for p in model.parameters()])

    # get number of trainable weights
    elif trainable:
        return sum([p.numel() for p in model.parameters() if p.requires_grad])

    # get number of frozen weights
    else:
        return sum([p.numel() for p in model.parameters() if not p.requires_grad])
