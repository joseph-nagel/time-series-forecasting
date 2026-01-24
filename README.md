# Time series forecasting

This repository contains simple time series forecasting examples.
Synthetic experiments are conducted where different models are trained on the same toy dataset.
This includes classical **autoregressive moving-average** (ARMA) models as well as more modern
**long short-term memory** (LSTM) and **temporal convolutional network** (TCN) models.
All of these approaches leverage powerful mechanisms for sequence modeling.


## Installation

```bash
pip install -e .
```


## Training

```bash
python scripts/main.py fit --config config/tcn.yaml
```

```bash
tensorboard --logdir run/
```


## Notebooks

- [ARMA](notebooks/arma.ipynb)
- [LSTM](notebooks/lstm.ipynb)
- [TCN](notebooks/tcn.ipynb)
