# Time series forecasting

This repository contains a small playground for time series forecasting.
Simple experiments are conducted where different models are trained on the same toy dataset.
This includes classical **autoregressive moving-average** (ARMA) models as well as more modern
**long short-term memory** (LSTM) and **temporal convolutional network** (TCN) models.
All of these approaches leverage powerful mechanisms for sequence modeling.


## Instructions

- Install package:
    ```bash
    pip install -e .
    ```
- Train model:
    ```bash
    python scripts/main.py fit --config config/tcn.yaml
    ```
- Test model:
    ```bash
    python scripts/main.py test --config config/tcn.yaml \
      --ckpt_path run/tcn/version_0/checkpoints/best.ckpt
    ```
- Monitor metrics:
    ```bash
    tensorboard --logdir run/
    ```


## Notebooks

- [ARMA](notebooks/arma.ipynb)
- [LSTM](notebooks/lstm.ipynb)
- [TCN](notebooks/tcn.ipynb)
