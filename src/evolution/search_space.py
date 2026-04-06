"""
Hyperparameter search space for the evolutionary algorithm.

Defines the 15 genes that compose each individual's genotype, spanning
architecture (layers, units, activations), regularization (dropout, L2,
noise, clipping), training (optimizer, LR, loss, batch size), and
preprocessing (scaler type). The total combinatorial space is ~10^9.
"""

SEARCH_SPACE = {
    "n_layers": [1, 2],
    "units1": [64, 96, 128],
    "units2": [64, 96, 128],
    "units3": [32, 64, 96],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "l2": [0.0, 1e-6, 1e-5, 1e-4],
    "dense_units": [0, 64, 128, 256],
    "dense_activation": ["relu", "gelu", "elu", "leaky_relu"],
    "learning_rate": [1e-3, 5e-4, 3e-4, 2e-4, 1e-4],
    "batch_size": [128, 256],
    "clipnorm": [0.5, 1.0, 2.0, 5.0],
    "optimizer_name": ["adam", "adamw"],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "loss_name": ["mse", "mae", "huber1", "huber2"],
    "gaussian_noise_std": [0.0, 0.01],
    "scaler_name": ["standard", "robust", "minmax"],
    "lookback": [96, 120, 144],
    
}