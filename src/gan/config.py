"""
TimeGAN hyperparameter configuration.

Defines the default settings for the TimeGAN model: sequence length
(matched to LOOKBACK + HORIZON = 144), GRU hidden dimension, number
of layers, and training epochs for each of the three training phases
(autoencoder, supervisor, adversarial).
"""

TIMEGAN_CONFIG = {
    "seq_len": 144,
    "hidden_dim": 24,
    "num_layers": 3,
    "batch_size": 128,
    "ae_epochs": 20,
    "sup_epochs": 20,
    "adv_epochs": 15,
    "learning_rate": 1e-3,
    "gamma": 1.0,
}