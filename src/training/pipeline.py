"""Training pipeline - build model from config, train, return results."""

from src.models.gru import build_gru_model
from src.models.train_eval import train_model


def build_model_from_cfg(cfg, lookback, n_features, horizon):
    """Build a GRU model from a config dict.

    Args:
        cfg: dict with model architecture params (units1, units2, n_layers, etc.)
        lookback: input sequence length
        n_features: number of input features
        horizon: forecast horizon

    Returns:
        Compiled Keras model.
    """
    return build_gru_model(
        L=lookback,
        n_features=n_features,
        H=horizon,
        units1=cfg.get("units1", 96),
        units2=cfg.get("units2", 64),
        units3=cfg.get("units3", 96),
        n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.0),
        l2=cfg.get("l2", 1e-6),
        dense_units=cfg.get("dense_units"),
        dense_activation=cfg.get("dense_activation", "linear"),
        learning_rate=cfg.get("learning_rate", 2e-4),
        clipnorm=cfg.get("clipnorm", 2.0),
        optimizer_name=cfg.get("optimizer_name", "adam"),
        weight_decay=cfg.get("weight_decay", 0.0),
        loss_name=cfg.get("loss_name", "mse"),
        gaussian_noise_std=cfg.get("gaussian_noise_std", 0.0),
    )


def train_pipeline(cfg, X_train, y_train, X_val, y_val, lookback, n_features,
                   horizon, epochs=50, verbose=1):
    """Build and train a model from config.

    Args:
        cfg: dict with model architecture and training params
        X_train, y_train: training data
        X_val, y_val: validation data
        lookback, n_features, horizon: model dimensions
        epochs: max training epochs
        verbose: training verbosity

    Returns:
        (model, history) tuple.
    """
    model = build_model_from_cfg(cfg, lookback, n_features, horizon)

    batch_size = cfg.get("batch_size", 128)

    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )

    return model, history
