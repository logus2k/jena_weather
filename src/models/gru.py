"""
Configurable GRU model builder for multi-step temperature forecasting.

Constructs a Keras Functional API model with 1–3 stacked GRU layers,
optional dense projection, and flexible regularization (L2, dropout,
Gaussian noise). Supports Adam and AdamW optimizers with gradient
clipping, and multiple loss functions (MSE, MAE, Huber). All
architectural and training parameters are exposed as arguments so the
evolutionary algorithm can optimize them jointly.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_gru_model(
    L, n_features, H,
    units1=64,
    units2=32,
    n_layers=1,
    dropout=0.2,
    l2=0.0,
    dense_units=0,
    dense_activation="relu",
    learning_rate=1e-3,
    clipnorm=1.0,
    optimizer_name="adam",
    units3=32,
    weight_decay=1e-5,
    loss_name="mse",
    gaussian_noise_std=0.0,
):
    reg = keras.regularizers.l2(l2) if l2 and l2 > 0 else None

    inputs = keras.Input(shape=(L, n_features))
    x = inputs

    if gaussian_noise_std and gaussian_noise_std > 0:
        x = layers.GaussianNoise(gaussian_noise_std)(x)

    x = layers.GRU(
        units1,
        return_sequences=(n_layers >= 2),
        dropout=dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=reg
    )(x)

    if n_layers >= 2:
        x = layers.GRU(
            units2,
            return_sequences=(n_layers == 3),
            dropout=dropout,
            recurrent_dropout=0.0,
            kernel_regularizer=reg
        )(x)

    if n_layers == 3:
        x = layers.GRU(
            units3,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=0.0,
            kernel_regularizer=reg
        )(x)

    if dense_units and dense_units > 0:
        if dense_activation == "leaky_relu":
            x = layers.Dense(dense_units, kernel_regularizer=reg)(x)
            x = layers.LeakyReLU(negative_slope=0.1)(x)
        else:
            x = layers.Dense(
                dense_units,
                activation=dense_activation,
                kernel_regularizer=reg
            )(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(H)(x)
    model = keras.Model(inputs, outputs)

    if optimizer_name.lower() == "adamw":
        opt = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm
        )
    else:
        opt = keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm
        )

    if loss_name == "mae":
        loss = "mae"
    elif loss_name == "huber1":
        loss = keras.losses.Huber(delta=1.0)
    elif loss_name == "huber2":
        loss = keras.losses.Huber(delta=2.0)
    else:
        loss = "mse"

    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model