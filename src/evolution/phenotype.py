import numpy as np

from src.features.scaling import get_scaler
from src.features.windowing import make_windows
from src.models.gru import build_gru_model
from src.models.train_eval import (
    train_model,
    evaluate_scaled_forecasts,
    evaluate_original_scale_forecasts,
)


def inverse_target_with_scaler(y_scaled, scaler, target_idx, n_features):
    original_shape = y_scaled.shape
    y_flat = y_scaled.reshape(-1)

    dummy = np.zeros((len(y_flat), n_features), dtype=float)
    dummy[:, target_idx] = y_flat

    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(original_shape)


def genotype_to_phenotype(
    genotype,
    df_train,
    df_val,
    final_feature_cols,
    target_idx,
    lookback=120,
    horizon=24,
):
    scaler = get_scaler(genotype["scaler_name"])

    X_train_df = df_train[final_feature_cols].copy()
    X_val_df = df_val[final_feature_cols].copy()

    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)

    X_train, y_train = make_windows(X_train_scaled, target_idx, lookback, horizon)
    X_val, y_val = make_windows(X_val_scaled, target_idx, lookback, horizon)

    model = build_gru_model(
        L=lookback,
        n_features=X_train.shape[2],
        H=horizon,
        units1=genotype["units1"],
        units2=genotype["units2"],
        units3=genotype["units3"],
        n_layers=genotype["n_layers"],
        dropout=genotype["dropout"],
        l2=genotype["l2"],
        dense_units=genotype["dense_units"],
        dense_activation=genotype["dense_activation"],
        learning_rate=genotype["learning_rate"],
        clipnorm=genotype["clipnorm"],
        optimizer_name=genotype["optimizer_name"],
        weight_decay=genotype["weight_decay"],
        loss_name=genotype["loss_name"],
        gaussian_noise_std=genotype["gaussian_noise_std"],
    )

    phenotype = {
        "scaler": scaler,
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
    return phenotype


def evaluate_phenotype(
    phenotype,
    genotype,
    final_feature_cols,
    target_idx,
    epochs=20,
    verbose=0,
):
    train_model(
        model=phenotype["model"],
        X_train=phenotype["X_train"],
        y_train=phenotype["y_train"],
        X_val=phenotype["X_val"],
        y_val=phenotype["y_val"],
        batch_size=genotype["batch_size"],
        epochs=epochs,
        verbose=verbose,
    )

    y_pred_val = phenotype["model"].predict(phenotype["X_val"], verbose=0)

    scaled_metrics = evaluate_scaled_forecasts(phenotype["y_val"], y_pred_val)

    n_features = len(final_feature_cols)
    y_val_inv = inverse_target_with_scaler(
        phenotype["y_val"], phenotype["scaler"], target_idx, n_features
    )
    y_pred_val_inv = inverse_target_with_scaler(
        y_pred_val, phenotype["scaler"], target_idx, n_features
    )

    original_metrics = evaluate_original_scale_forecasts(y_val_inv, y_pred_val_inv)

    return {
        "fitness": original_metrics["mae"],
        "metrics": {
            "mae_scaled": scaled_metrics["mae_scaled"],
            "rmse_scaled": scaled_metrics["rmse_scaled"],
            "mae_degC": original_metrics["mae"],
            "rmse_degC": original_metrics["rmse"],
        },
    }
