import numpy as np
from sklearn.metrics import mean_absolute_error


def multioutput_mae(y_true, y_pred):
    return mean_absolute_error(y_true.flatten(), y_pred.flatten())


def permutation_feature_importance(model, X, y, feature_names, metric_fn=None, n_repeats=3, random_state=42):
    if metric_fn is None:
        metric_fn = multioutput_mae

    rng = np.random.default_rng(random_state)

    baseline_pred = model.predict(X, verbose=0)
    baseline_score = metric_fn(y, baseline_pred)

    importances = []

    for feat_idx, feat_name in enumerate(feature_names):
        scores = []

        for _ in range(n_repeats):
            X_permuted = X.copy()
            perm = rng.permutation(X.shape[0])
            X_permuted[:, :, feat_idx] = X_permuted[perm, :, feat_idx]

            perm_pred = model.predict(X_permuted, verbose=0)
            perm_score = metric_fn(y, perm_pred)
            scores.append(perm_score - baseline_score)

        importances.append({
            "feature": feat_name,
            "importance_mean": float(np.mean(scores)),
            "importance_std": float(np.std(scores)),
        })

    return baseline_score, importances
