"""
Scaler factory for feature normalization.

Returns a scikit-learn scaler instance by name. The scaler type is
one of the genes in the evolutionary search space, allowing the GA
to optimize the preprocessing strategy alongside model architecture.
"""

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def get_scaler(name: str = "standard"):
    """Return a fresh scaler instance: 'standard', 'robust', or 'minmax'."""
    name = name.lower()

    if name == "standard":
        return StandardScaler()
    elif name == "robust":
        return RobustScaler()
    elif name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler: {name}")