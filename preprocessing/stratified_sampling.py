from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import pandas as pd


def stratified_sampling(
    selection_rate,
    X,
    y=None,
    by=None
):
    """
    Stratified sampling

    Arguments
    ---------
    selection_rate: float
        Probability of one sample to be selected
    X: pd.DataFrame
        Dataset
    y: list or None
        Stratify and labels
    by: str
        Key or list of stratify key
    """
    if y is None:
        y = X[by]
    # print(y)

    X_s, _, _, _ = train_test_split(X, y, stratify=y, train_size=selection_rate)
    return X_s
