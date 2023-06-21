from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split


def stratified_sampling(
    selection_rate,
    X,
    y=None,
    by=None
):
    """
    selection_rate: Probability of one sample to be selected
    X: pandas dataframe
    y: stratify and labels
    by: key or list of stratify key
    """
    if y is None:
        y = X[by]
    # print(y)

    X_s, _, _, _ = train_test_split(X, y, stratify=y, train_size=selection_rate)
    return X_s
