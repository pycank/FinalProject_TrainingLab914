from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import pandas as pd


def stratified_sampling(
    selection_rate,
    X,
    y=None,
    by=None,
    keep_thr=2
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
    keep_thr: str
        Keep sample if lable is less than or equal to keep threshold
    """
    if y is None:
        if isinstance(by, str):
            y = X[by]
        elif isinstance(by, list):
            y = X[by].agg('-'.join, axis=1).to_frame()[0]
        else:
            raise Exception("y and by is not defined!")

    count_df = y.value_counts().rename_axis('unique_values').reset_index(name='counts')
    keep_df = X[y.isin(count_df[count_df['counts']<=keep_thr]['unique_values'])]
    to_split_X_df = X[y.isin(count_df[count_df['counts']>keep_thr]['unique_values'])]
    to_split_y_df = y[y.isin(count_df[count_df['counts']>keep_thr]['unique_values'])]

    selection_rate = selection_rate - len(keep_df) / len(y)

    print("Stratified_sampling")
    print(f"  Keep: {len(keep_df)} ({len(keep_df) / len(y)})")
    print(f"  New selection_rate: {selection_rate}")

    if selection_rate <= 0.01:
        raise Exception(f"New selection rate too small: {selection_rate}")

    X_s, _, _, _ = train_test_split(to_split_X_df, to_split_y_df, stratify=to_split_y_df, train_size=selection_rate)
    X_total = pd.concat([X_s, keep_df])

    print(f"  Original: \t{len(X)}")
    print(f"  Total: \t{len(X_total)}")

    return X_total
