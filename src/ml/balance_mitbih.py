import pandas as pd
from sklearn.utils import resample


def balance_by_upsampling(df: pd.DataFrame, label_col: int = 187, n_per_class: int = 20000, seed: int = 42) -> pd.DataFrame:
    parts = []
    for cls in sorted(df[label_col].unique()):
        df_c = df[df[label_col] == cls]
        if len(df_c) >= n_per_class:
            parts.append(df_c.sample(n=n_per_class, random_state=seed))
        else:
            parts.append(resample(df_c, replace=True, n_samples=n_per_class, random_state=seed + cls))

    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
