import pandas as pd


def load_mitbih(train_csv: str, test_csv: str):
    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)

    train_df[187] = train_df[187].astype(int)
    test_df[187] = test_df[187].astype(int)

    return train_df, test_df
