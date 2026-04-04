import numpy as np
import pandas as pd


def interpolate_speed(df: pd.DataFrame) -> pd.DataFrame:
    before_zeros = (df == 0).sum().sum()

    df_interp = df.copy()
    df_interp[df_interp == 0] = np.nan
    df_interp = df_interp.interpolate(method="linear", axis=0, limit_direction="both")

    remaining_nans = df_interp.isna().sum().sum()

    print(f"zeros replaced: {before_zeros}")
    print(f"remaining NaNs: {remaining_nans}")

    return df_interp


def normalize(df: pd.DataFrame):
    mean = df.values.mean()
    std  = df.values.std()
    return (df - mean) / std, mean, std