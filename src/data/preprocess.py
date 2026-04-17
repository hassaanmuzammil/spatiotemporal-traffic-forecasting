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

# Flags
def get_temporal_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Weekend
    is_weekend = (df.index.dayofweek >= 5).astype(np.float32)
    
    # Holidays
    holiday_dates = [
        '2012-01-01', '2012-01-02', '2012-01-16', '2012-02-20', 
        '2012-05-28', '2012-07-04', '2012-09-03', '2012-10-08', 
        '2012-11-11', '2012-11-12', '2012-11-22', '2012-12-25'
    ]
    is_holiday = df.index.normalize().isin(pd.to_datetime(holiday_dates)).astype(np.float32)
    
    return pd.DataFrame({
        'is_weekend': is_weekend,
        'is_holiday': is_holiday
    }, index=df.index)