import numpy as np
import pandas as pd
import holidays

from src.data.loader import load_traffic_h5


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
    std = df.values.std()

    if std == 0:
        raise ValueError("Standard deviation is zero; cannot normalize data.")

    return (df - mean) / std, mean, std


def get_time_interval_minutes(df: pd.DataFrame) -> int:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")

    time_diffs = df.index.to_series().diff().dropna()

    if time_diffs.empty:
        raise ValueError("Cannot determine interval from fewer than 2 timestamps.")

    print("\nTop time intervals:")
    print(time_diffs.value_counts().head())

    interval_minutes = int(time_diffs.mode()[0].total_seconds() / 60)
    print(f"\nDetected interval: {interval_minutes} minutes")

    return interval_minutes


def get_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")

    interval_minutes = get_time_interval_minutes(df)
    steps_per_day = int((24 * 60) / interval_minutes)

    day_of_week = df.index.dayofweek.astype(np.float32)
    minutes_in_day = (df.index.hour * 60 + df.index.minute).astype(np.float32)
    time_step = (minutes_in_day / interval_minutes).astype(np.float32)

    time_of_day_sin = np.sin(2 * np.pi * time_step / steps_per_day).astype(np.float32)
    time_of_day_cos = np.cos(2 * np.pi * time_step / steps_per_day).astype(np.float32)

    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7).astype(np.float32)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7).astype(np.float32)

    is_weekend = (df.index.dayofweek >= 5).astype(np.float32)

    us_holidays = holidays.US(years=df.index.year.unique())
    holiday_dates = set(us_holidays.keys())
    is_holiday = pd.Index(df.index.date).isin(holiday_dates).astype(np.float32)

    temporal_df = pd.DataFrame(
        {
            "time_of_day_sin": time_of_day_sin,
            "time_of_day_cos": time_of_day_cos,
            "day_of_week_sin": day_of_week_sin,
            "day_of_week_cos": day_of_week_cos,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
        },
        index=df.index,
    )

    return temporal_df


def get_unique_holiday_dates(df: pd.DataFrame, temporal_df: pd.DataFrame) -> pd.Index:
    holiday_dates_in_data = pd.Index(df.index.date)[temporal_df["is_holiday"] == 1]
    unique_holiday_dates = pd.Index(holiday_dates_in_data).unique()
    return unique_holiday_dates


if __name__ == "__main__":
    dataset = "METR-LA"

    print(f"Loading dataset: {dataset}")
    df = load_traffic_h5(dataset)

    print("\nData shape:")
    print(df.shape)

    print("\nFirst 5 timestamps:")
    print(df.index[:5])

    temporal_df = get_temporal_features(df)

    print("\nTemporal feature shape:")
    print(temporal_df.shape)

    print("\nFirst 5 rows of temporal features:")
    print(temporal_df.head())

    print("\nTemporal feature columns:")
    print(temporal_df.columns.tolist())

    print("\nWeekend count:")
    print(int(temporal_df["is_weekend"].sum()))

    print("\nHoliday timestamp count:")
    print(int(temporal_df["is_holiday"].sum()))

    unique_holiday_dates = get_unique_holiday_dates(df, temporal_df)

    print("\nUnique holiday dates in dataset:")
    print(list(unique_holiday_dates))

    print("\nNumber of holiday days:")
    print(len(unique_holiday_dates))