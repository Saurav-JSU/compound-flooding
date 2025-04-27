# src/compound_flooding/preprocess.py
"""
Module: preprocess.py
Responsibilities:
- Normalize datetime index to UTC
- Convert units: total_precipitation (m→mm/h), surface_pressure (Pa→hPa), temperature_2m (K→°C)
- Optional linear detrending of sea_level
- Handle missing sentinel (-99.9999)
- Handle small gaps in sea_level via interpolation
- Clip sea_level spikes
- Return cleaned xarray.Dataset ready for Tier-1 analysis
"""
import numpy as np
import pandas as pd
import xarray as xr


def preprocess_dataframe(
    df: pd.DataFrame,
    detrend: bool = False,
    max_gap_hours: int = 2,
    spike_threshold: float = None
) -> xr.Dataset:
    """
    Clean raw station DataFrame:
    1. Replace missing sentinel (-99.9999) with NaN
    2. Convert units:
       - total_precipitation from meters to mm/h
       - surface_pressure from Pa to hPa
       - temperature_2m from Kelvin to °C
    3. Set datetime index and enforce UTC
    4. Drop duplicates
    5. Optional linear detrend of sea_level
    6. Interpolate gaps <= max_gap_hours hours in sea_level
    7. Clip sea_level spikes at spike_threshold (if provided)

    Parameters
    ----------
    df : pandas.DataFrame
        Must include 'datetime' column and necessary variables.
    detrend : bool
        Remove linear trend from sea_level.
    max_gap_hours : int
        Max gap duration in hours for interpolation.
    spike_threshold : float or None
        Clip sea_level to this max value (m).

    Returns
    -------
    xr.Dataset
    """
    # Copy frame and handle sentinel missing values
    df = df.copy()
    df.replace(-99.9999, np.nan, inplace=True)

    # Unit conversions
    if 'total_precipitation' in df:
        df['total_precipitation'] = df['total_precipitation'].astype(float) * 1000.0
    if 'surface_pressure' in df:
        df['surface_pressure'] = df['surface_pressure'].astype(float) / 100.0
    if 'temperature_2m' in df:
        df['temperature_2m'] = df['temperature_2m'].astype(float) - 273.15

    # Validate datetime
    if 'datetime' not in df.columns:
        raise KeyError("DataFrame must have 'datetime' column")

    # Index by datetime
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    # Convert to UTC then drop tzinfo for interpolation
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)

    # Drop duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Detrend sea_level if requested
    if detrend and 'sea_level' in df.columns:
        # Detrend sea_level using only valid (non-NaN) points
        ser = df['sea_level'].astype(float)
        valid = ser.dropna()
        if len(valid) >= 2:
            # positions of valid observations
            pos = np.arange(len(ser))[~np.isnan(ser.values)]
            # fit trend on valid data
            m, b = np.polyfit(pos, valid.values, 1)
            # subtract trend from full series, preserving NaNs
            df['sea_level'] = ser - (m * np.arange(len(ser)) + b)

    # Convert to xarray
    ds = xr.Dataset.from_dataframe(df)

    # Interpolate small gaps in sea_level
    if 'sea_level' in ds:
        gap = np.timedelta64(max_gap_hours, 'h')
        ds['sea_level'] = ds['sea_level'].interpolate_na(
            dim='datetime',
            max_gap=gap
        )

    # Clip spikes
    if spike_threshold is not None and 'sea_level' in ds:
        ds['sea_level'] = ds['sea_level'].clip(max=spike_threshold)

    return ds