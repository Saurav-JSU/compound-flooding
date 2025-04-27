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
import logging
from typing import Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MISSING_SENTINEL = -99.9999
DEFAULT_MAX_GAP_HOURS = 2
REQUIRED_COLUMNS = ['datetime', 'sea_level']
OPTIONAL_COLUMNS = [
    'total_precipitation', 'surface_pressure', 'temperature_2m',
    'u_component_of_wind_10m', 'v_component_of_wind_10m', 'ground_precipitation'
]


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the input DataFrame has required columns and structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
        
    # Check required columns
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        return False, f"Missing required columns: {', '.join(missing_required)}"
    
    # Check for datetime column if not already indexed
    if 'datetime' not in df.columns and df.index.name != 'datetime':
        return False, "DataFrame must have 'datetime' column or datetime index"
    
    # Check if there's any data
    if df.empty:
        return False, "DataFrame is empty"
        
    # Check if sea_level is all NaN
    if df['sea_level'].isna().all():
        return False, "sea_level column contains only NaN values"
        
    return True, ""


def detrend_series(series: pd.Series) -> pd.Series:
    """
    Remove linear trend from a time series.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
        
    Returns
    -------
    pd.Series
        Detrended time series
    """
    # Convert to float to ensure proper calculation
    series = series.astype(float)
    
    # Only use valid (non-NaN) points for detrending
    valid = series.dropna()
    
    # Check if we have enough valid points for detrending
    if len(valid) < 2:
        logger.warning("Not enough valid points for detrending (min 2 required)")
        return series
        
    # Get positions of valid observations
    pos = np.arange(len(series))[~np.isnan(series.values)]
    
    # Fit trend on valid data
    try:
        m, b = np.polyfit(pos, valid.values, 1)
        # Subtract trend from full series, preserving NaNs
        detrended = series - (m * np.arange(len(series)) + b)
        logger.info(f"Detrended series with slope: {m:.2e}, intercept: {b:.4f}")
        return detrended
    except Exception as e:
        logger.warning(f"Failed to detrend series: {e}")
        return series


def interpolate_gaps(da: xr.DataArray, max_gap_hours: int) -> xr.DataArray:
    """
    Interpolate small gaps in a DataArray.
    
    Parameters
    ----------
    da : xr.DataArray
        Input data array
    max_gap_hours : int
        Maximum gap size in hours to interpolate
        
    Returns
    -------
    xr.DataArray
        Data array with gaps interpolated
    """
    if max_gap_hours <= 0:
        return da
        
    # Count missing values before interpolation
    n_missing_before = np.isnan(da.values).sum()
    if n_missing_before == 0:
        logger.info("No missing values to interpolate")
        return da
        
    # Find largest gap size (in number of consecutive NaNs)
    mask = np.isnan(da.values)
    longest_gap = 0
    current_gap = 0
    
    for i in range(len(mask)):
        if mask[i]:
            current_gap += 1
        else:
            longest_gap = max(longest_gap, current_gap)
            current_gap = 0
    
    # Also check the last gap
    longest_gap = max(longest_gap, current_gap)
    
    logger.info(f"Found {n_missing_before} missing values, longest gap: {longest_gap} time steps")
    
    # If longest gap (in time steps) exceeds our max gap hours, we interpolate up to max_gap_hours
    max_gap = np.timedelta64(max_gap_hours, 'h')
    
    # Perform the interpolation
    da_interp = da.interpolate_na(dim='datetime', max_gap=max_gap)
    
    # Count missing values after interpolation
    n_missing_after = np.isnan(da_interp.values).sum()
    n_interpolated = n_missing_before - n_missing_after
    
    logger.info(f"Interpolated {n_interpolated} values, {n_missing_after} missing values remain")
    
    return da_interp


def preprocess_dataframe(
    df: pd.DataFrame,
    detrend: bool = False,
    max_gap_hours: int = DEFAULT_MAX_GAP_HOURS,
    spike_threshold: Optional[float] = None
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
    df : pd.DataFrame
        Must include 'datetime' column and necessary variables.
    detrend : bool, default=False
        Remove linear trend from sea_level.
    max_gap_hours : int, default=2
        Max gap duration in hours for interpolation.
    spike_threshold : float or None, default=None
        Clip sea_level to this max value (m).

    Returns
    -------
    xr.Dataset
        Clean dataset ready for analysis
        
    Raises
    ------
    ValueError
        If DataFrame doesn't meet requirements
    """
    # Validate input
    is_valid, error_msg = validate_dataframe(df)
    if not is_valid:
        raise ValueError(f"Invalid input DataFrame: {error_msg}")
    
    if max_gap_hours < 0:
        raise ValueError(f"max_gap_hours must be non-negative, got {max_gap_hours}")
        
    # Copy DataFrame to avoid modifying the original
    df = df.copy()
    
    # Handle missing values
    logger.info(f"Replacing missing sentinel values ({MISSING_SENTINEL}) with NaN")
    df.replace(MISSING_SENTINEL, np.nan, inplace=True)
    
    # Unit conversions
    if 'total_precipitation' in df.columns:
        logger.info("Converting total_precipitation from m to mm/h")
        df['total_precipitation'] = df['total_precipitation'].astype(float) * 1000.0
        
    if 'surface_pressure' in df.columns:
        logger.info("Converting surface_pressure from Pa to hPa")
        df['surface_pressure'] = df['surface_pressure'].astype(float) / 100.0
        
    if 'temperature_2m' in df.columns:
        logger.info("Converting temperature_2m from K to °C")
        df['temperature_2m'] = df['temperature_2m'].astype(float) - 273.15
    
    # Handle datetime index
    if df.index.name != 'datetime':
        if 'datetime' in df.columns:
            logger.info("Setting datetime index")
            df.set_index('datetime', inplace=True)
        else:
            raise ValueError("DataFrame must have 'datetime' column")
    
    # Ensure datetime is properly formatted
    if not pd.api.types.is_datetime64_dtype(df.index):
        logger.info("Converting index to datetime")
        df.index = pd.to_datetime(df.index)
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    # Normalize to UTC
    logger.info("Normalizing datetime to UTC")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)
    
    # Drop duplicate timestamps
    n_before = len(df)
    df = df[~df.index.duplicated(keep='first')]
    n_after = len(df)
    if n_before > n_after:
        logger.info(f"Dropped {n_before - n_after} duplicate timestamps")
    
    # Detrend sea_level if requested
    if detrend and 'sea_level' in df.columns:
        logger.info("Detrending sea_level")
        df['sea_level'] = detrend_series(df['sea_level'])
    
    # Convert to xarray
    logger.info("Converting to xarray Dataset")
    ds = xr.Dataset.from_dataframe(df)
    
    # Interpolate small gaps in sea_level
    if 'sea_level' in ds and max_gap_hours > 0:
        logger.info(f"Interpolating sea_level gaps ≤ {max_gap_hours} hours")
        ds['sea_level'] = interpolate_gaps(ds['sea_level'], max_gap_hours)
    
    # Clip spikes
    if spike_threshold is not None and 'sea_level' in ds:
        if not np.isfinite(spike_threshold) or spike_threshold <= 0:
            logger.warning(f"Invalid spike_threshold: {spike_threshold}, skipping spike removal")
        else:
            n_spikes = (ds['sea_level'] > spike_threshold).sum().item()
            logger.info(f"Clipping {n_spikes} sea_level spikes above {spike_threshold}m")
            ds['sea_level'] = ds['sea_level'].clip(max=spike_threshold)
    
    # Final validation
    if np.isnan(ds['sea_level'].values).all():
        logger.warning("WARNING: sea_level contains only NaN values after preprocessing")
        
    return ds


if __name__ == '__main__':
    import argparse
    import sys
    import tempfile
    
    parser = argparse.ArgumentParser(description='Test preprocess module')
    parser.add_argument('--input-csv', help='Station CSV to preprocess (optional)')
    parser.add_argument('--detrend', action='store_true', help='Detrend sea_level')
    parser.add_argument('--max-gap', type=int, default=2, help='Max gap hours for interpolation')
    parser.add_argument('--spike', type=float, default=None, help='Spike threshold')
    args = parser.parse_args()
    
    try:
        if args.input_csv:
            # Real data test
            print(f"Testing with real data from {args.input_csv}")
            df = pd.read_csv(args.input_csv, parse_dates=['datetime'])
            
            # Print data info before preprocessing
            print(f"Original data: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            if 'sea_level' in df.columns:
                print(f"Sea level range: {df['sea_level'].min()} to {df['sea_level'].max()}")
                # Count missing values before preprocessing
                missing_before = df['sea_level'].isna().sum()
                sentinel_before = (df['sea_level'] == MISSING_SENTINEL).sum()
                print(f"Missing sea level values: {missing_before} NaN + {sentinel_before} sentinel values")
            
            ds = preprocess_dataframe(
                df, 
                detrend=args.detrend, 
                max_gap_hours=args.max_gap, 
                spike_threshold=args.spike
            )
            
            # Print data info after preprocessing
            print("\nPreprocessed data:")
            print(f"Dimensions: {dict(ds.sizes)}")  # Use sizes instead of dims to avoid warning
            print(f"Variables: {list(ds.data_vars)}")
            if 'sea_level' in ds:
                print(f"Sea level range: {float(ds.sea_level.min())} to {float(ds.sea_level.max())}")
                # Count missing values after preprocessing
                missing_after = np.isnan(ds['sea_level'].values).sum()
                print(f"Missing sea level values after preprocessing: {missing_after}")
            
            # Save preprocessed data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                ds.to_netcdf(tmp.name)
                print(f"\nSaved preprocessed data to {tmp.name}")
        else:
            # Synthetic data test
            print("Testing with synthetic data")
            
            # Create synthetic DataFrame
            dates = pd.date_range(start='2020-01-01', periods=100, freq='h')  # Use 'h' instead of 'H'
            sea_level = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.linspace(0, 1, 100) + 1.0  # with trend
            
            # Add some missing values and spikes
            sea_level[10:15] = np.nan  # 5-hour gap
            sea_level[50] = 10.0  # spike
            
            # Add a sentinel value too
            sea_level[70] = MISSING_SENTINEL
            
            # Create a synthetic DataFrame with minimal required columns
            df = pd.DataFrame({
                'datetime': dates,
                'sea_level': sea_level,
                'total_precipitation': np.random.rand(100) * 1e-3,  # in meters
                'surface_pressure': np.random.normal(101325, 1000, 100),  # in Pa
                'temperature_2m': np.random.normal(288, 5, 100),  # in K
            })
            
            # Print data info before preprocessing
            print(f"Original data: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            print(f"Sea level range: {df['sea_level'].min()} to {df['sea_level'].max()}")
            # Count missing and sentinel values
            missing_before = df['sea_level'].isna().sum()
            sentinel_before = (df['sea_level'] == MISSING_SENTINEL).sum()
            print(f"Missing sea level values: {missing_before} NaN + {sentinel_before} sentinel values")
            
            # Test preprocessing
            ds = preprocess_dataframe(
                df, 
                detrend=args.detrend,
                max_gap_hours=args.max_gap,
                spike_threshold=args.spike
            )
            
            # Print data info after preprocessing
            print("\nPreprocessed data:")
            print(f"Dimensions: {dict(ds.sizes)}")  # Use sizes instead of dims
            print(f"Variables: {list(ds.data_vars)}")
            print(f"Sea level range: {float(ds.sea_level.min())} to {float(ds.sea_level.max())}")
            
            # Check if gaps were interpolated
            n_missing = np.isnan(ds['sea_level'].values).sum()
            print(f"Missing values after interpolation: {n_missing}")
            
            # Verify interpolation with a visual check of values
            print("\nVerifying interpolation of 5-hour gap:")
            # Check the original gap location (indices 10-14)
            for i in range(9, 16):  # Show values around the gap
                print(f"Index {i}: {float(ds.sea_level.isel(datetime=i).values)}")
            
            # Check if spike was clipped if threshold was provided
            if args.spike is not None:
                max_sl = float(ds.sea_level.max())
                print(f"\nMaximum sea level after clipping: {max_sl}")
                print(f"Original spike at index 50: {float(ds.sea_level.isel(datetime=50).values)}")
                assert max_sl <= args.spike, f"Sea level max ({max_sl}) exceeds spike threshold ({args.spike})"
            
            # Check unit conversion
            if 'total_precipitation' in ds:
                print(f"\nTotal precipitation range: {float(ds.total_precipitation.min())} to {float(ds.total_precipitation.max())} mm/h")
            
            if 'surface_pressure' in ds:
                print(f"Surface pressure range: {float(ds.surface_pressure.min())} to {float(ds.surface_pressure.max())} hPa")
            
            if 'temperature_2m' in ds:
                print(f"Temperature range: {float(ds.temperature_2m.min())} to {float(ds.temperature_2m.max())} °C")
            
        print("\nPreprocess module smoke test completed successfully.")
    except Exception as e:
        print(f"Error during test: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)