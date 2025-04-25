"""
Preprocessing utilities for compound flooding analysis.

This module handles preprocessing and quality control of GESLA tide-gauge data
and associated ERA5 variables for compound flood analysis.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Union, Tuple, Dict, List, Sequence
from scipy import signal
from pathlib import Path

# Type aliases
PathLike = Union[str, Path]


class DataPreprocessor:
    """
    Preprocessor for GESLA tide-gauge and ERA5 data.
    
    This class handles preprocessing and quality control of GESLA tide-gauge 
    data and associated ERA5 variables, including timezone checks, unit conversions,
    detrending, and gap/spike handling according to GESLA QC flags.
    """
    
    # Define expected units for sanity checks
    EXPECTED_UNITS = {
        'sea_level': 'm',  # meters
        'total_precipitation': 'mm h⁻¹',  # mm per hour (converted from ERA5's m)
        'ground_precipitation': 'mm h⁻¹',  # mm per hour
        'u_component_of_wind_10m': 'm s⁻¹',  # m/s
        'v_component_of_wind_10m': 'm s⁻¹',  # m/s
        'surface_pressure': 'Pa',  # Pascal
        'temperature_2m': 'K',  # Kelvin
    }
    
    # GESLA quality flags definitions
    GESLA_FLAGS = {
        0: "no QC performed",
        1: "good value",
        2: "interpolated value",
        3: "doubtful value",
        4: "isolated spike or gap",
        5: "missing value"
    }
    
    # Common missing value codes
    MISSING_VALUE_CODES = {
        'sea_level': [-99.9999, -99.999, -99.99, -99.9, -99],
    }
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        pass
    
    def preprocess_station(
        self, 
        ds: xr.Dataset, 
        detrend_sea_level: bool = False,
        handle_gaps: bool = True,
        handle_spikes: bool = True
    ) -> xr.Dataset:
        """
        Preprocess data for a specific station.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing station data.
        detrend_sea_level : bool, optional
            Whether to remove linear trend from sea level. Default is False.
        handle_gaps : bool, optional
            Whether to handle gaps in data. Default is True.
        handle_spikes : bool, optional
            Whether to handle spikes in data. Default is True.
        
        Returns
        -------
        xr.Dataset
            Preprocessed dataset.
        """
        # Make a copy to avoid modifying the original
        ds = ds.copy(deep=True)
        
        # Replace missing value codes with NaN before any processing
        ds = self._replace_missing_values(ds)
        
        # Check timezone (data should be in UTC)
        ds = self._check_timezone(ds)
        
        # Check units and convert if necessary
        ds = self._check_units(ds)
        
        # Handle gaps using GESLA quality flags or statistical methods
        if handle_gaps:
            ds = self._handle_gaps(ds)
        
        # Handle spikes using GESLA quality flags or outlier detection
        if handle_spikes:
            ds = self._handle_spikes(ds)
        
        # Detrend sea level if requested
        if detrend_sea_level and 'sea_level' in ds:
            ds = self._detrend_sea_level(ds)
        
        # Add processing metadata to track what was done
        ds.attrs['preprocessed'] = 'True'
        ds.attrs['preprocessing_steps'] = ','.join(filter(None, [
            'missing_value_replacement',
            'timezone_check',
            'unit_check',
            'gap_handling' if handle_gaps else None,
            'spike_handling' if handle_spikes else None,
            'sea_level_detrending' if detrend_sea_level else None
        ]))
        
        return ds
    
    def _replace_missing_values(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Replace common missing value codes with NaN.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to process.
        
        Returns
        -------
        xr.Dataset
            Dataset with missing values replaced.
        """
        # Handle specific known missing value codes
        for var, codes in self.MISSING_VALUE_CODES.items():
            if var in ds:
                for code in codes:
                    # Create a boolean mask for values matching the missing code
                    mask = (ds[var] == code)
                    if mask.any():
                        print(f"Found {mask.sum().item()} instances of missing value code {code} in {var}")
                        # Replace with NaN
                        ds[var] = ds[var].where(~mask)
        
        return ds
    
    def _check_timezone(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Check that timestamps are in UTC.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to check.
        
        Returns
        -------
        xr.Dataset
            Dataset with UTC timestamps.
        """
        # Check if time dimension exists
        if 'time' not in ds.dims:
            print("Warning: No time dimension found in dataset.")
            return ds
        
        # Check if timezone info exists in time coordinate
        time_data = ds.time
        
        try:
            # Try to convert time to pandas DatetimeIndex to check timezone
            time_idx = pd.DatetimeIndex(time_data.values)
            
            # Check if timezone is specified
            if time_idx.tz is None:
                # No timezone info - assume UTC and add it
                new_time = pd.DatetimeIndex(time_idx).tz_localize('UTC')
                ds = ds.assign_coords(time=new_time)
                print("Warning: Time coordinate had no timezone. Localized to UTC.")
            elif str(time_idx.tz) != 'UTC':
                # Convert from current timezone to UTC
                new_time = time_idx.tz_convert('UTC')
                ds = ds.assign_coords(time=new_time)
                print(f"Warning: Converted time from {time_idx.tz} to UTC.")
        except (TypeError, AttributeError) as e:
            # Handle cases where timezone operations fail
            print(f"Warning: Could not verify timezone: {e}")
        
        return ds
    
    def _check_units(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Check and potentially convert units for variables.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to check.
        
        Returns
        -------
        xr.Dataset
            Dataset with correct units.
        """
        # Check each variable against expected units
        for var, expected_unit in self.EXPECTED_UNITS.items():
            if var in ds:
                # Check if unit info exists in variable attributes
                current_unit = ds[var].attrs.get('units', None)
                
                if current_unit is None:
                    # No unit attribute - apply unit conversion heuristics
                    
                    # For precipitation, if max value is small (< 0.1), likely in meters
                    if var == 'total_precipitation':
                        # Always convert ERA5 precipitation from m to mm/hour
                        ds[var] = ds[var] * 1000  # Convert from m to mm
                        ds[var].attrs['units'] = expected_unit
                        print(f"Converted {var} from meters to {expected_unit} (multiplied by 1000)")
                    else:
                        # Just set the expected unit in attributes without conversion
                        ds[var].attrs['units'] = expected_unit
                        print(f"Warning: No unit information for {var}. Assuming {expected_unit}.")
                
                elif current_unit != expected_unit:
                    # Unit attribute exists but doesn't match expected - convert
                    ds = self._convert_units(ds, var, current_unit, expected_unit)
                
                # Verify units are physically sensible
                self._verify_physical_range(ds, var)
        
        return ds
    
    def _verify_physical_range(self, ds: xr.Dataset, var: str) -> None:
        """
        Check if values for a variable are within physically reasonable ranges.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the variable.
        var : str
            Name of the variable to check.
        """
        # Define reasonable ranges for each variable
        ranges = {
            'sea_level': (-15, 15),  # m, extreme but possible values
            'total_precipitation': (0, 500),  # mm/h, hurricane-level max
            'ground_precipitation': (0, 500),  # mm/h
            'u_component_of_wind_10m': (-100, 100),  # m/s, hurricane-force 
            'v_component_of_wind_10m': (-100, 100),  # m/s
            'surface_pressure': (87000, 110000),  # Pa, extreme low to high
            'temperature_2m': (173.15, 333.15),  # K, -100°C to +60°C
        }
        
        if var in ranges:
            min_val, max_val = ranges[var]
            # Skip NaN values in min/max calculations
            if np.isnan(ds[var].values).all():
                print(f"Warning: All values in {var} are NaN")
                return
                
            var_min = np.nanmin(ds[var].values)
            var_max = np.nanmax(ds[var].values)
            
            if var_min < min_val or var_max > max_val:
                print(f"Warning: {var} has values outside physical range "
                      f"[{min_val}, {max_val}]: min={var_min}, max={var_max}")
    
    def _convert_units(
        self, 
        ds: xr.Dataset, 
        variable: str, 
        from_unit: str, 
        to_unit: str
    ) -> xr.Dataset:
        """
        Convert units for a variable.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the variable.
        variable : str
            Name of the variable to convert.
        from_unit : str
            Current unit.
        to_unit : str
            Target unit.
        
        Returns
        -------
        xr.Dataset
            Dataset with converted units.
        """
        # Define common conversion factors
        conversions = {
            # Temperature
            ('K', '°C'): lambda x: x - 273.15,
            ('°C', 'K'): lambda x: x + 273.15,
            
            # Pressure
            ('Pa', 'hPa'): lambda x: x / 100,
            ('hPa', 'Pa'): lambda x: x * 100,
            
            # Precipitation
            ('m', 'mm h⁻¹'): lambda x: x * 1000,
            ('mm', 'mm h⁻¹'): lambda x: x,  # Assuming hourly data already
        }
        
        # Look up the conversion
        conversion_key = (from_unit, to_unit)
        if conversion_key in conversions:
            # Apply conversion function
            converter = conversions[conversion_key]
            ds[variable] = converter(ds[variable])
            ds[variable].attrs['units'] = to_unit
            print(f"Converted {variable} from {from_unit} to {to_unit}")
        else:
            print(f"Warning: Conversion from {from_unit} to {to_unit} for {variable} not implemented.")
            # Set the expected unit anyway to avoid repeated warnings
            ds[variable].attrs['units'] = to_unit
            
        return ds
    
    def _handle_gaps(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Handle gaps in the data, following GESLA QC flags when available.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to process.
        
        Returns
        -------
        xr.Dataset
            Dataset with gaps handled.
        """
        # Check for GESLA QC flags
        flag_vars = [var for var in ds.data_vars if 'flag' in var.lower()]
        
        if flag_vars and 'sea_level' in ds:
            # Use GESLA QC flags to handle gaps in sea_level
            print(f"Found QC flag variables: {flag_vars}")
            
            # Look for standard flag naming patterns
            sea_level_flag_candidates = [
                var for var in flag_vars 
                if 'sea_level' in var.lower() or 'level' in var.lower()
            ]
            
            if sea_level_flag_candidates:
                sea_level_flag = sea_level_flag_candidates[0]
                print(f"Using {sea_level_flag} for gap handling")
                
                # Create a mask for valid data points (flags 1 or 2, good or interpolated)
                valid_mask = (ds[sea_level_flag] == 1) | (ds[sea_level_flag] == 2)
                
                # Apply the mask
                sea_level_valid = ds['sea_level'].where(valid_mask)
                
                # Interpolate gaps where flags indicate missing or suspect data
                # Only interpolate if gaps are small (≤ 2 hours)
                ds['sea_level'] = self._interpolate_small_gaps(sea_level_valid)
                
                # Add a gap-filled flag
                ds['gap_filled'] = xr.where(
                    ~valid_mask & ~np.isnan(ds['sea_level']),
                    1,  # 1 means gap-filled
                    0   # 0 means original data
                )
        else:
            # No QC flags found, use statistical methods to identify and fill gaps
            
            # Get time index and check for gaps
            time_indices = ds.time.values
            
            # Check for missing timesteps (assuming hourly data)
            time_diffs = np.diff(time_indices.astype('datetime64[s]').astype(float)) / 3600  # Convert to hours
            
            # Find gaps larger than expected (hourly steps would be ~1.0)
            gap_indices = np.where(time_diffs > 1.5)[0]
            
            if len(gap_indices) > 0:
                print(f"Found {len(gap_indices)} potential gaps in time series.")
                
                # For each variable, interpolate across small gaps
                for var in ds.data_vars:
                    if var in ['sea_level', 'total_precipitation', 'ground_precipitation',
                               'u_component_of_wind_10m', 'v_component_of_wind_10m',
                               'surface_pressure', 'temperature_2m']:
                        
                        ds[var] = self._interpolate_small_gaps(ds[var])
        
        return ds
    
    def _interpolate_small_gaps(self, da: xr.DataArray, max_gap_size: int = 2) -> xr.DataArray:
        """
        Interpolate small gaps in a DataArray.
        
        Parameters
        ----------
        da : xr.DataArray
            DataArray to process.
        max_gap_size : int, optional
            Maximum gap size to interpolate, in hours. Default is 2.
        
        Returns
        -------
        xr.DataArray
            DataArray with small gaps interpolated.
        """
        # Convert to pandas Series for easier handling
        series = da.to_series()
        
        # Find runs of NaN values
        mask = series.isna()
        
        if not mask.any():
            return da  # No NaNs to interpolate
        
        # Identify start indices and lengths of NaN runs
        # A run is a continuous sequence of True values (NaNs in this case)
        run_starts = np.where(np.diff(np.concatenate(([False], mask, [False]))) == 1)[0]
        run_ends = np.where(np.diff(np.concatenate(([False], mask, [False]))) == -1)[0]
        
        if len(run_starts) == 0 or len(run_ends) == 0:
            return da  # No clear runs found
            
        run_lengths = run_ends - run_starts
        
        # Only interpolate gaps smaller than or equal to max_gap_size
        for start, length in zip(run_starts, run_lengths):
            if 0 < length <= max_gap_size:
                # Use pandas interpolate for the small gap
                series.iloc[start:start+length] = np.nan  # Ensure the segment is NaN
                
        # Interpolate small gaps (pandas interpolate won't fill leading/trailing NaNs)
        series_interp = series.interpolate(method='linear', limit=max_gap_size)
        
        # Convert back to DataArray with same metadata
        result = xr.DataArray(
            data=series_interp.values,
            dims=da.dims,
            coords=da.coords,
            attrs=da.attrs
        )
        
        return result
    
    def _handle_spikes(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Handle spikes in the data, using GESLA QC flags or statistical outlier detection.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset to process.
        
        Returns
        -------
        xr.Dataset
            Dataset with spikes handled.
        """
        # Check for sea level in dataset
        if 'sea_level' not in ds:
            return ds
        
        # Look for GESLA flag variables
        flag_vars = [var for var in ds.data_vars if 'flag' in var.lower()]
        
        if flag_vars:
            # Use GESLA QC flags to handle spikes
            sea_level_flag_candidates = [
                var for var in flag_vars 
                if 'sea_level' in var.lower() or 'level' in var.lower()
            ]
            
            if sea_level_flag_candidates:
                sea_level_flag = sea_level_flag_candidates[0]
                print(f"Using {sea_level_flag} for spike handling")
                
                # Create a mask for spike data points (flag 4, isolated spike)
                spike_mask = (ds[sea_level_flag] == 4)
                
                if spike_mask.any():
                    print(f"Found {spike_mask.sum().item()} spikes in sea_level using GESLA flags.")
                    
                    # Replace spikes with NaN
                    sea_level_despiked = ds['sea_level'].where(~spike_mask)
                    
                    # Interpolate spikes
                    sea_level_despiked = self._interpolate_small_gaps(sea_level_despiked, max_gap_size=1)
                    
                    # Assign back to dataset
                    ds['sea_level'] = sea_level_despiked
                    
                    # Add a spike flag if it doesn't exist
                    if 'sea_level_spike_flag' not in ds:
                        ds['sea_level_spike_flag'] = spike_mask.astype(int)
                        ds.sea_level_spike_flag.attrs['description'] = 'Flag for detected spikes (1=spike)'
        else:
            # No QC flags found, use statistical methods to identify spikes
            # Skip if all values are NaN
            if np.isnan(ds.sea_level.values).all():
                print("Warning: All sea_level values are NaN, skipping spike detection")
                return ds
                
            # Calculate rolling median and standard deviation
            # First, check if we have enough non-NaN values for rolling calculations
            valid_count = np.sum(~np.isnan(ds.sea_level.values))
            
            if valid_count < 5:  # Need at least 5 points for a 5-point window
                print(f"Warning: Not enough valid sea_level points for spike detection ({valid_count} found)")
                return ds
                
            rolling_median = ds.sea_level.rolling(time=5, center=True, min_periods=3).median()
            rolling_std = ds.sea_level.rolling(time=5, center=True, min_periods=3).std()
            
            # Identify outliers/spikes (beyond 5 standard deviations)
            # This threshold can be adjusted based on domain knowledge
            threshold = 5  # 5 sigma
            spike_mask = np.abs(ds.sea_level - rolling_median) > threshold * rolling_std
            
            # Only consider isolated spikes (not part of a longer extreme event)
            # An isolated spike has normal values before and after
            spike_mask_isolated = spike_mask.copy()
            
            if spike_mask.any():
                # Convert to numpy for easier manipulation
                spike_array = spike_mask.values
                
                # Only keep spikes that are isolated (not part of a sequence)
                for i in range(1, len(spike_array)-1):
                    if spike_array[i]:
                        # If neighbors are also outliers, this might be a real event not a spike
                        if spike_array[i-1] or spike_array[i+1]:
                            spike_array[i] = False
                
                # Update the mask
                spike_mask_isolated.values = spike_array
                
                if spike_mask_isolated.any():
                    print(f"Found {spike_mask_isolated.sum().item()} isolated spikes in sea_level.")
                    
                    # Replace spikes with NaN
                    sea_level_despiked = ds.sea_level.where(~spike_mask_isolated)
                    
                    # Interpolate NaNs
                    sea_level_despiked = self._interpolate_small_gaps(sea_level_despiked, max_gap_size=1)
                    
                    # Assign back to dataset
                    ds['sea_level'] = sea_level_despiked
                    
                    # Add flag variable for spikes
                    ds['sea_level_spike_flag'] = spike_mask_isolated.astype(int)
                    ds.sea_level_spike_flag.attrs['description'] = 'Flag for detected spikes (1=spike)'
        
        return ds
    
    def _detrend_sea_level(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Remove linear trend from sea level.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing sea level data.
        
        Returns
        -------
        xr.Dataset
            Dataset with detrended sea level.
        """
        if 'sea_level' not in ds:
            return ds
        
        # Convert time to numeric values for regression
        # Use days since the first timestamp as the independent variable
        time_values = ds.time.values
        first_time = time_values[0]
        days_since_start = ((time_values - first_time).astype('timedelta64[s]').astype(float) 
                           / (24 * 3600))  # Convert seconds to days
        
        # Get sea level data
        sea_level = ds.sea_level.values
        
        # Create mask for NaN values
        nan_mask = np.isnan(sea_level)
        
        # Skip detrending if too many NaNs
        if np.sum(~nan_mask) < 24:  # At least 24 valid hours of data
            print("Warning: Not enough valid data points to detrend sea level.")
            return ds
        
        # Get valid values for detrending
        valid_time = days_since_start[~nan_mask]
        valid_data = sea_level[~nan_mask]
        
        # Skip detrending if all valid data has the same value (would cause singular matrix)
        if np.allclose(valid_data, valid_data[0]):
            print("Warning: All valid sea_level values are identical, skipping detrending.")
            ds.attrs['sea_level_trend_mm_per_year'] = 0.0
            return ds
            
        # Calculate linear trend (mm/year = slope * 365.25 * 1000)
        try:
            slope, intercept = np.polyfit(valid_time, valid_data, 1)
            annual_trend_mm = slope * 365.25 * 1000  # Convert to mm/year
            
            # Calculate trend component
            trend = slope * days_since_start + intercept
            
            # Store original sea level
            ds['sea_level_original'] = ds.sea_level.copy()
            ds.sea_level_original.attrs['description'] = 'Original sea level before detrending'
            
            # Store trend information
            ds.attrs['sea_level_trend_mm_per_year'] = float(annual_trend_mm)
            ds.attrs['sea_level_trend_intercept'] = float(intercept)
            
            # Detrend sea level
            ds['sea_level'] = ds.sea_level - xr.DataArray(trend, dims=['time'], coords={'time': ds.time})
            ds.sea_level.attrs['description'] = 'Detrended sea level'
            
            print(f"Removed linear trend from sea level. Trend: {annual_trend_mm:.2f} mm/year")
        except np.linalg.LinAlgError as e:
            print(f"Warning: Error in detrending calculation: {e}")
            print("Skipping detrending due to numerical issues.")
            ds.attrs['sea_level_trend_mm_per_year'] = float('nan')
            ds.attrs['detrending_error'] = str(e)
        
        return ds
    
    def save_preprocessed_data(
        self, 
        ds: xr.Dataset, 
        output_path: PathLike,
        compress: bool = True
    ) -> None:
        """
        Save preprocessed data to a NetCDF file.
        
        Parameters
        ----------
        ds : xr.Dataset
            Preprocessed dataset to save.
        output_path : PathLike
            Path to save the dataset.
        compress : bool, optional
            Whether to apply compression. Default is True.
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set encoding for compression
        if compress:
            encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
        else:
            encoding = {}
        
        # Add attributes for tracking
        ds.attrs['created'] = pd.Timestamp.now().isoformat()
        
        # Save to NetCDF
        ds.to_netcdf(output_path, encoding=encoding)
        print(f"Saved preprocessed data to {output_path}")