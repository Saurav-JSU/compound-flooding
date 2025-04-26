# compound_flooding/preprocessing/data_correction.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('data_correction')

def correct_sea_level_outliers(df, metadata, quality_info):
    """
    Correct unrealistic sea level values based on physical limits.
    
    Strategies:
    1. Remove extreme statistical outliers (>5σ from mean)
    2. Cap values at physically plausible regional maxima
    3. Fix potential unit/datum conversion errors
    """
    null_value = metadata.get('null_value', -99.9999)
    
    # Get valid sea level data
    valid_mask = df['sea_level'] != null_value
    valid_sl = df.loc[valid_mask, 'sea_level']
    
    if valid_sl.empty:
        return df, quality_info
    
    # 1. Statistical outlier detection
    mean_sl = valid_sl.mean()
    std_sl = valid_sl.std()
    
    # Determine if we have outlier issues
    if quality_info.get('sl_quality') in ['unrealistic_values']:
        # Check for potential unit conversion errors (common issue)
        # If max value is approximately 100x median, possible unit error
        median_sl = valid_sl.median()
        max_sl = valid_sl.max()
        
        if max_sl > 50 and max_sl / median_sl > 50:
            # Likely unit error - correct extreme values
            logger.info(f"Likely unit error detected for {metadata['site_code']}: max={max_sl}, median={median_sl}")
            
            # Identify extreme values (e.g., >5σ from mean or above a reasonable threshold)
            extreme_mask = valid_mask & (df['sea_level'] > mean_sl + 5 * std_sl)
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                # Correct by dividing by 100 or 10 (common conversion factors)
                if max_sl > 100:
                    df.loc[extreme_mask, 'sea_level'] = df.loc[extreme_mask, 'sea_level'] / 100
                else:
                    df.loc[extreme_mask, 'sea_level'] = df.loc[extreme_mask, 'sea_level'] / 10
                
                logger.info(f"Corrected {extreme_count} extreme values for {metadata['site_code']}")
                
                # Update quality info
                quality_info['sl_extreme_correction_count'] = extreme_count
                quality_info['sl_extreme_correction_method'] = 'unit_conversion'
        else:
            # Cap at a reasonable maximum (e.g., 10m for most regions)
            reasonable_max = 10.0  # 10 meters is a reasonable max for most regions
            
            # For Great Lakes or special regions, adjust if needed
            if metadata.get('latitude') is not None:
                # Great Lakes region
                if metadata['latitude'] > 41 and metadata['latitude'] < 49 and metadata['longitude'] > -93 and metadata['longitude'] < -76:
                    reasonable_max = 5.0  # Great Lakes have smaller ranges
            
            extreme_mask = valid_mask & (df['sea_level'] > reasonable_max)
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                # Cap the values
                df.loc[extreme_mask, 'sea_level'] = reasonable_max
                
                logger.info(f"Capped {extreme_count} sea level values at {reasonable_max}m for {metadata['site_code']}")
                
                # Update quality info
                quality_info['sl_extreme_correction_count'] = extreme_count
                quality_info['sl_extreme_correction_method'] = 'capping'
    
    return df, quality_info

def fill_missing_sea_level(df, metadata, quality_info):
    """
    Fill short gaps in sea level data using appropriate methods.
    
    Strategies:
    1. Linear interpolation for very short gaps (1-2 hours)
    2. Tidal prediction for medium gaps if frequency components are known
    3. Document longer gaps properly
    """
    null_value = metadata.get('null_value', -99.9999)
    
    # Check if missing data is an issue
    if quality_info.get('sl_quality') in ['minor_missing', 'significant_missing'] and quality_info.get('sl_pct_missing', 0) < 25:
        # Create a mask for missing values
        missing_mask = df['sea_level'] == null_value
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            # Make a copy of the sea level data
            sl_filled = df['sea_level'].copy()
            
            # Identify short gaps (1-2 consecutive hours)
            # This requires finding runs of missing values
            sl_filled_series = pd.Series(sl_filled)
            
            # Create groups of consecutive values
            run_id = (missing_mask != missing_mask.shift()).cumsum()
            runs = missing_mask.groupby(run_id).count()
            
            # Find runs of missing values that are short (1-2 hours)
            short_runs = run_id[missing_mask].value_counts()
            short_runs = short_runs[short_runs <= 2]
            
            filled_count = 0
            
            # For each short run, use linear interpolation
            for run in short_runs.index:
                run_mask = (run_id == run) & missing_mask
                
                if run_mask.sum() <= 2:  # Double-check it's a short gap
                    # Get position of this run
                    run_positions = np.where(run_mask)[0]
                    
                    # Only fill if not at the edges of the dataset
                    if min(run_positions) > 0 and max(run_positions) < len(df) - 1:
                        # Linear interpolation
                        start_idx = min(run_positions) - 1
                        end_idx = max(run_positions) + 1
                        
                        start_val = df.iloc[start_idx]['sea_level']
                        end_val = df.iloc[end_idx]['sea_level']
                        
                        # Only interpolate if both endpoints are valid
                        if start_val != null_value and end_val != null_value:
                            # Calculate interpolated values
                            gap_length = end_idx - start_idx - 1
                            step = (end_val - start_val) / (gap_length + 1)
                            
                            for i, pos in enumerate(range(start_idx + 1, end_idx)):
                                sl_filled.iloc[pos] = start_val + step * (i + 1)
                                filled_count += 1
            
            if filled_count > 0:
                # Update the DataFrame with filled values
                df['sea_level'] = sl_filled
                
                # Update quality info
                quality_info['sl_filled_count'] = filled_count
                quality_info['sl_pct_missing_after_fill'] = (missing_count - filled_count) / len(df) * 100
                quality_info['sl_fill_method'] = 'linear_interpolation'
                
                logger.info(f"Filled {filled_count} missing sea level values for {metadata['site_code']}")
    
    return df, quality_info

def correct_precipitation_data(df, metadata, quality_info):
    """
    Correct precipitation data issues and merge sources properly.
    
    Strategies:
    1. Use ground precipitation where available, ERA5 as backup
    2. Correct unit inconsistencies
    3. Check for physically implausible values
    """
    # Ensure we have the consolidated precipitation column
    if 'precipitation_mm' not in df.columns:
        # Create consolidated precipitation column
        if 'ground_precipitation' in df.columns and 'total_precipitation_mm' in df.columns:
            # Start with ERA5 (already converted to mm)
            df['precipitation_mm'] = df['total_precipitation_mm'].copy()
            
            # Check if ground_precipitation appears to be in mm (larger than ERA5)
            max_ground = df['ground_precipitation'].max()
            max_era5 = df['total_precipitation'].max()
            
            # If ground data seems to be in mm already
            if max_ground > 0 and max_era5 > 0 and max_ground > max_era5 * 10:
                # Use ground data where available and non-zero
                use_ground_mask = (~df['ground_precipitation'].isna()) & (df['ground_precipitation'] > 0)
                df.loc[use_ground_mask, 'precipitation_mm'] = df.loc[use_ground_mask, 'ground_precipitation']
                
                quality_info['ground_precip_used_count'] = use_ground_mask.sum()
                quality_info['ground_precip_used_pct'] = use_ground_mask.sum() / len(df) * 100
            else:
                # Need to convert ground data to mm first
                df['ground_precipitation_mm'] = df['ground_precipitation'] * 1000
                
                # Use converted ground data where available and non-zero
                use_ground_mask = (~df['ground_precipitation_mm'].isna()) & (df['ground_precipitation_mm'] > 0)
                df.loc[use_ground_mask, 'precipitation_mm'] = df.loc[use_ground_mask, 'ground_precipitation_mm']
                
                quality_info['ground_precip_used_count'] = use_ground_mask.sum()
                quality_info['ground_precip_used_pct'] = use_ground_mask.sum() / len(df) * 100
        elif 'total_precipitation_mm' in df.columns:
            # Only ERA5 data available
            df['precipitation_mm'] = df['total_precipitation_mm'].copy()
    
    # Check for unrealistic precipitation values (>500mm in an hour is unlikely)
    extreme_precip_mask = df['precipitation_mm'] > 500
    if extreme_precip_mask.sum() > 0:
        # Cap at 500mm per hour
        df.loc[extreme_precip_mask, 'precipitation_mm'] = 500
        
        quality_info['precip_extreme_correction_count'] = extreme_precip_mask.sum()
        quality_info['precip_extreme_correction_method'] = 'capping'
        
        logger.info(f"Capped {extreme_precip_mask.sum()} extreme precipitation values for {metadata['site_code']}")
    
    return df, quality_info

def correct_station_data(df, metadata, quality_info):
    """
    Apply all necessary corrections to station data.
    
    Args:
        df: DataFrame with station data
        metadata: Dict with station metadata
        quality_info: Dict with quality assessment results
        
    Returns:
        Corrected DataFrame and updated quality info
    """
    # Make a copy to avoid modifying original
    df_corrected = df.copy()
    quality_updated = quality_info.copy()
    
    # Apply corrections in sequence
    df_corrected, quality_updated = correct_sea_level_outliers(df_corrected, metadata, quality_updated)
    df_corrected, quality_updated = fill_missing_sea_level(df_corrected, metadata, quality_updated)
    df_corrected, quality_updated = correct_precipitation_data(df_corrected, metadata, quality_updated)
    
    # Recalculate quality metrics after corrections
    quality_updated['correction_applied'] = True
    
    return df_corrected, quality_updated