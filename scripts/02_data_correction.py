# scripts/02_data_correction_fixed.py
import os
import sys
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import traceback

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'data_correction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('data_correction')

# Data correction functions - with better error handling
def correct_sea_level_outliers(df, metadata, quality_info):
    """Correct unrealistic sea level values safely."""
    null_value = metadata.get('null_value', -99.9999)
    
    # Get valid sea level data
    valid_mask = df['sea_level'] != null_value
    valid_sl = df.loc[valid_mask, 'sea_level']
    
    if valid_sl.empty:
        logger.warning(f"No valid sea level data for station {metadata['site_code']}")
        return df, quality_info
    
    # Statistical outlier detection
    mean_sl = valid_sl.mean()
    std_sl = valid_sl.std()
    
    # Determine if we have outlier issues
    if quality_info.get('sl_quality') in ['unrealistic_values']:
        # Check for potential unit conversion errors
        median_sl = valid_sl.median()
        max_sl = valid_sl.max()
        
        if max_sl > 50 and max_sl / median_sl > 50:
            # Likely unit error - correct extreme values
            logger.info(f"Likely unit error detected for {metadata['site_code']}: max={max_sl}, median={median_sl}")
            
            # Identify extreme values
            extreme_mask = valid_mask & (df['sea_level'] > mean_sl + 5 * std_sl)
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                # Correct by dividing by 100 or 10
                if max_sl > 100:
                    df.loc[extreme_mask, 'sea_level'] = df.loc[extreme_mask, 'sea_level'] / 100
                else:
                    df.loc[extreme_mask, 'sea_level'] = df.loc[extreme_mask, 'sea_level'] / 10
                
                logger.info(f"Corrected {extreme_count} extreme values for {metadata['site_code']}")
                
                # Update quality info
                quality_info['sl_extreme_correction_count'] = extreme_count
                quality_info['sl_extreme_correction_method'] = 'unit_conversion'
        else:
            # Cap at a reasonable maximum
            reasonable_max = 10.0
            
            # For Great Lakes, adjust if needed
            if metadata.get('latitude') is not None:
                # Great Lakes region
                if metadata['latitude'] > 41 and metadata['latitude'] < 49 and metadata['longitude'] > -93 and metadata['longitude'] < -76:
                    reasonable_max = 5.0
            
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
    """Fill short gaps in sea level data safely."""
    null_value = metadata.get('null_value', -99.9999)
    
    # Check if missing data is an issue
    if quality_info.get('sl_quality') in ['minor_missing', 'significant_missing'] and quality_info.get('sl_pct_missing', 0) < 25:
        # Create a mask for missing values
        missing_mask = df['sea_level'] == null_value
        missing_count = missing_mask.sum()
        
        if missing_count > 0 and missing_count < len(df):  # Ensure we don't have all missing
            # Make a copy of the sea level data
            sl_filled = df['sea_level'].copy()
            
            # Identify short gaps (1-2 consecutive hours)
            run_id = (missing_mask != missing_mask.shift()).cumsum()
            
            # Find runs of missing values
            run_stats = missing_mask.groupby(run_id).agg(['sum', 'count'])
            short_gaps = run_stats[(run_stats['sum'] > 0) & (run_stats['sum'] <= 2)].index.tolist()
            
            filled_count = 0
            
            # For each short run, use linear interpolation
            for run in short_gaps:
                run_mask = (run_id == run) & missing_mask
                
                if run_mask.sum() <= 2:  # Double-check it's a short gap
                    # Get position of this run
                    run_positions = np.where(run_mask)[0]
                    
                    if len(run_positions) == 0:
                        continue  # Skip if somehow empty
                    
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
    """Correct precipitation data issues safely."""
    # Check if needed columns exist
    if 'total_precipitation' not in df.columns:
        logger.warning(f"No precipitation data for station {metadata['site_code']}")
        return df, quality_info
    
    # Ensure we have the consolidated precipitation column
    if 'precipitation_mm' not in df.columns:
        # Create consolidated precipitation column
        # Convert from m to mm
        df['total_precipitation_mm'] = df['total_precipitation'] * 1000
        df['precipitation_mm'] = df['total_precipitation_mm'].copy()
        
        # If ground data exists, use it where available
        if 'ground_precipitation' in df.columns:
            # Check if ground data has any valid values
            if df['ground_precipitation'].notna().sum() > 0:
                # Check if ground_precipitation appears to be in mm
                max_ground = df['ground_precipitation'].max()
                max_era5 = df['total_precipitation'].max()
                
                # If ground data seems to be in mm already (and both have some non-zero values)
                if max_ground > 0 and max_era5 > 0 and max_ground > max_era5 * 10:
                    # Use ground data where available and non-zero
                    use_ground_mask = (~df['ground_precipitation'].isna()) & (df['ground_precipitation'] > 0)
                    df.loc[use_ground_mask, 'precipitation_mm'] = df.loc[use_ground_mask, 'ground_precipitation']
                    
                    quality_info['ground_precip_used_count'] = use_ground_mask.sum()
                    quality_info['ground_precip_used_pct'] = use_ground_mask.sum() / len(df) * 100
                elif max_ground > 0:  # If ground data exists but might be in meters
                    # Need to convert ground data to mm first
                    df['ground_precipitation_mm'] = df['ground_precipitation'] * 1000
                    
                    # Use converted ground data where available and non-zero
                    use_ground_mask = (~df['ground_precipitation_mm'].isna()) & (df['ground_precipitation_mm'] > 0)
                    df.loc[use_ground_mask, 'precipitation_mm'] = df.loc[use_ground_mask, 'ground_precipitation_mm']
                    
                    quality_info['ground_precip_used_count'] = use_ground_mask.sum()
                    quality_info['ground_precip_used_pct'] = use_ground_mask.sum() / len(df) * 100
    
    # Check for unrealistic precipitation values
    if 'precipitation_mm' in df.columns:
        extreme_precip_mask = df['precipitation_mm'] > 500
        if extreme_precip_mask.sum() > 0:
            # Cap at 500mm per hour
            df.loc[extreme_precip_mask, 'precipitation_mm'] = 500
            
            quality_info['precip_extreme_correction_count'] = extreme_precip_mask.sum()
            quality_info['precip_extreme_correction_method'] = 'capping'
            
            logger.info(f"Capped {extreme_precip_mask.sum()} extreme precipitation values for {metadata['site_code']}")
    
    return df, quality_info

def analyze_station_quality(df, metadata):
    """Reanalyze station quality after corrections."""
    null_value = metadata.get('null_value', -99.9999)
    
    # Basic stats
    quality_info = {
        'site_code': metadata['site_code'],
        'site_name': metadata.get('site_name', 'Unknown'),
        'latitude': metadata.get('latitude', None),
        'longitude': metadata.get('longitude', None)
    }
    
    # Get valid sea level data
    valid_sl = df[df['sea_level'] != null_value]['sea_level']
    
    # Sea level stats
    if valid_sl.empty:
        logger.warning(f"No valid sea level data for station {metadata['site_code']}")
        quality_info['sl_quality'] = 'unusable'
        quality_info['quality_tier'] = 'D'
        return quality_info
    
    quality_info['sl_valid_count'] = len(valid_sl)
    quality_info['sl_missing_count'] = (df['sea_level'] == null_value).sum()
    quality_info['sl_pct_missing'] = 100 * quality_info['sl_missing_count'] / len(df)
    quality_info['sl_min'] = valid_sl.min()
    quality_info['sl_max'] = valid_sl.max()
    quality_info['sl_p95'] = valid_sl.quantile(0.95)
    quality_info['sl_p99'] = valid_sl.quantile(0.99)
    
    # Precipitation stats
    if 'precipitation_mm' in df.columns:
        quality_info['precip_missing_count'] = df['precipitation_mm'].isna().sum()
        quality_info['precip_pct_missing'] = 100 * quality_info['precip_missing_count'] / len(df)
        quality_info['precip_max'] = df['precipitation_mm'].max()
        quality_info['precip_p95'] = df['precipitation_mm'].quantile(0.95)
        quality_info['precip_p99'] = df['precipitation_mm'].quantile(0.99)
    
    # Joint exceedance analysis
    sl_threshold = quality_info['sl_p95']
    precip_threshold = quality_info.get('precip_p95', None)
    
    if precip_threshold is not None:
        sl_exceed = (df['sea_level'] != null_value) & (df['sea_level'] > sl_threshold)
        precip_exceed = df['precipitation_mm'] > precip_threshold
        
        quality_info['sl_exceedances'] = sl_exceed.sum()
        quality_info['precip_exceedances'] = precip_exceed.sum()
        quality_info['joint_exceedances'] = (sl_exceed & precip_exceed).sum()
        
        # Calculate CPR
        if quality_info['sl_exceedances'] > 0 and quality_info['precip_exceedances'] > 0:
            p_sl = quality_info['sl_exceedances'] / len(df)
            p_precip = quality_info['precip_exceedances'] / len(df)
            p_joint = quality_info['joint_exceedances'] / len(df)
            
            quality_info['CPR'] = p_joint / (p_sl * p_precip) if (p_sl * p_precip) > 0 else None
    
    # Determine quality tier
    if quality_info.get('sl_p99', 0) > 50:
        quality_info['sl_quality'] = 'unrealistic_values'
        quality_info['quality_tier'] = 'D'
    elif quality_info.get('sl_pct_missing', 100) > 25:
        quality_info['sl_quality'] = 'excessive_missing'
        quality_info['quality_tier'] = 'C'
    elif quality_info.get('joint_exceedances', 0) == 0:
        quality_info['joint_quality'] = 'no_joint_events'
        quality_info['quality_tier'] = 'D'
    elif quality_info.get('sl_pct_missing', 0) > 10:
        quality_info['sl_quality'] = 'significant_missing'
        quality_info['quality_tier'] = 'B'
    elif quality_info.get('sl_pct_missing', 0) > 5:
        quality_info['sl_quality'] = 'minor_missing'
        quality_info['quality_tier'] = 'B'
    elif quality_info.get('joint_exceedances', 0) < 50:
        quality_info['joint_quality'] = 'few_joint_events'
        quality_info['quality_tier'] = 'B'
    else:
        quality_info['sl_quality'] = 'excellent'
        quality_info['quality_tier'] = 'A'
    
    return quality_info

def correct_station_data(df, metadata, quality_info):
    """Apply all necessary corrections to station data."""
    # Make a copy to avoid modifying original
    df_corrected = df.copy()
    quality_updated = quality_info.copy()
    
    try:
        # Apply corrections in sequence
        df_corrected, quality_updated = correct_sea_level_outliers(df_corrected, metadata, quality_updated)
        df_corrected, quality_updated = fill_missing_sea_level(df_corrected, metadata, quality_updated)
        df_corrected, quality_updated = correct_precipitation_data(df_corrected, metadata, quality_updated)
        
        # Recalculate quality metrics after corrections
        quality_updated['correction_applied'] = True
        
        return df_corrected, quality_updated
    except Exception as e:
        logger.error(f"Error during correction for {metadata['site_code']}: {e}")
        logger.debug(traceback.format_exc())
        return df, quality_info

def process_station_correction(file_info):
    """Process a single station for correction."""
    file_path, quality_info = file_info
    site_code = quality_info['site_code']
    
    # Skip Tier A stations (already good) and Tier D stations (too problematic)
    if quality_info['quality_tier'] not in ['B', 'C']:
        return {
            'site_code': site_code,
            'correction_status': 'skipped',
            'quality_tier': quality_info['quality_tier'],
            'reason': f"Tier {quality_info['quality_tier']} station - {'no correction needed' if quality_info['quality_tier'] == 'A' else 'too problematic'}"
        }
    
    try:
        # Load metadata
        metadata = {
            'site_code': site_code,
            'site_name': quality_info.get('site_name', 'Unknown'),
            'latitude': quality_info.get('latitude', None),
            'longitude': quality_info.get('longitude', None),
            'null_value': quality_info.get('null_value', -99.9999)
        }
        
        # Load station data
        df = pd.read_csv(file_path, parse_dates=['datetime'])
        
        # Check if station has any valid sea level data
        valid_sl = df[df['sea_level'] != metadata['null_value']]['sea_level']
        if valid_sl.empty:
            logger.warning(f"Station {site_code} has no valid sea level data")
            return {
                'site_code': site_code,
                'correction_status': 'failed',
                'error': "No valid sea level data"
            }
        
        # Apply corrections
        df_corrected, quality_updated = correct_station_data(df, metadata, quality_info)
        
        # Analyze quality of corrected data
        reanalyzed_quality = analyze_station_quality(df_corrected, metadata)
        
        # Save corrected data
        output_dir = os.path.join('data', 'processed', 'corrected')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{site_code}_corrected.csv")
        df_corrected.to_csv(output_file, index=False)
        
        # Prepare result
        result = {
            'site_code': site_code,
            'correction_status': 'success',
            'original_quality_tier': quality_info['quality_tier'],
            'new_quality_tier': reanalyzed_quality['quality_tier'],
            'correction_methods': []
        }
        
        # Log specific corrections
        if quality_updated.get('sl_extreme_correction_count', 0) > 0:
            result['correction_methods'].append(f"sea_level_outliers_{quality_updated['sl_extreme_correction_method']}")
            result['sl_extreme_correction_count'] = quality_updated['sl_extreme_correction_count']
        
        if quality_updated.get('sl_filled_count', 0) > 0:
            result['correction_methods'].append(f"missing_sea_level_{quality_updated['sl_fill_method']}")
            result['sl_filled_count'] = quality_updated['sl_filled_count']
        
        if quality_updated.get('precip_extreme_correction_count', 0) > 0:
            result['correction_methods'].append(f"precipitation_outliers_{quality_updated['precip_extreme_correction_method']}")
            result['precip_extreme_correction_count'] = quality_updated['precip_extreme_correction_count']
        
        if len(result['correction_methods']) == 0:
            result['correction_methods'].append("only_minor_adjustments")
        
        return result
    
    except Exception as e:
        logger.error(f"Error correcting station {site_code}: {e}")
        logger.debug(traceback.format_exc())
        return {
            'site_code': site_code,
            'correction_status': 'failed',
            'error': str(e)
        }

def process_in_parallel(items, process_func, max_workers=None, desc="Processing"):
    """Process a list of items in parallel."""
    start_time = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Process results as they complete with a progress bar
        futures = {executor.submit(process_func, item): item for item in items}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
    
    end_time = time.time()
    logger.info(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    
    return results

def main():
    """Main function to run the data correction process."""
    logger.info("Starting data correction process")
    
    # Load quality assessment results
    quality_file = os.path.join('data', 'processed', 'station_quality_assessment.csv')
    quality_df = pd.read_csv(quality_file)
    
    # Get station files for Tier B and C stations
    data_dir = os.path.join('compound_flooding', 'GESLA_ERA5_with_sea_level')
    
    # Filter for Tier B and C stations
    tier_bc_stations = quality_df[quality_df['quality_tier'].isin(['B', 'C'])]
    logger.info(f"Found {len(tier_bc_stations)} stations in Tiers B and C that need correction")
    
    # Prepare file info list
    file_info_list = []
    for _, row in tier_bc_stations.iterrows():
        site_code = row['site_code']
        file_path = os.path.join(data_dir, f"{site_code}_ERA5_with_sea_level.csv")
        
        if os.path.exists(file_path):
            file_info_list.append((file_path, row.to_dict()))
    
    logger.info(f"Processing {len(file_info_list)} stations for correction")
    
    # Process corrections in parallel
    start_time = time.time()
    results = process_in_parallel(
        file_info_list,
        process_station_correction,
        max_workers=96,  # Use all available cores
        desc="Correcting station data"
    )
    
    end_time = time.time()
    logger.info(f"Correction process completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    results_df = pd.DataFrame(results)
    
    success_count = (results_df['correction_status'] == 'success').sum()
    failed_count = (results_df['correction_status'] == 'failed').sum()
    skipped_count = (results_df['correction_status'] == 'skipped').sum()
    
    logger.info(f"Correction results: {success_count} successful, {failed_count} failed, {skipped_count} skipped")
    
    # Check for quality tier improvements
    if 'original_quality_tier' in results_df.columns and 'new_quality_tier' in results_df.columns:
        improved_mask = results_df['new_quality_tier'] < results_df['original_quality_tier']
        improved_stations = results_df[improved_mask]
        same_tier_stations = results_df[results_df['original_quality_tier'] == results_df['new_quality_tier']]
        
        logger.info(f"Quality improvements: {len(improved_stations)} stations improved, {len(same_tier_stations)} remained in same tier")
    
    # Save results
    results_df.to_csv(os.path.join('data', 'processed', 'correction_results.csv'), index=False)
    logger.info(f"Correction results saved to data/processed/correction_results.csv")
    
    # Create consolidated dataset with all usable stations
    logger.info("Creating consolidated dataset with all usable stations")
    
    # Load Tier A stations (already good quality)
    tier_a_stations = quality_df[quality_df['quality_tier'] == 'A']
    tier_a_codes = tier_a_stations['site_code'].tolist()
    
    # Get successfully corrected stations that are now good quality
    corrected_good_stations = results_df[
        (results_df['correction_status'] == 'success') & 
        (results_df['new_quality_tier'].isin(['A', 'B']))
    ]
    corrected_good_codes = corrected_good_stations['site_code'].tolist()
    
    # Combined list of usable station codes
    usable_station_codes = tier_a_codes + corrected_good_codes
    logger.info(f"Consolidated dataset will include {len(usable_station_codes)} high-quality stations")
    
    # Save the list of stations for Tier 1 analysis
    usable_stations_df = pd.DataFrame({'site_code': usable_station_codes})
    usable_stations_df.to_csv(os.path.join('data', 'processed', 'usable_stations_for_tier1.csv'), index=False)
    
    logger.info("Data correction process completed")

if __name__ == "__main__":
    main()