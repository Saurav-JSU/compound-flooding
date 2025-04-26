import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime
import concurrent.futures
import time
from tqdm import tqdm

# Define paths
metadata_path = 'compound_flooding/data/GESLA/usa_metadata.csv'
data_dir = 'compound_flooding/GESLA_ERA5_with_sea_level/'

# Read metadata
metadata = pd.read_csv(metadata_path)
print(f"Metadata contains information for {len(metadata)} stations")

# Get a list of all CSV files in the data directory
csv_files = glob.glob(data_dir + '*_ERA5_with_sea_level.csv')
print(f"Found {len(csv_files)} station data files")

# Define function to process a single file
def process_file(file_path):
    # Extract site code from filename
    site_code = os.path.basename(file_path).split('_ERA5_with_sea_level.csv')[0]
    
    # Get metadata for this station
    station_meta = metadata[metadata['SITE CODE'] == site_code]
    site_name = station_meta['SITE NAME'].iloc[0] if not station_meta.empty else 'Unknown'
    lat = station_meta['LATITUDE'].iloc[0] if not station_meta.empty else None
    lon = station_meta['LONGITUDE'].iloc[0] if not station_meta.empty else None
    
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    
    # Get available columns
    columns = df.columns.tolist()
    
    # Calculate basic statistics
    start_date = df['datetime'].min()
    end_date = df['datetime'].max()
    duration_years = (end_date - start_date).days / 365.25
    
    # Prepare the results dictionary
    stats = {
        'site_code': site_code,
        'site_name': site_name,
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'duration_years': duration_years,
        'total_rows': len(df),
        'available_columns': columns,
    }
    
    # Sea level analysis
    if 'sea_level' in columns:
        missing_sea_level = (df['sea_level'] == -99.9999).sum()
        pct_missing_sea_level = (missing_sea_level / len(df)) * 100
        valid_sea_level = df[df['sea_level'] != -99.9999]['sea_level']
        
        stats.update({
            'missing_sea_level': missing_sea_level,
            'pct_missing_sea_level': pct_missing_sea_level,
            'max_sea_level': valid_sea_level.max() if not valid_sea_level.empty else None,
            'min_sea_level': valid_sea_level.min() if not valid_sea_level.empty else None,
            'mean_sea_level': valid_sea_level.mean() if not valid_sea_level.empty else None,
            'p95_sea_level': valid_sea_level.quantile(0.95) if not valid_sea_level.empty else None,
            'p99_sea_level': valid_sea_level.quantile(0.99) if not valid_sea_level.empty else None,
        })
    
    # Precipitation analysis
    if 'total_precipitation' in columns:
        # Convert from m to mm (as mentioned in your PDF)
        df['total_precipitation_mm'] = df['total_precipitation'] * 1000
        
        stats.update({
            'missing_total_precip': df['total_precipitation'].isna().sum(),
            'pct_missing_total_precip': (df['total_precipitation'].isna().sum() / len(df)) * 100,
            'max_total_precip_mm': df['total_precipitation_mm'].max(),
            'mean_total_precip_mm': df['total_precipitation_mm'].mean(),
            'p95_total_precip_mm': df['total_precipitation_mm'].quantile(0.95),
            'p99_total_precip_mm': df['total_precipitation_mm'].quantile(0.99),
        })
    
    # Ground precipitation analysis (if available)
    if 'ground_precipitation' in columns:
        stats.update({
            'missing_ground_precip': df['ground_precipitation'].isna().sum(),
            'pct_missing_ground_precip': (df['ground_precipitation'].isna().sum() / len(df)) * 100,
            'max_ground_precip': df['ground_precipitation'].max(),
            'mean_ground_precip': df['ground_precipitation'].mean(),
            'p95_ground_precip': df['ground_precipitation'].quantile(0.95),
            'p99_ground_precip': df['ground_precipitation'].quantile(0.99),
        })
    
    # Wind component analysis
    if 'u_component_of_wind_10m' in columns and 'v_component_of_wind_10m' in columns:
        # Calculate wind speed
        df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
        
        stats.update({
            'max_wind_speed': df['wind_speed'].max(),
            'mean_wind_speed': df['wind_speed'].mean(),
            'p95_wind_speed': df['wind_speed'].quantile(0.95),
            'p99_wind_speed': df['wind_speed'].quantile(0.99),
        })
    
    # Analyze potential extreme events (preliminary count)
    if 'sea_level' in columns and 'total_precipitation' in columns:
        # Use 95th percentiles as preliminary thresholds
        if not valid_sea_level.empty:
            sea_level_threshold = valid_sea_level.quantile(0.95)
            precip_threshold = df['total_precipitation_mm'].quantile(0.95)
            
            # Count exceedances
            sea_level_exceedances = (valid_sea_level > sea_level_threshold).sum()
            precip_exceedances = (df['total_precipitation_mm'] > precip_threshold).sum()
            
            # Count joint exceedances (same hour)
            joint_exceedances = ((df['sea_level'] > sea_level_threshold) & 
                                (df['total_precipitation_mm'] > precip_threshold)).sum()
            
            stats.update({
                'sea_level_exceedances': sea_level_exceedances,
                'precip_exceedances': precip_exceedances,
                'joint_exceedances': joint_exceedances,
                'CPR': (joint_exceedances * len(df)) / (sea_level_exceedances * precip_exceedances) 
                       if (sea_level_exceedances * precip_exceedances) > 0 else None
            })
    
    return stats

# Use parallel processing to analyze all files
print(f"\nAnalyzing {len(csv_files)} files using parallel processing...")
start_time = time.time()

# Use a ProcessPoolExecutor to leverage all available cores
with concurrent.futures.ProcessPoolExecutor(max_workers=80) as executor:  # Use 96 cores
    # Map the process_file function to all CSV files and collect results
    results = list(tqdm(executor.map(process_file, csv_files), total=len(csv_files)))

end_time = time.time()
print(f"Analysis completed in {end_time - start_time:.2f} seconds")

# Convert results to dataframe
summary_df = pd.DataFrame(results)

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total number of stations analyzed: {len(summary_df)}")
print(f"Average data duration: {summary_df['duration_years'].mean():.2f} years")
print(f"Average percentage of missing sea level data: {summary_df['pct_missing_sea_level'].mean():.2f}%")

# Analyze column availability
available_columns = {}
for index, row in summary_df.iterrows():
    for col in row['available_columns']:
        available_columns[col] = available_columns.get(col, 0) + 1

print("\nColumn availability across all files:")
for col, count in available_columns.items():
    print(f"{col}: Available in {count}/{len(summary_df)} files ({count/len(summary_df)*100:.1f}%)")

# Additional summary statistics for Tier 1 planning
print("\nKey statistics for Tier 1 analysis planning:")
if 'p99_sea_level' in summary_df.columns:
    print(f"Median 99th percentile sea level: {summary_df['p99_sea_level'].median():.4f} m")
if 'p99_total_precip_mm' in summary_df.columns:
    print(f"Median 99th percentile precipitation: {summary_df['p99_total_precip_mm'].median():.2f} mm")
if 'joint_exceedances' in summary_df.columns:
    print(f"Average number of joint exceedances per station: {summary_df['joint_exceedances'].mean():.2f}")
if 'CPR' in summary_df.columns:
    print(f"Average Conditional Probability Ratio (CPR): {summary_df['CPR'].mean():.2f}")

# Save detailed summary to a CSV file
summary_df.drop(columns=['available_columns'], errors='ignore').to_csv('station_data_summary.csv', index=False)
print("Detailed summary saved to station_data_summary.csv")

# Generate a more compact summary for quick reference
compact_summary = summary_df[['site_code', 'site_name', 'start_date', 'end_date', 
                             'duration_years', 'pct_missing_sea_level', 
                             'p99_sea_level', 'p99_total_precip_mm', 
                             'joint_exceedances', 'CPR']].copy()

compact_summary.to_csv('tier1_planning_summary.csv', index=False)
print("Compact summary for Tier 1 planning saved to tier1_planning_summary.csv")