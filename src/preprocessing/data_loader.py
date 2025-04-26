# compound_flooding/preprocessing/data_loader.py
import pandas as pd
import os
import glob
import numpy as np
from datetime import datetime

def load_metadata(metadata_path):
    """
    Load station metadata.
    
    Args:
        metadata_path: Path to the metadata CSV file
        
    Returns:
        DataFrame containing station metadata
    """
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata for {len(metadata)} stations")
    return metadata

def get_station_files(data_dir, pattern='*_ERA5_with_sea_level.csv'):
    """
    Get a list of all station data files.
    
    Args:
        data_dir: Directory containing station CSV files
        pattern: Glob pattern to match station files
        
    Returns:
        List of file paths
    """
    file_paths = glob.glob(os.path.join(data_dir, pattern))
    print(f"Found {len(file_paths)} station data files")
    return file_paths

def extract_site_code(file_path):
    """Extract site code from file path."""
    return os.path.basename(file_path).split('_ERA5_with_sea_level.csv')[0]

def load_station_data(file_path, parse_dates=True):
    """
    Load a single station's data and process precipitation data correctly.
    
    Args:
        file_path: Path to the station CSV file
        parse_dates: Whether to parse datetime column
        
    Returns:
        DataFrame containing station data with properly processed precipitation
    """
    try:
        if parse_dates:
            df = pd.read_csv(file_path, parse_dates=['datetime'])
        else:
            df = pd.read_csv(file_path)
        
        # Process precipitation data:
        # 1. Convert ERA5 total_precipitation from m to mm
        if 'total_precipitation' in df.columns:
            df['total_precipitation_mm'] = df['total_precipitation'] * 1000
        
        # 2. Create a consolidated precipitation column using ground data when available
        if 'ground_precipitation' in df.columns and 'total_precipitation_mm' in df.columns:
            # Check if ground_precipitation appears to already be in mm (common case)
            # We can detect this by comparing max values - if ground is ~1000x larger than 
            # raw ERA5, it's likely already in mm
            if df['ground_precipitation'].max() > df['total_precipitation'].max() * 100:
                # Ground data likely already in mm
                df['precipitation_mm'] = df['ground_precipitation'].copy()
            else:
                # Ground data might be in meters too, convert to mm
                df['ground_precipitation_mm'] = df['ground_precipitation'] * 1000
                df['precipitation_mm'] = df['ground_precipitation_mm'].copy()
                
            # Fill missing ground data with ERA5 data
            mask = df['precipitation_mm'].isna() | (df['precipitation_mm'] == 0)
            df.loc[mask, 'precipitation_mm'] = df.loc[mask, 'total_precipitation_mm']
        elif 'total_precipitation_mm' in df.columns:
            # Only ERA5 data available
            df['precipitation_mm'] = df['total_precipitation_mm'].copy()
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_station_metadata(metadata_df, site_code):
    """
    Get metadata for a specific station.
    
    Args:
        metadata_df: DataFrame containing metadata
        site_code: Site code to look up
        
    Returns:
        Dict with metadata or None if not found
    """
    station_meta = metadata_df[metadata_df['SITE CODE'] == site_code]
    
    if station_meta.empty:
        return None
        
    # Convert to dictionary for easier access
    meta_dict = {
        'site_code': site_code,
        'site_name': station_meta['SITE NAME'].iloc[0] if 'SITE NAME' in station_meta.columns else 'Unknown',
        'latitude': station_meta['LATITUDE'].iloc[0] if 'LATITUDE' in station_meta.columns else None,
        'longitude': station_meta['LONGITUDE'].iloc[0] if 'LONGITUDE' in station_meta.columns else None,
        'country': station_meta['COUNTRY'].iloc[0] if 'COUNTRY' in station_meta.columns else 'Unknown',
        'null_value': station_meta['NULL VALUE'].iloc[0] if 'NULL VALUE' in station_meta.columns else -99.9999
    }
    
    return meta_dict

def load_station_with_metadata(file_path, metadata_df):
    """
    Load a station's data and its metadata in one function.
    
    Args:
        file_path: Path to the station CSV file
        metadata_df: DataFrame containing metadata
        
    Returns:
        Tuple of (DataFrame with station data, Dict with metadata)
    """
    site_code = extract_site_code(file_path)
    station_meta = get_station_metadata(metadata_df, site_code)
    
    if station_meta is None:
        print(f"No metadata found for station {site_code}")
        return None, None
    
    df = load_station_data(file_path)
    
    return df, station_meta