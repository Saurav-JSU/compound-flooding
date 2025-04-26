# src/preprocessing/quality_control.py
import pandas as pd
import numpy as np
from datetime import datetime

def check_sea_level_range(df, null_value=-99.9999, max_plausible=50.0):
    """
    Check for unrealistic sea level values.
    
    Args:
        df: Station data DataFrame
        null_value: Value used for missing data
        max_plausible: Maximum plausible sea level value in meters
        
    Returns:
        Dict with sea level statistics and quality flags
    """
    # Get valid sea level data
    valid_sl = df[df['sea_level'] != null_value]['sea_level']
    
    if valid_sl.empty:
        return {
            'sl_valid_count': 0,
            'sl_missing_count': len(df),
            'sl_pct_missing': 100.0,
            'sl_min': None,
            'sl_max': None,
            'sl_p95': None,
            'sl_p99': None,
            'sl_extreme_values': 0,
            'sl_quality': 'unusable',
            'sl_quality_score': 0
        }
    
    # Calculate statistics
    sl_stats = {
        'sl_valid_count': len(valid_sl),
        'sl_missing_count': (df['sea_level'] == null_value).sum(),
        'sl_pct_missing': 100 * (df['sea_level'] == null_value).sum() / len(df),
        'sl_min': valid_sl.min(),
        'sl_max': valid_sl.max(),
        'sl_p95': valid_sl.quantile(0.95),
        'sl_p99': valid_sl.quantile(0.99),
        'sl_extreme_values': (valid_sl > max_plausible).sum()
    }
    
    # Determine quality based on range and missing data
    if sl_stats['sl_p99'] > max_plausible:
        sl_stats['sl_quality'] = 'unrealistic_values'
        sl_stats['sl_quality_score'] = 1  # Very poor
    elif sl_stats['sl_pct_missing'] > 25:
        sl_stats['sl_quality'] = 'excessive_missing'
        sl_stats['sl_quality_score'] = 2  # Poor
    elif sl_stats['sl_pct_missing'] > 10:
        sl_stats['sl_quality'] = 'significant_missing'
        sl_stats['sl_quality_score'] = 3  # Fair
    elif sl_stats['sl_pct_missing'] > 5:
        sl_stats['sl_quality'] = 'minor_missing'
        sl_stats['sl_quality_score'] = 4  # Good
    else:
        sl_stats['sl_quality'] = 'excellent'
        sl_stats['sl_quality_score'] = 5  # Excellent
        
    return sl_stats

def check_precipitation_data(df):
    """
    Check precipitation data quality using the consolidated column.
    
    Args:
        df: Station data DataFrame
        
    Returns:
        Dict with precipitation statistics and quality flags
    """
    # Ensure we have the consolidated precipitation column
    if 'precipitation_mm' not in df.columns:
        if 'total_precipitation_mm' in df.columns:
            df['precipitation_mm'] = df['total_precipitation_mm'].copy()
            
            # If ground data exists, use it where available
            if 'ground_precipitation' in df.columns:
                # Check if ground_precipitation appears to be in mm
                if df['ground_precipitation'].max() > df['total_precipitation'].max() * 100:
                    ground_mask = ~df['ground_precipitation'].isna() & (df['ground_precipitation'] > 0)
                    df.loc[ground_mask, 'precipitation_mm'] = df.loc[ground_mask, 'ground_precipitation']
                else:
                    # Convert ground data to mm
                    df['ground_precipitation_mm'] = df['ground_precipitation'] * 1000
                    ground_mask = ~df['ground_precipitation_mm'].isna() & (df['ground_precipitation_mm'] > 0)
                    df.loc[ground_mask, 'precipitation_mm'] = df.loc[ground_mask, 'ground_precipitation_mm']
        
    # Calculate statistics on the consolidated column
    precip_stats = {
        'precip_missing_count': df['precipitation_mm'].isna().sum(),
        'precip_pct_missing': 100 * df['precipitation_mm'].isna().sum() / len(df),
        'precip_max': df['precipitation_mm'].max(),
        'precip_p95': df['precipitation_mm'].quantile(0.95),
        'precip_p99': df['precipitation_mm'].quantile(0.99)
    }
    
    # Add source information
    if 'ground_precipitation' in df.columns:
        ground_pct = 100 * (~df['ground_precipitation'].isna() & (df['ground_precipitation'] > 0)).sum() / len(df)
        precip_stats['ground_precip_coverage_pct'] = ground_pct
    
    # Determine quality based on missing data
    if precip_stats['precip_pct_missing'] > 25:
        precip_stats['precip_quality'] = 'excessive_missing'
        precip_stats['precip_quality_score'] = 2  # Poor
    elif precip_stats['precip_pct_missing'] > 10:
        precip_stats['precip_quality'] = 'significant_missing'
        precip_stats['precip_quality_score'] = 3  # Fair
    elif precip_stats['precip_pct_missing'] > 5:
        precip_stats['precip_quality'] = 'minor_missing'
        precip_stats['precip_quality_score'] = 4  # Good
    else:
        precip_stats['precip_quality'] = 'excellent'
        precip_stats['precip_quality_score'] = 5  # Excellent
        
    return precip_stats

def check_joint_exceedances(df, null_value=-99.9999, sl_percentile=95, precip_percentile=95):
    """Calculate preliminary joint exceedance statistics."""
    # Get valid sea level data
    valid_sl = df[df['sea_level'] != null_value]['sea_level']
    
    if valid_sl.empty:
        return {
            'joint_exceedances': 0,
            'sl_exceedances': 0,
            'precip_exceedances': 0,
            'CPR': None,
            'joint_quality': 'unusable'
        }
    
    # Ensure we have the consolidated precipitation column
    if 'precipitation_mm' not in df.columns:
        if 'total_precipitation_mm' in df.columns:
            df['precipitation_mm'] = df['total_precipitation_mm'].copy()
            
            # Use ground data where available if it exists
            if 'ground_precipitation' in df.columns:
                # Determine if ground data is already in mm
                if df['ground_precipitation'].max() > df['total_precipitation'].max() * 100:
                    ground_mask = ~df['ground_precipitation'].isna() & (df['ground_precipitation'] > 0)
                    df.loc[ground_mask, 'precipitation_mm'] = df.loc[ground_mask, 'ground_precipitation']
                else:
                    # Convert ground data to mm
                    df['ground_precipitation_mm'] = df['ground_precipitation'] * 1000
                    ground_mask = ~df['ground_precipitation_mm'].isna() & (df['ground_precipitation_mm'] > 0)
                    df.loc[ground_mask, 'precipitation_mm'] = df.loc[ground_mask, 'ground_precipitation_mm']
        
    # Calculate thresholds
    sl_threshold = valid_sl.quantile(sl_percentile/100)
    precip_threshold = df['precipitation_mm'].quantile(precip_percentile/100)
    
    # Create masks for exceedances
    sl_exceed = (df['sea_level'] != null_value) & (df['sea_level'] > sl_threshold)
    precip_exceed = df['precipitation_mm'] > precip_threshold
    
    # Calculate exceedance counts
    sl_exceedances = sl_exceed.sum()
    precip_exceedances = precip_exceed.sum()
    joint_exceedances = (sl_exceed & precip_exceed).sum()
    
    # Calculate Conditional Probability Ratio
    if sl_exceedances > 0 and precip_exceedances > 0:
        # CPR = P(A∩B) / (P(A) * P(B))
        p_sl = sl_exceedances / len(df)
        p_precip = precip_exceedances / len(df)
        p_joint = joint_exceedances / len(df)
        
        CPR = p_joint / (p_sl * p_precip) if (p_sl * p_precip) > 0 else None
    else:
        CPR = None
        
    # Determine quality based on joint exceedances
    joint_stats = {
        'sl_threshold': sl_threshold,
        'precip_threshold': precip_threshold,
        'joint_exceedances': joint_exceedances,
        'sl_exceedances': sl_exceedances,
        'precip_exceedances': precip_exceedances,
        'CPR': CPR
    }
    
    if joint_exceedances == 0:
        joint_stats['joint_quality'] = 'no_joint_events'
    elif joint_exceedances < 10:
        joint_stats['joint_quality'] = 'very_few_joint_events'
    elif joint_exceedances < 50:
        joint_stats['joint_quality'] = 'few_joint_events'
    else:
        joint_stats['joint_quality'] = 'sufficient_joint_events'
    
    return joint_stats

def calculate_overall_quality(station_stats):
    """
    Calculate overall station quality based on individual metrics.
    
    Args:
        station_stats: Dict with station statistics
        
    Returns:
        Dict with overall quality assessment
    """
    # Determine the quality tier
    if (station_stats.get('sl_quality') == 'unrealistic_values' or
        station_stats.get('joint_quality') == 'no_joint_events'):
        tier = 'D'  # Unusable
        tier_desc = 'Unusable - severe data quality issues'
    elif (station_stats.get('sl_quality') == 'excessive_missing' or
          station_stats.get('precip_quality') == 'excessive_missing' or
          station_stats.get('joint_quality') == 'very_few_joint_events'):
        tier = 'C'  # Problematic
        tier_desc = 'Problematic - major issues but potentially usable with caveats'
    elif (station_stats.get('sl_quality') in ['significant_missing', 'minor_missing'] or
          station_stats.get('joint_quality') == 'few_joint_events'):
        tier = 'B'  # Good
        tier_desc = 'Good - minor issues that can be corrected'
    else:
        tier = 'A'  # Excellent
        tier_desc = 'Excellent - complete data, realistic values'
        
    # Calculate a numeric quality score (0-100)
    sl_score = station_stats.get('sl_quality_score', 0) * 10  # 0-50
    precip_score = station_stats.get('precip_quality_score', 0) * 5  # 0-25
    
    # Joint exceedance score
    if station_stats.get('joint_quality') == 'sufficient_joint_events':
        joint_score = 25
    elif station_stats.get('joint_quality') == 'few_joint_events':
        joint_score = 15
    elif station_stats.get('joint_quality') == 'very_few_joint_events':
        joint_score = 5
    else:
        joint_score = 0
        
    overall_score = sl_score + precip_score + joint_score
    
    return {
        'quality_tier': tier,
        'quality_description': tier_desc,
        'quality_score': overall_score
    }

def analyze_station_quality(df, metadata, null_value=-99.9999):
    """
    Perform comprehensive quality analysis on a station.
    
    Args:
        df: Station data DataFrame
        metadata: Dict with station metadata
        null_value: Value used for missing data
        
    Returns:
        Dict with comprehensive quality assessment
    """
    # Get station timing information
    timing_stats = {
        'start_date': df['datetime'].min(),
        'end_date': df['datetime'].max(),
        'duration_years': (df['datetime'].max() - df['datetime'].min()).days / 365.25,
        'total_rows': len(df)
    }
    
    # Check sea level data
    sl_stats = check_sea_level_range(df, null_value=null_value)
    
    # Check precipitation data
    precip_stats = check_precipitation_data(df)
    
    # Check joint exceedances
    joint_stats = check_joint_exceedances(df, null_value=null_value)
    
    # Combine all statistics
    station_stats = {**metadata, **timing_stats, **sl_stats, **precip_stats, **joint_stats}
    
    # Calculate overall quality
    quality_assessment = calculate_overall_quality(station_stats)
    station_stats.update(quality_assessment)
    
    return station_stats