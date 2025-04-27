"""
Event visualization module for compound flooding analysis.

This module provides visualizations of specific events (e.g., hurricanes, storms):
- Time series during major flood events
- Cross-station analysis of event responses
- Return period context for observed events
- Comparison between different events
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta

# Import base visualization utilities
from compound_flooding.visualization.base import (
    FIG_SIZES, set_publication_style, save_figure, 
    RED_BLUE_CMAP, CPR_CMAP, RISK_CMAP, SEA_CMAP, PRECIP_CMAP
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_event_data(
    netcdf_path: str,
    event_dates: Tuple[str, str],
    variables: List[str] = ['sea_level', 'total_precipitation']
) -> xr.Dataset:
    """
    Load event data from a NetCDF file for a specific time period.
    
    Parameters
    ----------
    netcdf_path : str
        Path to NetCDF file
    event_dates : Tuple[str, str]
        Start and end dates of event in ISO format ('YYYY-MM-DD')
    variables : List[str], optional
        Variables to extract
        
    Returns
    -------
    xr.Dataset
        Dataset with event data
    """
    try:
        # Load dataset
        ds = xr.open_dataset(netcdf_path)
        
        # Convert dates to datetime
        start_date = pd.to_datetime(event_dates[0])
        end_date = pd.to_datetime(event_dates[1])
        
        # Select time period and variables
        ds_event = ds.sel(datetime=slice(start_date, end_date))
        
        # Keep only requested variables
        if variables:
            var_keep = [v for v in variables if v in ds_event]
            if len(var_keep) == 0:
                logger.warning("None of the requested variables found in dataset")
                ds.close()
                return None
            ds_event = ds_event[var_keep]
        
        # Close original dataset
        ds.close()
        
        return ds_event
        
    except Exception as e:
        logger.error(f"Error loading event data: {e}")
        return None


def find_event_peaks(
    ds_event: xr.Dataset,
    variables: List[str] = ['sea_level', 'total_precipitation'],
    window_hours: int = 12
) -> Dict[str, Dict[str, Any]]:
    """
    Find peak values for each variable during an event.
    
    Parameters
    ----------
    ds_event : xr.Dataset
        Dataset with event data
    variables : List[str], optional
        Variables to analyze
    window_hours : int, optional
        Window size for peak detection (hours)
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary with peak information for each variable
    """
    peaks = {}
    
    for var in variables:
        if var not in ds_event:
            continue
            
        # Get variable data
        da = ds_event[var]
        
        # Skip if all values are NaN
        if np.isnan(da.values).all():
            continue
        
        # Find peak value
        peak_idx = np.nanargmax(da.values)
        peak_value = da.values[peak_idx]
        peak_time = da.datetime.values[peak_idx]
        
        # Find window around peak
        time_values = pd.DatetimeIndex(da.datetime.values)
        peak_dt = pd.to_datetime(peak_time)
        window_start = peak_dt - pd.Timedelta(hours=window_hours/2)
        window_end = peak_dt + pd.Timedelta(hours=window_hours/2)
        
        # Find values in window
        window_mask = (time_values >= window_start) & (time_values <= window_end)
        window_vals = da.values[window_mask]
        window_times = time_values[window_mask]
        
        # Compute window statistics
        mean_val = np.nanmean(window_vals)
        std_val = np.nanstd(window_vals)
        
        # Store peak information
        peaks[var] = {
            'peak_value': float(peak_value),
            'peak_time': peak_time,
            'window_mean': float(mean_val),
            'window_std': float(std_val),
            'window_start': window_start,
            'window_end': window_end
        }
    
    return peaks


def calculate_event_joint_exceedance(
    ds_event: xr.Dataset,
    thresholds: Dict[str, float],
    lag_hours: int = 0
) -> Dict[str, Any]:
    """
    Calculate joint exceedance statistics during an event.
    
    Parameters
    ----------
    ds_event : xr.Dataset
        Dataset with event data
    thresholds : Dict[str, float]
        Thresholds for each variable
    lag_hours : int, optional
        Lag window for joint exceedance
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with joint exceedance statistics
    """
    # Check if we have the required variables
    required_vars = ['sea_level', 'total_precipitation']
    if not all(var in ds_event for var in required_vars):
        logger.warning("Missing required variables for joint exceedance calculation")
        return None
    
    # Extract data arrays
    da_sl = ds_event['sea_level']
    da_pr = ds_event['total_precipitation']
    
    # Check if we have thresholds for both variables
    if not all(var in thresholds for var in required_vars):
        logger.warning("Missing thresholds for joint exceedance calculation")
        return None
    
    # Get thresholds
    sl_threshold = thresholds['sea_level']
    pr_threshold = thresholds['total_precipitation']
    
    # Convert to pandas Series for easier handling
    sl = da_sl.to_series()
    pr = da_pr.to_series()
    
    # Combine into DataFrame and drop NaN values
    df = pd.DataFrame({'sea_level': sl, 'total_precipitation': pr}).dropna()
    n_total = len(df)
    
    if n_total == 0:
        logger.warning("No overlapping non-NaN observations for joint exceedance calculation")
        return None
    
    # If lag window, compute rolling max of precipitation
    if lag_hours > 0:
        window = 2 * lag_hours + 1
        pr_roll = df['total_precipitation'].rolling(window=window, center=True, min_periods=1).max()
    else:
        pr_roll = df['total_precipitation']
    
    # Exceedance indicators
    sl_exceed = df['sea_level'] > sl_threshold
    pr_exceed = pr_roll > pr_threshold
    
    # Count exceedances
    n_sl = int(sl_exceed.sum())
    n_pr = int(pr_exceed.sum())
    n_joint = int((sl_exceed & pr_exceed).sum())
    
    # Compute probabilities
    p_sl = n_sl / n_total
    p_pr = n_pr / n_total
    p_joint = n_joint / n_total
    p_ind = p_sl * p_pr
    
    # Compute CPR
    cpr = p_joint / p_ind if p_ind > 0 else np.nan
    
    # Compile results
    results = {
        'n_total': n_total,
        'n_sl': n_sl,
        'n_pr': n_pr,
        'n_joint': n_joint,
        'p_sl': p_sl,
        'p_pr': p_pr,
        'p_joint': p_joint,
        'p_independent': p_ind,
        'cpr': cpr,
        'sl_threshold': sl_threshold,
        'pr_threshold': pr_threshold
    }
    
    return results


def plot_event_timeseries(
    ds_event: xr.Dataset,
    thresholds: Dict[str, float] = None,
    station_code: str = None,
    event_name: str = None,
    tier1_data: Dict = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a time series plot of an event with thresholds and return period context.
    
    Parameters
    ----------
    ds_event : xr.Dataset
        Dataset with event data
    thresholds : Dict[str, float], optional
        Thresholds for each variable. If None, extracted from tier1_data if provided.
    station_code : str, optional
        Station code
    event_name : str, optional
        Name of the event (e.g., hurricane name)
    tier1_data : Dict, optional
        Tier 1 analysis results for the station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if we have the required variables
    required_vars = ['sea_level', 'total_precipitation']
    if not all(var in ds_event for var in required_vars):
        logger.warning("Missing required variables for event time series plot")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, "Missing required variables for event plot",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Get thresholds from tier1_data if not provided
    if thresholds is None:
        thresholds = {}
        if tier1_data and station_code in tier1_data:
            t1_data = tier1_data[station_code]
            for var in required_vars:
                if var in t1_data:
                    thresholds[var] = t1_data[var].get('threshold')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZES['tall'], sharex=True)
    
    # Extract time values
    time_values = pd.DatetimeIndex(ds_event.datetime.values)
    
    # Find event peaks
    peaks = find_event_peaks(ds_event, required_vars)
    
    # Set up subplot for sea level
    ax = ax1
    var = 'sea_level'
    
    # Get data
    da = ds_event[var]
    sl_values = da.values
    sl_times = time_values
    
    # Plot sea level time series
    ax.plot(sl_times, sl_values, 'b-', linewidth=2, label='Sea Level')
    
    # Add threshold if available
    if var in thresholds and thresholds[var] is not None:
        ax.axhline(thresholds[var], color='r', linestyle='--', 
                  label=f'Threshold ({thresholds[var]:.2f})')
    
    # Mark peak if available
    if var in peaks:
        peak_info = peaks[var]
        peak_value = peak_info['peak_value']
        peak_time = peak_info['peak_time']
        ax.plot(peak_time, peak_value, 'ro', markersize=8, 
               label=f'Peak: {peak_value:.2f}')
        
        # Add return period context if available
        if tier1_data and station_code in tier1_data:
            t1_data = tier1_data[station_code]
            if var in t1_data and 'return_levels' in t1_data[var]:
                rl_data = t1_data[var]['return_levels']
                if isinstance(rl_data, list):
                    # Find closest return period
                    closest_rl = None
                    closest_diff = float('inf')
                    for rl in rl_data:
                        if isinstance(rl, dict) and 'return_level' in rl and 'return_period' in rl:
                            diff = abs(rl['return_level'] - peak_value)
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_rl = rl
                    
                    if closest_rl:
                        # Add text annotation with return period
                        rp = closest_rl['return_period']
                        rl = closest_rl['return_level']
                        ax.text(peak_time, peak_value, f" ≈{rp:.0f}yr", 
                               fontsize=10, ha='left', va='bottom')
    
    # Set labels and title
    ax.set_ylabel('Sea Level (m)')
    ax.set_title(f"Sea Level Time Series{' - ' + station_code if station_code else ''}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Set up subplot for precipitation
    ax = ax2
    var = 'total_precipitation'
    
    # Get data
    da = ds_event[var]
    pr_values = da.values
    pr_times = time_values
    
    # Plot precipitation time series
    ax.plot(pr_times, pr_values, 'g-', linewidth=2, label='Precipitation')
    
    # Add threshold if available
    if var in thresholds and thresholds[var] is not None:
        ax.axhline(thresholds[var], color='r', linestyle='--', 
                  label=f'Threshold ({thresholds[var]:.2f})')
    
    # Mark peak if available
    if var in peaks:
        peak_info = peaks[var]
        peak_value = peak_info['peak_value']
        peak_time = peak_info['peak_time']
        ax.plot(peak_time, peak_value, 'ro', markersize=8, 
               label=f'Peak: {peak_value:.2f}')
        
        # Add return period context if available
        if tier1_data and station_code in tier1_data:
            t1_data = tier1_data[station_code]
            if var in t1_data and 'return_levels' in t1_data[var]:
                rl_data = t1_data[var]['return_levels']
                if isinstance(rl_data, list):
                    # Find closest return period
                    closest_rl = None
                    closest_diff = float('inf')
                    for rl in rl_data:
                        if isinstance(rl, dict) and 'return_level' in rl and 'return_period' in rl:
                            diff = abs(rl['return_level'] - peak_value)
                            if diff < closest_diff:
                                closest_diff = diff
                                closest_rl = rl
                    
                    if closest_rl:
                        # Add text annotation with return period
                        rp = closest_rl['return_period']
                        rl = closest_rl['return_level']
                        ax.text(peak_time, peak_value, f" ≈{rp:.0f}yr", 
                               fontsize=10, ha='left', va='bottom')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Precipitation (mm/h)')
    ax.set_title(f"Precipitation Time Series")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format x-axis
    plt.gcf().autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Calculate joint exceedance if thresholds are available
    if thresholds and all(var in thresholds for var in required_vars):
        joint_stats = calculate_event_joint_exceedance(ds_event, thresholds)
        
        if joint_stats:
            # Add text with joint exceedance statistics
            text_str = (
                f"Joint Exceedance Statistics:\n"
                f"Sea Level Exceedances: {joint_stats['n_sl']}/{joint_stats['n_total']} ({joint_stats['p_sl']:.2f})\n"
                f"Precipitation Exceedances: {joint_stats['n_pr']}/{joint_stats['n_total']} ({joint_stats['p_pr']:.2f})\n"
                f"Joint Exceedances: {joint_stats['n_joint']}/{joint_stats['n_total']} ({joint_stats['p_joint']:.2f})\n"
                f"CPR: {joint_stats['cpr']:.2f}"
            )
            
            fig.text(0.02, 0.02, text_str, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add event title if provided
    if event_name:
        fig.suptitle(f"Event: {event_name}", fontsize=16)
        fig.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{station_code}_{event_name}_timeseries".replace(' ', '_'))
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_multi_station_event(
    event_data: Dict[str, xr.Dataset],
    thresholds: Dict[str, Dict[str, float]],
    event_name: str = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare event impact across multiple stations.
    
    Parameters
    ----------
    event_data : Dict[str, xr.Dataset]
        Dictionary mapping station codes to event datasets
    thresholds : Dict[str, Dict[str, float]]
        Dictionary mapping station codes to variable thresholds
    event_name : str, optional
        Name of the event (e.g., hurricane name)
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if we have data for at least one station
    if not event_data:
        logger.warning("No event data provided")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, "No event data provided",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Extract station codes
    station_codes = list(event_data.keys())
    n_stations = len(station_codes)
    
    # Prepare data for comparison
    peaks_sl = []
    peaks_pr = []
    joint_exceed = []
    cprs = []
    
    for station in station_codes:
        ds = event_data[station]
        
        # Skip if required variables not available
        if not all(var in ds for var in ['sea_level', 'total_precipitation']):
            peaks_sl.append(np.nan)
            peaks_pr.append(np.nan)
            joint_exceed.append(np.nan)
            cprs.append(np.nan)
            continue
        
        # Find peaks
        peaks = find_event_peaks(ds, ['sea_level', 'total_precipitation'])
        
        # Extract peak values
        sl_peak = peaks.get('sea_level', {}).get('peak_value', np.nan)
        pr_peak = peaks.get('total_precipitation', {}).get('peak_value', np.nan)
        
        peaks_sl.append(sl_peak)
        peaks_pr.append(pr_peak)
        
        # Calculate joint exceedance if thresholds available
        if station in thresholds:
            joint_stats = calculate_event_joint_exceedance(ds, thresholds[station])
            if joint_stats:
                joint_exceed.append(joint_stats['p_joint'])
                cprs.append(joint_stats['cpr'])
            else:
                joint_exceed.append(np.nan)
                cprs.append(np.nan)
        else:
            joint_exceed.append(np.nan)
            cprs.append(np.nan)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES['full'])
    
    # Plot 1: Sea Level Peaks
    ax = axes[0, 0]
    valid = np.isfinite(peaks_sl)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(peaks_sl))
        sorted_stations = [station_codes[i] for i in sorted_indices]
        sorted_peaks = [peaks_sl[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_stations, sorted_peaks, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            if i < len(sorted_peaks) // 3:
                bar.set_color('green')
            elif i < 2 * len(sorted_peaks) // 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Sea Level Peak (m)')
        ax.set_title('Sea Level Peaks by Station')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid sea level peak data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Plot 2: Precipitation Peaks
    ax = axes[0, 1]
    valid = np.isfinite(peaks_pr)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(peaks_pr))
        sorted_stations = [station_codes[i] for i in sorted_indices]
        sorted_peaks = [peaks_pr[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_stations, sorted_peaks, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            if i < len(sorted_peaks) // 3:
                bar.set_color('green')
            elif i < 2 * len(sorted_peaks) // 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Precipitation Peak (mm/h)')
        ax.set_title('Precipitation Peaks by Station')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid precipitation peak data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Plot 3: Joint Exceedance Probability
    ax = axes[1, 0]
    valid = np.isfinite(joint_exceed)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(joint_exceed))
        sorted_stations = [station_codes[i] for i in sorted_indices]
        sorted_values = [joint_exceed[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_stations, sorted_values, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            if i < len(sorted_values) // 3:
                bar.set_color('green')
            elif i < 2 * len(sorted_values) // 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Joint Exceedance Probability')
        ax.set_title('Joint Exceedance by Station')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid joint exceedance data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Plot 4: CPR
    ax = axes[1, 1]
    valid = np.isfinite(cprs)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(cprs))
        sorted_stations = [station_codes[i] for i in sorted_indices]
        sorted_values = [cprs[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_stations, sorted_values, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            value = sorted_values[i]
            if value < 1:
                bar.set_color('blue')  # Negative dependence
            elif value < 2:
                bar.set_color('green')  # Weak positive dependence
            elif value < 4:
                bar.set_color('orange')  # Moderate positive dependence
            else:
                bar.set_color('red')  # Strong positive dependence
        
        # Add reference line for independence
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.7, label='Independence')
        
        ax.set_xlabel('Conditional Probability Ratio (CPR)')
        ax.set_title('CPR by Station')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid CPR data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Add event title if provided
    if event_name:
        fig.suptitle(f"Multi-Station Analysis: {event_name}", fontsize=16)
        fig.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"multi_station_{event_name}".replace(' ', '_'))
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_multi_event_comparison(
    events_data: Dict[str, Dict[str, Any]],
    station_code: str,
    tier1_data: Dict = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare multiple events at the same station.
    
    Parameters
    ----------
    events_data : Dict[str, Dict[str, Any]]
        Dictionary mapping event names to event metrics
    station_code : str
        Station code
    tier1_data : Dict, optional
        Tier 1 analysis results for the station
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Check if we have data for at least one event
    if not events_data:
        logger.warning("No event data provided")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, "No event data provided",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Extract event names
    event_names = list(events_data.keys())
    n_events = len(event_names)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES['full'])
    
    # Prepare data for comparison
    sl_peaks = []
    pr_peaks = []
    joint_probs = []
    cprs = []
    
    for event in event_names:
        event_metrics = events_data[event]
        
        # Extract metrics
        sl_peak = event_metrics.get('sea_level_peak', np.nan)
        pr_peak = event_metrics.get('precipitation_peak', np.nan)
        joint_prob = event_metrics.get('joint_probability', np.nan)
        cpr = event_metrics.get('cpr', np.nan)
        
        sl_peaks.append(sl_peak)
        pr_peaks.append(pr_peak)
        joint_probs.append(joint_prob)
        cprs.append(cpr)
    
    # Plot 1: Sea Level Peaks
    ax = axes[0, 0]
    valid = np.isfinite(sl_peaks)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(sl_peaks))
        sorted_events = [event_names[i] for i in sorted_indices]
        sorted_peaks = [sl_peaks[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_events, sorted_peaks, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            if i < len(sorted_peaks) // 3:
                bar.set_color('green')
            elif i < 2 * len(sorted_peaks) // 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add return period context if available
        if tier1_data and station_code in tier1_data:
            t1_data = tier1_data[station_code]
            if 'sea_level' in t1_data and 'return_levels' in t1_data['sea_level']:
                rl_data = t1_data['sea_level']['return_levels']
                if isinstance(rl_data, list):
                    # Add reference lines for return periods
                    for rp in [10, 50, 100]:
                        for rl in rl_data:
                            if isinstance(rl, dict) and rl.get('return_period') == rp:
                                rl_val = rl.get('return_level')
                                if rl_val is not None:
                                    ax.axvline(rl_val, color='blue', linestyle='--', alpha=0.7,
                                             label=f'{rp}yr return level')
                                    ax.text(rl_val, 0, f'{rp}yr', fontsize=8, color='blue',
                                           ha='center', va='bottom')
                                    break
        
        ax.set_xlabel('Sea Level Peak (m)')
        ax.set_title('Sea Level Peaks by Event')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid sea level peak data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Plot 2: Precipitation Peaks
    ax = axes[0, 1]
    valid = np.isfinite(pr_peaks)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(pr_peaks))
        sorted_events = [event_names[i] for i in sorted_indices]
        sorted_peaks = [pr_peaks[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_events, sorted_peaks, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            if i < len(sorted_peaks) // 3:
                bar.set_color('green')
            elif i < 2 * len(sorted_peaks) // 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Add return period context if available
        if tier1_data and station_code in tier1_data:
            t1_data = tier1_data[station_code]
            if 'total_precipitation' in t1_data and 'return_levels' in t1_data['total_precipitation']:
                rl_data = t1_data['total_precipitation']['return_levels']
                if isinstance(rl_data, list):
                    # Add reference lines for return periods
                    for rp in [10, 50, 100]:
                        for rl in rl_data:
                            if isinstance(rl, dict) and rl.get('return_period') == rp:
                                rl_val = rl.get('return_level')
                                if rl_val is not None:
                                    ax.axvline(rl_val, color='blue', linestyle='--', alpha=0.7,
                                             label=f'{rp}yr return level')
                                    ax.text(rl_val, 0, f'{rp}yr', fontsize=8, color='blue',
                                           ha='center', va='bottom')
                                    break
        
        ax.set_xlabel('Precipitation Peak (mm/h)')
        ax.set_title('Precipitation Peaks by Event')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid precipitation peak data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Plot 3: Joint Probabilities
    ax = axes[1, 0]
    valid = np.isfinite(joint_probs)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(joint_probs))
        sorted_events = [event_names[i] for i in sorted_indices]
        sorted_values = [joint_probs[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_events, sorted_values, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            if i < len(sorted_values) // 3:
                bar.set_color('green')
            elif i < 2 * len(sorted_values) // 3:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Joint Exceedance Probability')
        ax.set_title('Joint Exceedance by Event')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid joint probability data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Plot 4: CPRs
    ax = axes[1, 1]
    valid = np.isfinite(cprs)
    if np.any(valid):
        sorted_indices = np.argsort(np.array(cprs))
        sorted_events = [event_names[i] for i in sorted_indices]
        sorted_values = [cprs[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_events, sorted_values, alpha=0.7)
        
        # Customize bars
        for i, bar in enumerate(bars):
            value = sorted_values[i]
            if value < 1:
                bar.set_color('blue')  # Negative dependence
            elif value < 2:
                bar.set_color('green')  # Weak positive dependence
            elif value < 4:
                bar.set_color('orange')  # Moderate positive dependence
            else:
                bar.set_color('red')  # Strong positive dependence
        
        # Add reference line for independence
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.7, label='Independence')
        
        # Add reference line for station's baseline CPR if available
        if tier1_data and station_code in tier1_data:
            t1_data = tier1_data[station_code]
            if 'joint' in t1_data and 'empirical' in t1_data['joint']:
                baseline_cpr = t1_data['joint']['empirical'].get('cpr')
                if baseline_cpr is not None:
                    ax.axvline(baseline_cpr, color='blue', linestyle='-', alpha=0.7,
                              label=f'Station baseline: {baseline_cpr:.2f}')
        
        ax.set_xlabel('Conditional Probability Ratio (CPR)')
        ax.set_title('CPR by Event')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid CPR data",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Add station title
    fig.suptitle(f"Multi-Event Comparison: {station_code}", fontsize=16)
    fig.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{station_code}_multi_event_comparison")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def analyze_hurricane_event(
    netcdf_dir: str,
    event_dates: Tuple[str, str],
    event_name: str,
    tier1_data: Dict,
    output_dir: str,
    station_codes: List[str] = None,
    show: bool = False
) -> None:
    """
    Analyze and visualize a hurricane event across multiple stations.
    
    Parameters
    ----------
    netcdf_dir : str
        Directory containing NetCDF files
    event_dates : Tuple[str, str]
        Start and end dates of event in ISO format ('YYYY-MM-DD')
    event_name : str
        Name of the event (e.g., hurricane name)
    tier1_data : Dict
        Tier 1 analysis results
    output_dir : str
        Directory to save figures
    station_codes : List[str], optional
        List of station codes to analyze. If None, use all stations in tier1_data.
    show : bool, optional
        Whether to display the figures
    """
    # Set publication style
    set_publication_style()
    
    # Create output directory
    event_dir = os.path.join(output_dir, event_name.replace(' ', '_').lower())
    os.makedirs(event_dir, exist_ok=True)
    
    # Determine station codes to analyze
    if station_codes is None:
        station_codes = list(tier1_data.keys())
    
    logger.info(f"Analyzing {event_name} event for {len(station_codes)} stations")
    
    # Load event data for each station
    event_data = {}
    event_metrics = {}
    thresholds = {}
    
    for station in station_codes:
        # Skip if station not in tier1_data
        if station not in tier1_data:
            continue
            
        # Get thresholds from tier1 data
        station_thresholds = {}
        t1_data = tier1_data[station]
        
        for var in ['sea_level', 'total_precipitation']:
            if var in t1_data:
                station_thresholds[var] = t1_data[var].get('threshold')
        
        # Skip if we don't have thresholds
        if len(station_thresholds) < 2:
            continue
            
        # Look for NetCDF file
        nc_file = os.path.join(netcdf_dir, f"{station}.nc")
        if not os.path.exists(nc_file):
            continue
            
        # Load event data
        ds_event = load_event_data(nc_file, event_dates)
        if ds_event is None:
            continue
            
        # Store event data and thresholds
        event_data[station] = ds_event
        thresholds[station] = station_thresholds
        
        # Extract event metrics
        peaks = find_event_peaks(ds_event, ['sea_level', 'total_precipitation'])
        joint_stats = calculate_event_joint_exceedance(ds_event, station_thresholds)
        
        metrics = {}
        if 'sea_level' in peaks:
            metrics['sea_level_peak'] = peaks['sea_level']['peak_value']
            metrics['sea_level_peak_time'] = peaks['sea_level']['peak_time']
        
        if 'total_precipitation' in peaks:
            metrics['precipitation_peak'] = peaks['total_precipitation']['peak_value']
            metrics['precipitation_peak_time'] = peaks['total_precipitation']['peak_time']
        
        if joint_stats:
            metrics['joint_probability'] = joint_stats['p_joint']
            metrics['cpr'] = joint_stats['cpr']
        
        event_metrics[station] = metrics
    
    # Skip if no data found
    if not event_data:
        logger.warning(f"No data found for {event_name} event")
        return
    
    logger.info(f"Found data for {len(event_data)} stations")
    
    # Create individual station plots
    for station, ds in event_data.items():
        logger.info(f"Generating time series plot for station {station}")
        
        plot_event_timeseries(
            ds_event=ds,
            thresholds=thresholds[station],
            station_code=station,
            event_name=event_name,
            tier1_data=tier1_data,
            output_dir=event_dir,
            show=show
        )
    
    # Create multi-station comparison
    logger.info("Generating multi-station comparison")
    
    plot_multi_station_event(
        event_data=event_data,
        thresholds=thresholds,
        event_name=event_name,
        output_dir=event_dir,
        show=show
    )
    
    logger.info(f"{event_name} event analysis completed and saved to {event_dir}")


def compare_multiple_events(
    events_config: List[Dict],
    netcdf_dir: str,
    tier1_data: Dict,
    output_dir: str,
    station_codes: List[str] = None,
    show: bool = False
) -> None:
    """
    Compare multiple events at the same stations.
    
    Parameters
    ----------
    events_config : List[Dict]
        List of event configurations:
        [{'name': 'Event 1', 'dates': ('2018-01-01', '2018-01-10')}, ...]
    netcdf_dir : str
        Directory containing NetCDF files
    tier1_data : Dict
        Tier 1 analysis results
    output_dir : str
        Directory to save figures
    station_codes : List[str], optional
        List of station codes to analyze. If None, use all stations in tier1_data.
    show : bool, optional
        Whether to display the figures
    """
    # Set publication style
    set_publication_style()
    
    # Create output directory
    event_dir = os.path.join(output_dir, "event_comparisons")
    os.makedirs(event_dir, exist_ok=True)
    
    # Determine station codes to analyze
    if station_codes is None:
        station_codes = list(tier1_data.keys())
    
    logger.info(f"Comparing {len(events_config)} events for {len(station_codes)} stations")
    
    # Process each station
    for station in station_codes:
        # Skip if station not in tier1_data
        if station not in tier1_data:
            continue
            
        # Get thresholds from tier1 data
        station_thresholds = {}
        t1_data = tier1_data[station]
        
        for var in ['sea_level', 'total_precipitation']:
            if var in t1_data:
                station_thresholds[var] = t1_data[var].get('threshold')
        
        # Skip if we don't have thresholds
        if len(station_thresholds) < 2:
            continue
            
        # Look for NetCDF file
        nc_file = os.path.join(netcdf_dir, f"{station}.nc")
        if not os.path.exists(nc_file):
            continue
        
        # Process each event
        event_metrics = {}
        
        for event_config in events_config:
            event_name = event_config['name']
            event_dates = event_config['dates']
            
            # Load event data
            ds_event = load_event_data(nc_file, event_dates)
            if ds_event is None:
                continue
                
            # Extract event metrics
            peaks = find_event_peaks(ds_event, ['sea_level', 'total_precipitation'])
            joint_stats = calculate_event_joint_exceedance(ds_event, station_thresholds)
            
            metrics = {}
            if 'sea_level' in peaks:
                metrics['sea_level_peak'] = peaks['sea_level']['peak_value']
                metrics['sea_level_peak_time'] = peaks['sea_level']['peak_time']
            
            if 'total_precipitation' in peaks:
                metrics['precipitation_peak'] = peaks['total_precipitation']['peak_value']
                metrics['precipitation_peak_time'] = peaks['total_precipitation']['peak_time']
            
            if joint_stats:
                metrics['joint_probability'] = joint_stats['p_joint']
                metrics['cpr'] = joint_stats['cpr']
            
            event_metrics[event_name] = metrics
        
        # Skip if less than 2 events have data
        if len(event_metrics) < 2:
            logger.info(f"Insufficient event data for station {station}, skipping")
            continue
            
        # Create multi-event comparison
        logger.info(f"Generating multi-event comparison for station {station}")
        
        plot_multi_event_comparison(
            events_data=event_metrics,
            station_code=station,
            tier1_data=tier1_data,
            output_dir=event_dir,
            show=show
        )
    
    logger.info(f"Multi-event comparison completed and saved to {event_dir}")


if __name__ == "__main__":
    # Basic test of the module
    import sys
    from compound_flooding.visualization.base import (
        load_tier1_results,
        create_output_dirs
    )
    
    print("Testing events module...")
    
    # Set publication style
    set_publication_style()
    
    # Create output directories
    dirs = create_output_dirs('outputs/plots_test')
    
    # Check if we have required arguments
    if len(sys.argv) > 2:
        nc_file = sys.argv[1]
        tier1_file = sys.argv[2]
        
        try:
            # Load Tier 1 data
            tier1_data = load_tier1_results(os.path.dirname(tier1_file))
            print(f"Loaded Tier 1 data for {len(tier1_data)} stations")
            
            # Get station code from filename
            station_code = os.path.splitext(os.path.basename(nc_file))[0]
            
            # Define a test event
            event_dates = ('2018-01-01', '2018-01-10')  # Example dates
            event_name = "Test Storm"
            
            # Load event data
            ds_event = load_event_data(nc_file, event_dates)
            
            if ds_event is not None:
                print(f"Loaded event data for {station_code}")
                
                # Get thresholds from tier1 data
                thresholds = {}
                if station_code in tier1_data:
                    t1_data = tier1_data[station_code]
                    for var in ['sea_level', 'total_precipitation']:
                        if var in t1_data:
                            thresholds[var] = t1_data[var].get('threshold')
                
                # Test event time series plot
                plot_event_timeseries(
                    ds_event=ds_event,
                    thresholds=thresholds,
                    station_code=station_code,
                    event_name=event_name,
                    tier1_data=tier1_data,
                    output_dir=dirs['events']
                )
                
                print(f"Test complete. Check outputs in {dirs['events']}")
            else:
                print("No event data found in the specified date range")
                
        except Exception as e:
            print(f"Error testing with provided files: {e}")
    else:
        print("No test data provided. Please provide a NetCDF file and Tier 1 output file as arguments.")