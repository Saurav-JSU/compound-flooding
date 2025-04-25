#!/usr/bin/env python
"""
Compare ERA5 and ground precipitation data to check for unit discrepancies.
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Specify the file to inspect
file_path = "outputs/netcdf/240A_preprocessed.nc"

if len(sys.argv) > 1:
    file_path = sys.argv[1]

print(f"Analyzing precipitation in: {file_path}")

# Load the dataset
ds = xr.open_dataset(file_path)
station_code = Path(file_path).stem.split('_')[0]

# Set up plots directory
plots_dir = Path("outputs/plots")
plots_dir.mkdir(exist_ok=True, parents=True)

# Basic statistics for both types of precipitation
print("\n=== PRECIPITATION DATA COMPARISON ===")

for var in ['total_precipitation', 'ground_precipitation']:
    if var in ds:
        # Basic statistics
        data = ds[var].values
        non_nan = ~np.isnan(data)
        
        # Skip if no valid data
        if not np.any(non_nan):
            print(f"{var}: No valid data found")
            continue
            
        valid_data = data[non_nan]
        non_zero = valid_data > 0
        
        # Print statistics
        print(f"\n{var}:")
        print(f"  Valid points: {np.sum(non_nan)}/{len(data)} ({100*np.sum(non_nan)/len(data):.2f}%)")
        print(f"  Non-zero points: {np.sum(non_zero)}/{len(valid_data)} ({100*np.sum(non_zero)/len(valid_data):.2f}%)")
        print(f"  Min: {np.min(valid_data):.6f}")
        print(f"  Max: {np.max(valid_data):.6f}")
        print(f"  Mean: {np.mean(valid_data):.6f}")
        print(f"  Median non-zero: {np.median(valid_data[non_zero]) if np.any(non_zero) else 'N/A':.6f}")
        print(f"  95th percentile: {np.percentile(valid_data, 95):.6f}")
        print(f"  99th percentile: {np.percentile(valid_data, 99):.6f}")
        
        # Check if values might need unit conversion
        if var == 'total_precipitation' and np.max(valid_data) < 1.0:
            print(f"  NOTE: Maximum value < 1.0 suggests {var} might be in meters rather than mm/hour")
            print(f"        If converted to mm/hour: Max = {np.max(valid_data) * 1000:.2f} mm/hour")
    else:
        print(f"{var} not found in dataset")

# Time series plot for both precipitation types (1 month sample)
if 'total_precipitation' in ds and 'ground_precipitation' in ds:
    print("\nCreating precipitation comparison plots...")
    
    # Find a period with some precipitation for visualization
    # Combine both precipitation sources to find interesting periods
    total_precip = ds.total_precipitation.values
    ground_precip = ds.ground_precipitation.values
    
    # Use the maximum of both at each timestep
    combined_precip = np.maximum(total_precip, ground_precip)
    
    # Find index of highest precipitation
    try:
        # Get top 10 precipitation events
        top_indices = np.argsort(combined_precip)[-10:]
        
        # Take 3 days before and after the highest precipitation
        for idx in reversed(top_indices):  # Start with highest
            if idx > 72 and idx < len(combined_precip) - 72:
                center_idx = idx
                # Take 3 days before and after
                start_idx = center_idx - 72  # 3 days before (24 hours * 3)
                end_idx = center_idx + 72    # 3 days after
                
                # Check if this period has both types of precipitation
                period_total = total_precip[start_idx:end_idx]
                period_ground = ground_precip[start_idx:end_idx]
                
                if np.max(period_total) > 0 and np.max(period_ground) > 0:
                    break
        else:
            # If no good period found, use the highest precipitation point
            center_idx = np.argmax(combined_precip)
            start_idx = max(0, center_idx - 72)
            end_idx = min(len(combined_precip), center_idx + 72)
        
        # Extract the time slice
        time_slice = ds.isel(time=slice(start_idx, end_idx))
        
        # Create plots
        plt.figure(figsize=(12, 8))
        
        # Plot with units as-is
        plt.subplot(2, 1, 1)
        time_slice.total_precipitation.plot(label='ERA5 (total_precipitation)')
        time_slice.ground_precipitation.plot(label='Ground (ground_precipitation)')
        plt.title(f'Precipitation Comparison (Original Units) - Station {station_code}')
        plt.ylabel('Precipitation (original units)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot with ERA5 converted to mm/hour
        plt.subplot(2, 1, 2)
        # Convert ERA5 (total_precipitation) to mm/hour if it seems to be in meters
        if np.max(total_precip) < 1.0:
            (time_slice.total_precipitation * 1000).plot(label='ERA5 (converted to mm/hour)')
        else:
            time_slice.total_precipitation.plot(label='ERA5 (total_precipitation)')
            
        time_slice.ground_precipitation.plot(label='Ground (ground_precipitation)')
        plt.title('Precipitation Comparison (ERA5 converted to mm/hour if needed)')
        plt.ylabel('Precipitation (mm/hour)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = plots_dir / f"{station_code}_precipitation_comparison.png"
        plt.savefig(plot_file, dpi=100)
        print(f"Saved precipitation comparison to {plot_file}")
        
        # Create scatterplot to compare values
        plt.figure(figsize=(10, 8))
        
        # Get matching timestamps where both sources have data
        valid_mask = (~np.isnan(time_slice.total_precipitation)) & (~np.isnan(time_slice.ground_precipitation))
        
        if np.any(valid_mask):
            x = time_slice.total_precipitation.values[valid_mask]
            y = time_slice.ground_precipitation.values[valid_mask]
            
            # If ERA5 seems to be in meters, convert to mm/hour for comparison
            if np.max(total_precip) < 1.0:
                x = x * 1000  # Convert to mm/hour
                
            # Plot scatter
            plt.scatter(x, y, alpha=0.5)
            
            # Add a 1:1 line
            max_val = max(np.max(x), np.max(y))
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='1:1 line')
            
            # Compute correlation
            valid_pairs = (x > 0) & (y > 0)  # Only where both have precipitation
            if np.sum(valid_pairs) > 5:
                corr = np.corrcoef(x[valid_pairs], y[valid_pairs])[0, 1]
                plt.title(f'ERA5 vs Ground Precipitation (Correlation: {corr:.3f})')
            else:
                plt.title('ERA5 vs Ground Precipitation')
                
            # Labels
            plt.xlabel('ERA5 Precipitation (mm/hour)' if np.max(total_precip) < 1.0 else 'ERA5 Precipitation')
            plt.ylabel('Ground Precipitation (mm/hour)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            scatter_file = plots_dir / f"{station_code}_precipitation_scatter.png"
            plt.savefig(scatter_file, dpi=100)
            print(f"Saved precipitation scatter plot to {scatter_file}")
        else:
            print("No overlapping valid data points for scatter plot")
        
        # Distribution comparison
        plt.figure(figsize=(12, 6))
        
        # Plot distributions of non-zero values
        era5_data = ds.total_precipitation.values
        ground_data = ds.ground_precipitation.values
        
        era5_valid = era5_data[~np.isnan(era5_data) & (era5_data > 0)]
        ground_valid = ground_data[~np.isnan(ground_data) & (ground_data > 0)]
        
        # If ERA5 seems to be in meters, convert to mm/hour
        if np.max(era5_valid) < 1.0:
            era5_valid = era5_valid * 1000
        
        # Create histograms
        if len(era5_valid) > 0:
            plt.subplot(1, 2, 1)
            plt.hist(era5_valid, bins=30, log=True)
            plt.title('ERA5 Precipitation (non-zero values)')
            plt.xlabel('mm/hour' if np.max(total_precip) < 1.0 else 'Original units')
            plt.ylabel('Count (log scale)')
            plt.grid(True, alpha=0.3)
        
        if len(ground_valid) > 0:
            plt.subplot(1, 2, 2)
            plt.hist(ground_valid, bins=30, log=True)
            plt.title('Ground Precipitation (non-zero values)')
            plt.xlabel('mm/hour')
            plt.ylabel('Count (log scale)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        dist_file = plots_dir / f"{station_code}_precipitation_distributions.png"
        plt.savefig(dist_file, dpi=100)
        print(f"Saved precipitation distribution plots to {dist_file}")
        
    except Exception as e:
        print(f"Error creating precipitation plots: {str(e)}")
        
else:
    print("Both precipitation variables not found in dataset")

print("\nAnalysis complete.")