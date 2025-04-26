# src/visualization/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def set_plotting_style():
    """Set consistent plotting style."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_quality_distribution(quality_df, output_dir='plots'):
    """
    Plot distributions of key quality metrics.
    
    Args:
        quality_df: DataFrame with station quality information
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plotting_style()
    
    # Plot 1: Distribution of quality tiers
    plt.figure(figsize=(10, 6))
    tier_counts = quality_df['quality_tier'].value_counts().sort_index()
    ax = tier_counts.plot(kind='bar', color=sns.color_palette("viridis", 4))
    plt.title('Distribution of Station Quality Tiers')
    plt.xlabel('Quality Tier')
    plt.ylabel('Number of Stations')
    
    # Add percentage labels
    total = tier_counts.sum()
    for i, count in enumerate(tier_counts):
        ax.text(i, count + 5, f'{count/total*100:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_tier_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 2: Missing sea level percentage
    plt.figure(figsize=(12, 6))
    sns.histplot(quality_df['sl_pct_missing'].clip(upper=50), bins=30)
    plt.axvline(x=5, color='green', linestyle='--', label='5% threshold')
    plt.axvline(x=25, color='red', linestyle='--', label='25% threshold')
    plt.title('Distribution of Missing Sea Level Data (%)')
    plt.xlabel('Missing Data (%)')
    plt.ylabel('Number of Stations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_sea_level_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 3: P99 sea level values
    plt.figure(figsize=(12, 6))
    sns.histplot(quality_df['sl_p99'].clip(upper=10), bins=30)
    plt.axvline(x=5, color='orange', linestyle='--', label='5m threshold')
    plt.title('Distribution of 99th Percentile Sea Level (clipped at 10m)')
    plt.xlabel('P99 Sea Level (m)')
    plt.ylabel('Number of Stations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'p99_sea_level_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 4: Joint exceedances
    plt.figure(figsize=(12, 6))
    sns.histplot(quality_df['joint_exceedances'].clip(upper=3000), bins=30)
    plt.title('Distribution of Joint Exceedances (clipped at 3000)')
    plt.xlabel('Number of Joint Exceedances')
    plt.ylabel('Number of Stations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_exceedances_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 5: CPR distribution
    plt.figure(figsize=(12, 6))
    valid_cpr = quality_df['CPR'].dropna().clip(upper=5)
    sns.histplot(valid_cpr, bins=30)
    plt.axvline(x=1, color='red', linestyle='--', label='Independence threshold')
    plt.title('Distribution of CPR (clipped at 5)')
    plt.xlabel('Conditional Probability Ratio')
    plt.ylabel('Number of Stations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpr_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 6: Map of station quality
    if all(col in quality_df.columns for col in ['latitude', 'longitude', 'quality_tier']):
        plt.figure(figsize=(15, 10))
        
        # Create a colormap for the tiers
        tier_colors = {'A': 'darkgreen', 'B': 'green', 'C': 'orange', 'D': 'red'}
        
        # Plot each tier separately for better legend control
        for tier, color in tier_colors.items():
            tier_df = quality_df[quality_df['quality_tier'] == tier]
            plt.scatter(tier_df['longitude'], tier_df['latitude'], 
                      c=color, label=f'Tier {tier}', alpha=0.7, edgecolors='white', s=50)
        
        plt.title('Spatial Distribution of Station Quality')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend(title='Quality Tier')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Try to add a basemap if cartopy is available
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            plt.figure(figsize=(15, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.STATES)
            
            for tier, color in tier_colors.items():
                tier_df = quality_df[quality_df['quality_tier'] == tier]
                ax.scatter(tier_df['longitude'], tier_df['latitude'], 
                         c=color, label=f'Tier {tier}', alpha=0.7, edgecolors='white', s=50,
                         transform=ccrs.PlateCarree())
            
            ax.set_global()
            ax.legend(title='Quality Tier')
            plt.title('Spatial Distribution of Station Quality')
        except ImportError:
            pass  # Use the simple plot if cartopy is not available
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'station_quality_map.png'), dpi=300)
        plt.close()

def plot_quality_issues(quality_df, output_dir='plots'):
    """
    Create plots highlighting specific quality issues.
    
    Args:
        quality_df: DataFrame with station quality information
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plotting_style()
    
    # Identify stations with quality issues
    unrealistic_sl = quality_df[quality_df['sl_quality'] == 'unrealistic_values']
    excessive_missing = quality_df[quality_df['sl_quality'] == 'excessive_missing']
    no_joint_events = quality_df[quality_df['joint_quality'] == 'no_joint_events']
    
    # Plot stations with unrealistic sea level values
    if not unrealistic_sl.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='site_code', y='sl_p99', data=unrealistic_sl.sort_values('sl_p99', ascending=False).head(20))
        plt.title('Stations with Unrealistic Sea Level Values (P99 > 50m)')
        plt.xlabel('Station Code')
        plt.ylabel('P99 Sea Level (m)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'unrealistic_sea_level_stations.png'), dpi=300)
        plt.close()
    
    # Plot stations with excessive missing data
    if not excessive_missing.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='site_code', y='sl_pct_missing', 
                   data=excessive_missing.sort_values('sl_pct_missing', ascending=False).head(20))
        plt.title('Stations with Excessive Missing Sea Level Data (>25%)')
        plt.xlabel('Station Code')
        plt.ylabel('Missing Sea Level Data (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'excessive_missing_data_stations.png'), dpi=300)
        plt.close()
    
    # Plot stations with no joint events
    if not no_joint_events.empty:
        plt.figure(figsize=(12, 6))
        if 'sl_exceedances' in no_joint_events.columns and 'precip_exceedances' in no_joint_events.columns:
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot sea level exceedances
            sns.barplot(x='site_code', y='sl_exceedances', data=no_joint_events.sort_values('sl_exceedances'), ax=ax1)
            ax1.set_title('Sea Level Exceedances for Stations with No Joint Events')
            ax1.set_xlabel('')
            ax1.set_ylabel('Sea Level Exceedances')
            ax1.tick_params(axis='x', rotation=90)
            
            # Plot precipitation exceedances
            sns.barplot(x='site_code', y='precip_exceedances', data=no_joint_events.sort_values('precip_exceedances'), ax=ax2)
            ax2.set_title('Precipitation Exceedances for Stations with No Joint Events')
            ax2.set_xlabel('Station Code')
            ax2.set_ylabel('Precipitation Exceedances')
            ax2.tick_params(axis='x', rotation=90)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'no_joint_events_stations.png'), dpi=300)
            plt.close()