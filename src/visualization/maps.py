"""
Maps visualization module for compound flooding analysis.

This module provides spatial visualizations of results across stations, including:
- Geographic distribution of GPD parameters
- Spatial patterns of correlation (τ, ρ)
- Regional patterns in CPR and tail dependence
- Risk hotspot identification
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import base visualization utilities
from compound_flooding.visualization.base import (
    FIG_SIZES, set_publication_style, save_figure, 
    RED_BLUE_CMAP, CPR_CMAP, RISK_CMAP, SEA_CMAP, PRECIP_CMAP,
    combine_with_metadata
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_base_map(
    station_df: pd.DataFrame, 
    region: str = 'usa',
    margin_pct: float = 0.1,
    rivers: bool = True,
    states: bool = True,
    counties: bool = False,
    resolution: str = 'i'
) -> Tuple[plt.Figure, plt.Axes, Basemap]:
    """
    Create a base map for station data visualization.
    
    Parameters
    ----------
    station_df : pd.DataFrame
        DataFrame with station metadata, must include 'latitude' and 'longitude' columns
    region : str, optional
        Region to display: 'usa', 'east_coast', 'gulf_coast', 'west_coast', 'world'
    margin_pct : float, optional
        Margin to add around stations (as percentage of extent)
    rivers : bool, optional
        Whether to draw rivers
    states : bool, optional
        Whether to draw state boundaries
    counties : bool, optional
        Whether to draw county boundaries
    resolution : str, optional
        Resolution of boundaries: 'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high)
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes, Basemap]
        Figure, Axes, and Basemap objects
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=FIG_SIZES['map'])
    
    # Extract lat/lon
    lats = station_df['latitude'].values
    lons = station_df['longitude'].values
    
    # Skip if no valid coordinates
    valid = np.isfinite(lats) & np.isfinite(lons)
    if np.sum(valid) == 0:
        logger.warning("No valid coordinates found in station data")
        ax.text(0.5, 0.5, "No valid coordinates found", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig, ax, None
    
    # Filter to valid coordinates
    lats = lats[valid]
    lons = lons[valid]
    
    # Default region settings
    if region == 'usa':
        # Contiguous USA
        lonmin, lonmax = -130, -65
        latmin, latmax = 23, 50
        proj = 'lcc'
        lat_0 = (latmin + latmax) / 2
        lon_0 = (lonmin + lonmax) / 2
        area_thresh = 1000
    elif region == 'east_coast':
        # US East Coast
        lonmin, lonmax = -82, -65
        latmin, latmax = 24, 46
        proj = 'lcc'
        lat_0 = (latmin + latmax) / 2
        lon_0 = (lonmin + lonmax) / 2
        area_thresh = 100
    elif region == 'gulf_coast':
        # Gulf of Mexico Coast
        lonmin, lonmax = -98, -80
        latmin, latmax = 24, 32
        proj = 'lcc'
        lat_0 = (latmin + latmax) / 2
        lon_0 = (lonmin + lonmax) / 2
        area_thresh = 100
    elif region == 'west_coast':
        # US West Coast
        lonmin, lonmax = -130, -115
        latmin, latmax = 32, 49
        proj = 'lcc'
        lat_0 = (latmin + latmax) / 2
        lon_0 = (lonmin + lonmax) / 2
        area_thresh = 100
    elif region == 'world':
        # World map
        lonmin, lonmax = -180, 180
        latmin, latmax = -60, 80
        proj = 'cyl'
        lat_0, lon_0 = 0, 0
        area_thresh = 10000
    else:
        # Auto-determine from data
        min_lat, max_lat = np.min(lats), np.max(lats)
        min_lon, max_lon = np.min(lons), np.max(lons)
        
        # Add margin
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        latmin = min_lat - lat_range * margin_pct
        latmax = max_lat + lat_range * margin_pct
        lonmin = min_lon - lon_range * margin_pct
        lonmax = max_lon + lon_range * margin_pct
        
        # Set projection params
        proj = 'lcc'  # Lambert Conformal Conic
        lat_0 = (latmin + latmax) / 2
        lon_0 = (lonmin + lonmax) / 2
        area_thresh = 1000
    
    # Create basemap
    m = Basemap(
        projection=proj,
        llcrnrlat=latmin, 
        urcrnrlat=latmax,
        llcrnrlon=lonmin, 
        urcrnrlon=lonmax,
        lat_0=lat_0, 
        lon_0=lon_0,
        resolution=resolution,
        area_thresh=area_thresh,
        ax=ax
    )
    
    # Draw map features
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.5)
    m.fillcontinents(color='lightgray', lake_color='lightblue', alpha=0.3)
    m.drawmapboundary(fill_color='lightblue', linewidth=0)
    
    # Draw additional features if requested
    if rivers:
        m.drawrivers(linewidth=0.1, color='blue')
    
    if states:
        m.drawstates(linewidth=0.2)
    
    if counties:
        m.drawcounties(linewidth=0.1)
    
    return fig, ax, m


def plot_station_map(
    station_df: pd.DataFrame,
    color_by: str = None,
    region: str = 'usa',
    title: str = 'Station Locations',
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a map of station locations.
    
    Parameters
    ----------
    station_df : pd.DataFrame
        DataFrame with station metadata, must include 'latitude' and 'longitude' columns
    color_by : str, optional
        Column to use for color coding stations
    region : str, optional
        Region to display: 'usa', 'east_coast', 'gulf_coast', 'west_coast', 'world'
    title : str, optional
        Map title
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Create base map
    fig, ax, m = create_base_map(station_df, region=region)
    
    # Return early if map creation failed
    if m is None:
        return fig
    
    # Extract lat/lon
    lats = station_df['latitude'].values
    lons = station_df['longitude'].values
    
    # Filter to valid coordinates
    valid = np.isfinite(lats) & np.isfinite(lons)
    lats = lats[valid]
    lons = lons[valid]
    
    # Convert lat/lon to map coordinates
    x, y = m(lons, lats)
    
    # Determine coloring
    if color_by and color_by in station_df.columns:
        # Filter to valid values for coloring
        values = station_df[color_by].values[valid]
        valid_colors = np.isfinite(values)
        
        if np.sum(valid_colors) > 0:
            # Create scatter plot with color coding
            sc = m.scatter(
                x[valid_colors], 
                y[valid_colors], 
                c=values[valid_colors],
                cmap=RISK_CMAP,
                s=60, 
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Create colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label(color_by)
            
            # Plot stations with no color data
            if np.sum(~valid_colors) > 0:
                m.scatter(
                    x[~valid_colors], 
                    y[~valid_colors], 
                    marker='o',
                    s=60, 
                    alpha=0.7,
                    facecolor='none',
                    edgecolor='gray',
                    linewidth=0.5
                )
        else:
            # If no valid color values, just show all stations
            m.scatter(x, y, marker='o', s=60, alpha=0.7, edgecolor='black', linewidth=0.5)
            
    else:
        # Simple scatter plot without color coding
        m.scatter(x, y, marker='o', s=60, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add station labels if not too many
    if len(lats) <= 20:
        for i, (xi, yi, code) in enumerate(zip(x, y, station_df['station_code'].values[valid])):
            txt = ax.text(xi, yi, code, fontsize=8, ha='center', va='bottom',
                         color='black', weight='bold')
            # Add outline to text for visibility
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground='white')
            ])
    
    # Set title
    ax.set_title(title)
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, "station_map")
        if color_by:
            filename += f"_{color_by}"
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_tier1_parameter_map(
    tier1_data: Dict,
    metadata_df: pd.DataFrame,
    parameter: str,
    variable: str = 'sea_level',
    region: str = 'usa',
    title: str = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a map of Tier 1 GPD parameter values across stations.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    metadata_df : pd.DataFrame
        DataFrame with station metadata, must include 'latitude' and 'longitude' columns
    parameter : str
        Parameter to visualize: 'shape', 'scale', 'threshold', 'return_level_100'
    variable : str, optional
        Variable to visualize: 'sea_level' or 'total_precipitation'
    region : str, optional
        Region to display: 'usa', 'east_coast', 'gulf_coast', 'west_coast', 'world'
    title : str, optional
        Map title. If None, auto-generated based on parameter and variable.
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract parameter values and combine with metadata
    stations = []
    values = []
    
    for station, data in tier1_data.items():
        if variable in data and 'gpd' in data[variable]:
            gpd = data[variable]['gpd']
            
            # Skip if error in GPD
            if 'error' in gpd:
                continue
                
            # Extract parameter value
            if parameter == 'shape':
                val = gpd.get('shape')
            elif parameter == 'scale':
                val = gpd.get('scale')
            elif parameter == 'threshold':
                val = data[variable].get('threshold')
            elif parameter == 'return_level_100':
                # Look for 100-year return level
                if 'return_levels' in data[variable]:
                    rl = data[variable]['return_levels']
                    if isinstance(rl, list):
                        for level in rl:
                            if isinstance(level, dict) and level.get('return_period') == 100:
                                val = level.get('return_level')
                                break
                        else:
                            # Not found
                            val = None
                    else:
                        val = None
                else:
                    val = None
            else:
                val = None
                
            # Skip if value not found
            if val is None:
                continue
                
            stations.append(station)
            values.append(val)
    
    # Skip if no values found
    if len(stations) == 0:
        logger.warning(f"No values found for {parameter} of {variable}")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, f"No values found for {parameter} of {variable}",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Create DataFrame with parameter values
    param_df = pd.DataFrame({
        'station_code': stations,
        'parameter': values
    })
    
    # Merge with metadata
    station_df = pd.merge(param_df, metadata_df, on='station_code', how='inner')
    
    # Skip if no stations with valid coordinates
    if len(station_df) == 0:
        logger.warning("No stations with valid metadata found")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, "No stations with valid metadata found",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Create base map
    fig, ax, m = create_base_map(station_df, region=region)
    
    # Return early if map creation failed
    if m is None:
        return fig
    
    # Extract lat/lon
    lats = station_df['latitude'].values
    lons = station_df['longitude'].values
    vals = station_df['parameter'].values
    
    # Filter to valid coordinates and values
    valid = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(vals)
    lats = lats[valid]
    lons = lons[valid]
    vals = vals[valid]
    
    # Skip if no valid data
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No valid data for visualization",
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Convert lat/lon to map coordinates
    x, y = m(lons, lats)
    
    # Select appropriate colormap and normalization
    if parameter == 'shape':
        # For shape parameter, use diverging colormap centered at 0
        cmap = 'RdBu_r'
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=min(-0.5, vals.min()), vmax=max(0.5, vals.max()))
    else:
        # For other parameters, use sequential colormap
        cmap = RISK_CMAP
        norm = None
    
    # Create scatter plot
    sc = m.scatter(x, y, c=vals, cmap=cmap, norm=norm,
                  s=100, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    
    # Set colorbar label
    if parameter == 'shape':
        cbar.set_label('Shape Parameter (ξ)')
    elif parameter == 'scale':
        cbar.set_label('Scale Parameter (σ)')
    elif parameter == 'threshold':
        cbar.set_label(f'{variable.replace("_", " ").title()} Threshold')
    elif parameter == 'return_level_100':
        cbar.set_label('100-year Return Level')
    else:
        cbar.set_label(parameter)
    
    # Add station labels if not too many
    if len(lats) <= 20:
        codes = [s for i, s in enumerate(station_df['station_code'].values[valid])]
        for i, (xi, yi, code) in enumerate(zip(x, y, codes)):
            txt = ax.text(xi, yi, code, fontsize=8, ha='center', va='bottom',
                         color='black', weight='bold')
            # Add outline to text for visibility
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground='white')
            ])
    
    # Set title
    if title is None:
        title = f"{parameter.replace('_', ' ').title()} of {variable.replace('_', ' ').title()}"
    ax.set_title(title)
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{variable}_{parameter}_map")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_tier2_parameter_map(
    tier2_data: Dict,
    metadata_df: pd.DataFrame,
    parameter: str,
    region: str = 'usa',
    title: str = None,
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a map of Tier 2 parameter values across stations.
    
    Parameters
    ----------
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    metadata_df : pd.DataFrame
        DataFrame with station metadata, must include 'latitude' and 'longitude' columns
    parameter : str
        Parameter to visualize: 'tau', 'cpr', 'tail_lower', 'tail_upper', 'rp_100'
    region : str, optional
        Region to display: 'usa', 'east_coast', 'gulf_coast', 'west_coast', 'world'
    title : str, optional
        Map title. If None, auto-generated based on parameter.
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract parameter values
    stations = []
    values = []
    
    for station, data in tier2_data.items():
        if 'tier2_analysis' not in data:
            continue
            
        analysis = data['tier2_analysis']
        
        if parameter == 'tau':
            # Derive tau from copula parameters
            val = None
            if 'copula' in analysis and 'parameters' in analysis['copula']:
                params = analysis['copula']['parameters']
                method = analysis['copula'].get('method', '')
                
                if method in ['Gaussian', 'StudentT'] and 'rho' in params:
                    rho = params['rho']
                    # Convert to tau using sin formula
                    val = 2 * np.arcsin(rho) / np.pi
                elif method == 'Gumbel' and 'theta' in params:
                    theta = params['theta']
                    # Tau = (theta-1)/theta for Gumbel
                    val = max(0, (theta - 1) / theta)
                elif method == 'Frank' and 'theta' in params:
                    # For Frank, use Spearman's rho as proxy (approximate)
                    theta = params['theta']
                    if abs(theta) < 1e-6:
                        val = 0
                    else:
                        # Very rough approximation
                        val = np.sign(theta) * min(0.8, abs(theta) / 10)
        
        elif parameter == 'cpr':
            # Get CPR at 0.99 level
            val = None
            if ('joint_exceedance' in analysis and 
                '0.99' in analysis['joint_exceedance']):
                val = analysis['joint_exceedance']['0.99'].get('cpr')
        
        elif parameter == 'tail_lower':
            # Get lower tail dependence
            val = None
            if 'tail_dependence' in analysis:
                val = analysis['tail_dependence'].get('lower')
        
        elif parameter == 'tail_upper':
            # Get upper tail dependence
            val = None
            if 'tail_dependence' in analysis:
                val = analysis['tail_dependence'].get('upper')
        
        elif parameter == 'rp_100':
            # Get 100-year AND return period
            val = None
            if ('joint_return_periods' in analysis and 
                '100' in analysis['joint_return_periods']):
                val = analysis['joint_return_periods']['100'].get('and_return_period')
        
        else:
            val = None
        
        # Skip if value not found
        if val is None:
            continue
            
        stations.append(station)
        values.append(val)
    
    # Skip if no values found
    if len(stations) == 0:
        logger.warning(f"No values found for {parameter}")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, f"No values found for {parameter}",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Create DataFrame with parameter values
    param_df = pd.DataFrame({
        'station_code': stations,
        'parameter': values
    })
    
    # Merge with metadata
    station_df = pd.merge(param_df, metadata_df, on='station_code', how='inner')
    
    # Skip if no stations with valid coordinates
    if len(station_df) == 0:
        logger.warning("No stations with valid metadata found")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, "No stations with valid metadata found",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Create base map
    fig, ax, m = create_base_map(station_df, region=region)
    
    # Return early if map creation failed
    if m is None:
        return fig
    
    # Extract lat/lon
    lats = station_df['latitude'].values
    lons = station_df['longitude'].values
    vals = station_df['parameter'].values
    
    # Filter to valid coordinates and values
    valid = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(vals)
    lats = lats[valid]
    lons = lons[valid]
    vals = vals[valid]
    
    # Skip if no valid data
    if len(vals) == 0:
        ax.text(0.5, 0.5, "No valid data for visualization",
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Convert lat/lon to map coordinates
    x, y = m(lons, lats)
    
    # Select appropriate colormap and normalization
    if parameter == 'tau':
        # For tau, use diverging colormap centered at 0
        cmap = 'RdBu_r'
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=min(-0.5, vals.min()), vmax=max(0.5, vals.max()))
    elif parameter == 'cpr':
        # For CPR, use CPR colormap with reference at 1
        cmap = CPR_CMAP
        vmin = min(0.5, vals.min())
        vmax = max(3.0, vals.max())
        # Center at 1 (independence)
        norm = colors.TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    elif parameter in ['tail_lower', 'tail_upper']:
        # For tail dependence, use risk colormap with range [0, 1]
        cmap = RISK_CMAP
        norm = colors.Normalize(vmin=0, vmax=1)
    elif parameter == 'rp_100':
        # For return period, use log scale
        cmap = RISK_CMAP
        norm = colors.LogNorm(vmin=max(10, vals.min()), vmax=max(1000, vals.max()))
    else:
        # Default
        cmap = RISK_CMAP
        norm = None
    
    # Create scatter plot
    sc = m.scatter(x, y, c=vals, cmap=cmap, norm=norm,
                  s=100, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    
    # Set colorbar label
    if parameter == 'tau':
        cbar.set_label("Kendall's Tau (τ)")
    elif parameter == 'cpr':
        cbar.set_label('Conditional Probability Ratio (99%)')
    elif parameter == 'tail_lower':
        cbar.set_label('Lower Tail Dependence')
    elif parameter == 'tail_upper':
        cbar.set_label('Upper Tail Dependence')
    elif parameter == 'rp_100':
        cbar.set_label('100-year Joint Return Period')
    else:
        cbar.set_label(parameter)
    
    # Add station labels if not too many
    if len(lats) <= 20:
        codes = [s for i, s in enumerate(station_df['station_code'].values[valid])]
        for i, (xi, yi, code) in enumerate(zip(x, y, codes)):
            txt = ax.text(xi, yi, code, fontsize=8, ha='center', va='bottom',
                         color='black', weight='bold')
            # Add outline to text for visibility
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground='white')
            ])
    
    # Set title
    if title is None:
        if parameter == 'tau':
            title = "Kendall's Tau (τ) across Stations"
        elif parameter == 'cpr':
            title = "Conditional Probability Ratio (99%) across Stations"
        elif parameter == 'tail_lower':
            title = "Lower Tail Dependence across Stations"
        elif parameter == 'tail_upper':
            title = "Upper Tail Dependence across Stations"
        elif parameter == 'rp_100':
            title = "100-year Joint Return Period across Stations"
        else:
            title = f"{parameter.replace('_', ' ').title()} across Stations"
    ax.set_title(title)
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, f"{parameter}_map")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def plot_compound_flood_risk_map(
    tier1_data: Dict,
    tier2_data: Dict,
    metadata_df: pd.DataFrame,
    region: str = 'usa',
    title: str = "Compound Flood Risk",
    output_dir: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a map of compound flood risk combining multiple metrics.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    metadata_df : pd.DataFrame
        DataFrame with station metadata, must include 'latitude' and 'longitude' columns
    region : str, optional
        Region to display: 'usa', 'east_coast', 'gulf_coast', 'west_coast', 'world'
    title : str, optional
        Map title
    output_dir : str, optional
        Directory to save figure
    show : bool, optional
        Whether to display the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Extract key parameters for risk assessment
    stations = []
    cpr_values = []  # Conditional Probability Ratio
    tau_values = []  # Kendall's tau
    sl_rl100_values = []  # 100-year Sea Level Return Level
    
    for station in tier1_data.keys():
        if station not in tier2_data:
            continue
            
        t1_data = tier1_data[station]
        t2_data = tier2_data[station]
        
        # Extract CPR
        cpr = None
        if ('joint' in t1_data and 
            'empirical' in t1_data['joint']):
            cpr = t1_data['joint']['empirical'].get('cpr')
        
        # Extract tau
        tau = None
        if ('tier2_analysis' in t2_data and 
            'copula' in t2_data['tier2_analysis'] and
            'parameters' in t2_data['tier2_analysis']['copula']):
            
            params = t2_data['tier2_analysis']['copula']['parameters']
            method = t2_data['tier2_analysis']['copula'].get('method', '')
            
            if method in ['Gaussian', 'StudentT'] and 'rho' in params:
                rho = params['rho']
                # Convert to tau using sin formula
                tau = 2 * np.arcsin(rho) / np.pi
            elif method == 'Gumbel' and 'theta' in params:
                theta = params['theta']
                # Tau = (theta-1)/theta for Gumbel
                tau = max(0, (theta - 1) / theta)
            elif method == 'Frank' and 'theta' in params:
                # For Frank, use Spearman's rho as proxy (approximate)
                theta = params['theta']
                if abs(theta) < 1e-6:
                    tau = 0
                else:
                    # Very rough approximation
                    tau = np.sign(theta) * min(0.8, abs(theta) / 10)
        
        # Extract 100-year sea level return level
        sl_rl100 = None
        if ('sea_level' in t1_data and 
            'return_levels' in t1_data['sea_level']):
            rl = t1_data['sea_level']['return_levels']
            if isinstance(rl, list):
                for level in rl:
                    if isinstance(level, dict) and level.get('return_period') == 100:
                        sl_rl100 = level.get('return_level')
                        break
        
        # Only add station if we have at least CPR or tau
        if cpr is not None or tau is not None:
            stations.append(station)
            cpr_values.append(cpr)
            tau_values.append(tau)
            sl_rl100_values.append(sl_rl100)
    
    # Skip if no values found
    if len(stations) == 0:
        logger.warning("No stations with both Tier 1 and Tier 2 data found")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, "No stations with both Tier 1 and Tier 2 data found",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Create DataFrame with parameters
    param_df = pd.DataFrame({
        'station_code': stations,
        'cpr': cpr_values,
        'tau': tau_values,
        'sl_rl100': sl_rl100_values
    })
    
    # Calculate compound risk score
    # We'll use a simple weighted average of normalized metrics
    def normalize(x):
        if isinstance(x, np.ndarray):
            x_norm = np.zeros_like(x, dtype=float)
            valid = np.isfinite(x)
            if np.sum(valid) > 0:
                x_valid = x[valid]
                x_min, x_max = np.min(x_valid), np.max(x_valid)
                if x_max > x_min:
                    x_norm[valid] = (x_valid - x_min) / (x_max - x_min)
            return x_norm
        else:
            return np.nan
    
    # Normalize CPR relative to independence (CPR=1)
    cpr_norm = np.array(cpr_values, dtype=float)
    valid = np.isfinite(cpr_norm)
    if np.sum(valid) > 0:
        # Scale such that CPR=1 (independence) maps to 0
        # and max CPR maps to 1
        cpr_norm[valid] = np.maximum(0, cpr_norm[valid] - 1) / np.maximum(1, np.max(cpr_norm[valid]) - 1)
    
    # Normalize tau to [0, 1] range (only consider positive dependence)
    tau_norm = np.array(tau_values, dtype=float)
    valid = np.isfinite(tau_norm)
    if np.sum(valid) > 0:
        tau_norm[valid] = np.maximum(0, tau_norm[valid]) / np.maximum(0.1, np.max(tau_norm[valid]))
    
    # Normalize sea level return level
    sl_rl100_norm = normalize(np.array(sl_rl100_values, dtype=float))
    
    # Calculate risk score (weighted average of available metrics)
    weights = {'cpr': 0.4, 'tau': 0.4, 'sl_rl100': 0.2}
    risk_score = np.zeros(len(stations))
    weight_sum = np.zeros(len(stations))
    
    # Add weighted CPR
    valid = np.isfinite(cpr_norm)
    if np.sum(valid) > 0:
        risk_score[valid] += weights['cpr'] * cpr_norm[valid]
        weight_sum[valid] += weights['cpr']
    
    # Add weighted tau
    valid = np.isfinite(tau_norm)
    if np.sum(valid) > 0:
        risk_score[valid] += weights['tau'] * tau_norm[valid]
        weight_sum[valid] += weights['tau']
    
    # Add weighted sea level return level
    valid = np.isfinite(sl_rl100_norm)
    if np.sum(valid) > 0:
        risk_score[valid] += weights['sl_rl100'] * sl_rl100_norm[valid]
        weight_sum[valid] += weights['sl_rl100']
    
    # Normalize by weight sum
    valid = weight_sum > 0
    if np.sum(valid) > 0:
        risk_score[valid] /= weight_sum[valid]
    
    # Add risk score to DataFrame
    param_df['risk_score'] = risk_score
    
    # Merge with metadata
    station_df = pd.merge(param_df, metadata_df, on='station_code', how='inner')
    
    # Skip if no stations with valid coordinates
    if len(station_df) == 0:
        logger.warning("No stations with valid metadata found")
        fig, ax = plt.subplots(figsize=FIG_SIZES['medium'])
        ax.text(0.5, 0.5, "No stations with valid metadata found",
               ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    # Create base map
    fig, ax, m = create_base_map(station_df, region=region)
    
    # Return early if map creation failed
    if m is None:
        return fig
    
    # Extract lat/lon
    lats = station_df['latitude'].values
    lons = station_df['longitude'].values
    risk = station_df['risk_score'].values
    
    # Filter to valid coordinates and values
    valid = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(risk)
    lats = lats[valid]
    lons = lons[valid]
    risk = risk[valid]
    
    # Skip if no valid data
    if len(risk) == 0:
        ax.text(0.5, 0.5, "No valid data for visualization",
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Convert lat/lon to map coordinates
    x, y = m(lons, lats)
    
    # Create scatter plot
    sc = m.scatter(x, y, c=risk, cmap=RISK_CMAP,
                  s=150, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Compound Flood Risk Score')
    
    # Add station labels if not too many
    if len(lats) <= 20:
        codes = [s for i, s in enumerate(station_df['station_code'].values[valid])]
        for i, (xi, yi, code) in enumerate(zip(x, y, codes)):
            txt = ax.text(xi, yi, code, fontsize=8, ha='center', va='bottom',
                         color='black', weight='bold')
            # Add outline to text for visibility
            txt.set_path_effects([
                PathEffects.withStroke(linewidth=2, foreground='white')
            ])
    
    # Add legend explaining the risk score
    legend_text = (
        "Risk Score Components:\n"
        f"- CPR: {weights['cpr']*100:.0f}%\n"
        f"- Kendall's τ: {weights['tau']*100:.0f}%\n"
        f"- 100yr Return Level: {weights['sl_rl100']*100:.0f}%"
    )
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=8,
           va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Set title
    ax.set_title(title)
    
    # Save if output directory is provided
    if output_dir:
        filename = os.path.join(output_dir, "compound_flood_risk_map")
        save_figure(fig, filename)
    
    if not show:
        plt.close(fig)
    
    return fig


def create_spatial_visualizations(
    tier1_data: Dict,
    tier2_data: Dict,
    metadata_df: pd.DataFrame,
    output_dir: str,
    regions: List[str] = ['usa'],
    show: bool = False
) -> None:
    """
    Create a set of spatial visualizations for compound flooding analysis.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    metadata_df : pd.DataFrame
        DataFrame with station metadata, must include 'latitude' and 'longitude' columns
    output_dir : str
        Directory to save figures
    regions : List[str], optional
        Regions to create maps for: 'usa', 'east_coast', 'gulf_coast', 'west_coast', 'world'
    show : bool, optional
        Whether to display the figures
    """
    # Set publication style
    set_publication_style()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip if no valid metadata
    if len(metadata_df) == 0:
        logger.warning("No valid metadata for creating maps")
        return
    
    logger.info(f"Creating spatial visualizations for {len(regions)} regions")
    
    # Create station maps
    for region in regions:
        logger.info(f"Generating maps for region: {region}")
        
        # Create station map
        plot_station_map(
            metadata_df,
            region=region,
            title=f"Station Locations - {region}",
            output_dir=output_dir,
            show=show
        )
        
        # Create Tier 1 parameter maps for sea level
        for param in ['shape', 'scale', 'threshold', 'return_level_100']:
            plot_tier1_parameter_map(
                tier1_data,
                metadata_df,
                parameter=param,
                variable='sea_level',
                region=region,
                output_dir=output_dir,
                show=show
            )
        
        # Create Tier 1 parameter maps for precipitation
        for param in ['shape', 'scale', 'threshold', 'return_level_100']:
            plot_tier1_parameter_map(
                tier1_data,
                metadata_df,
                parameter=param,
                variable='total_precipitation',
                region=region,
                output_dir=output_dir,
                show=show
            )
        
        # Create Tier 2 parameter maps
        for param in ['tau', 'cpr', 'tail_lower', 'tail_upper', 'rp_100']:
            plot_tier2_parameter_map(
                tier2_data,
                metadata_df,
                parameter=param,
                region=region,
                output_dir=output_dir,
                show=show
            )
        
        # Create compound flood risk map
        plot_compound_flood_risk_map(
            tier1_data,
            tier2_data,
            metadata_df,
            region=region,
            output_dir=output_dir,
            show=show
        )
    
    logger.info(f"Spatial visualizations completed and saved to {output_dir}")


if __name__ == "__main__":
    # Basic test of the module
    import sys
    from compound_flooding.visualization.base import (
        load_tier1_results, 
        load_tier2_results, 
        load_station_metadata,
        create_output_dirs
    )
    
    print("Testing maps module...")
    
    # Create output directories
    dirs = create_output_dirs('outputs/plots_test')
    
    # Check if we have metadata and at least one data file
    if len(sys.argv) > 2:
        metadata_file = sys.argv[1]
        data_file = sys.argv[2]
        
        try:
            # Load metadata
            metadata_df = load_station_metadata(metadata_file)
            print(f"Loaded metadata for {len(metadata_df)} stations")
            
            # Test station map
            plot_station_map(
                metadata_df,
                region='usa',
                output_dir=dirs['maps']
            )
            
            # Load some tier1 or tier2 data if available
            if 'tier1' in data_file:
                tier1_data = load_tier1_results(os.path.dirname(data_file))
                print(f"Loaded Tier 1 data for {len(tier1_data)} stations")
                
                # Test Tier 1 parameter map
                plot_tier1_parameter_map(
                    tier1_data,
                    metadata_df,
                    parameter='shape',
                    variable='sea_level',
                    output_dir=dirs['maps']
                )
                
            elif 'tier2' in data_file:
                tier2_data = load_tier2_results(os.path.dirname(data_file))
                print(f"Loaded Tier 2 data for {len(tier2_data)} stations")
                
                # Test Tier 2 parameter map
                plot_tier2_parameter_map(
                    tier2_data,
                    metadata_df,
                    parameter='tau',
                    output_dir=dirs['maps']
                )
            
            print(f"Test complete. Check outputs in {dirs['maps']}")
            
        except Exception as e:
            print(f"Error testing with provided files: {e}")
    else:
        print("No test data provided. Please provide metadata and Tier 1 or Tier 2 output file as arguments.")