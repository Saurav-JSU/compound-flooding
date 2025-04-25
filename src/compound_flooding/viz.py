"""
Visualization utilities for compound flooding analysis.

This module provides functions for creating plots and visualizations
of compound flooding analysis results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import glob
import re
from scipy.stats import genpareto
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Type aliases
PathLike = Union[str, Path]
ResultsType = Dict[str, Any]


class CompoundFloodingVisualizer:
    """
    Class for creating visualizations of compound flooding analysis.
    
    This class provides methods for visualizing results from the
    different tiers of compound flooding analysis.
    """
    
    def __init__(
        self,
        results_dir: PathLike = Path("outputs"),
        output_dir: PathLike = Path("outputs/plots"),
        station_metadata: Optional[pd.DataFrame] = None,
        show_plots: bool = False,
        dpi: int = 150,
        color_palette: str = "viridis"
    ):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        results_dir : PathLike, optional
            Directory containing analysis results.
        output_dir : PathLike, optional
            Directory to save visualizations.
        station_metadata : Optional[pd.DataFrame], optional
            DataFrame containing station metadata.
        show_plots : bool, optional
            Whether to display plots in addition to saving. Default is False.
        dpi : int, optional
            Resolution for saved plots. Default is 150.
        color_palette : str, optional
            Matplotlib colormap name for plots. Default is "viridis".
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.station_metadata = station_metadata
        self.show_plots = show_plots
        self.dpi = dpi
        self.color_palette = color_palette
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_tier1_results(self, station_code: Optional[str] = None) -> pd.DataFrame:
        """
        Load Tier 1 results from parquet files.
        
        Parameters
        ----------
        station_code : Optional[str], optional
            Station code to load. If None, load all stations.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing Tier 1 results.
        """
        tier1_dir = self.results_dir / "tier1"
        
        if station_code:
            # Load results for a specific station
            file_path = tier1_dir / f"{station_code}_tier1_results.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"Results not found for station {station_code}: {file_path}")
            
            return pd.read_parquet(file_path)
        else:
            # Load results for all stations
            file_paths = sorted(tier1_dir.glob("*_tier1_results.parquet"))
            
            if not file_paths:
                raise FileNotFoundError(f"No Tier 1 results found in {tier1_dir}")
            
            # Load and concatenate all results
            dfs = []
            for path in file_paths:
                try:
                    df = pd.read_parquet(path)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            
            if not dfs:
                raise ValueError("No results could be loaded")
            
            return pd.concat(dfs, ignore_index=True)
    
    def plot_station_map(
        self, 
        variable: str = "conditional_probability_ratio",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        log_scale: bool = False,
        highlight_significant: bool = True,
        significance_threshold: float = 1.5
    ) -> plt.Figure:
        """
        Create a map of stations colored by a Tier 1 statistic.
        
        Parameters
        ----------
        variable : str, optional
            Variable to use for coloring. Default is "conditional_probability_ratio".
        min_val : Optional[float], optional
            Minimum value for color scale. If None, use data minimum.
        max_val : Optional[float], optional
            Maximum value for color scale. If None, use data maximum.
        log_scale : bool, optional
            Whether to use a logarithmic color scale. Default is False.
        highlight_significant : bool, optional
            Whether to highlight statistically significant results. Default is True.
        significance_threshold : float, optional
            Threshold for significance. Default is 1.5.
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        if not CARTOPY_AVAILABLE:
            raise ImportError("Cartopy is required for creating maps. Install with: pip install cartopy")
        
        # Load all Tier 1 results
        try:
            results_df = self.load_tier1_results()
        except FileNotFoundError:
            raise ValueError("No Tier 1 results found. Run Tier 1 analysis first.")
        
        # Extract the variable of interest
        if variable == "conditional_probability_ratio":
            if "joint_exceedance_results.conditional_probability_ratio" in results_df.columns:
                var_name = "joint_exceedance_results.conditional_probability_ratio"
            else:
                raise ValueError(f"Variable '{variable}' not found in results")
        else:
            # Look for the variable in the dataframe
            matching_cols = [col for col in results_df.columns if variable in col]
            if not matching_cols:
                raise ValueError(f"Variable '{variable}' not found in results")
            var_name = matching_cols[0]
        
        # Extract the values for mapping
        map_data = pd.DataFrame({
            'station_code': results_df['station_code'],
            'latitude': results_df['metadata.latitude'],
            'longitude': results_df['metadata.longitude'],
            'value': results_df[var_name]
        })
        
        # Remove NaN values
        map_data = map_data.dropna(subset=['latitude', 'longitude', 'value'])
        
        if len(map_data) == 0:
            raise ValueError("No valid data points for mapping")
        
        # Set color scale limits
        if min_val is None:
            min_val = map_data['value'].min()
        
        if max_val is None:
            max_val = map_data['value'].max()
        
        # Create the map
        fig = plt.figure(figsize=(12, 8))
        
        # Use PlateCarree projection for simplicity
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Add coastlines and country borders
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.25)
        
        # Add natural features
        ax.add_feature(cfeature.OCEAN, alpha=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.5)
        
        # Set map extent based on data
        buffer = 2.0  # Degrees of padding around data extent
        lon_min = map_data['longitude'].min() - buffer
        lon_max = map_data['longitude'].max() + buffer
        lat_min = map_data['latitude'].min() - buffer
        lat_max = map_data['latitude'].max() + buffer
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Create colormap
        cmap = plt.cm.get_cmap(self.color_palette)
        
        # Create scatter plot
        if log_scale and min_val > 0:
            norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
        else:
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        # Plot all points
        sc = ax.scatter(
            map_data['longitude'], 
            map_data['latitude'],
            c=map_data['value'],
            cmap=cmap,
            norm=norm,
            edgecolor='black',
            s=80,
            alpha=0.8,
            transform=ccrs.PlateCarree()
        )
        
        # Highlight significant points if requested
        if highlight_significant and variable == "conditional_probability_ratio":
            significant = map_data[map_data['value'] > significance_threshold]
            if len(significant) > 0:
                ax.scatter(
                    significant['longitude'], 
                    significant['latitude'],
                    facecolor='none',
                    edgecolor='red',
                    s=140,
                    linewidth=2,
                    alpha=0.8,
                    transform=ccrs.PlateCarree()
                )
        
        # Add a colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        
        # Set title and labels
        if variable == "conditional_probability_ratio":
            title = "Conditional Probability Ratio (CPR) for Compound Flooding"
            cbar.set_label('CPR (>1 indicates dependence)')
        else:
            title = f"{variable} by Station"
            cbar.set_label(variable)
        
        plt.title(title, fontsize=14)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Save the figure
        output_path = self.output_dir / f"station_map_{variable.replace('.', '_')}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_return_level_curves(
        self, 
        station_codes: List[str],
        variable: str = "sea_level",
        return_periods: Optional[List[float]] = None,
        log_scale: bool = True
    ) -> plt.Figure:
        """
        Plot return level curves for multiple stations.
        
        Parameters
        ----------
        station_codes : List[str]
            List of station codes to include.
        variable : str, optional
            Variable for which to plot return levels. Default is "sea_level".
        return_periods : Optional[List[float]], optional
            Return periods to plot. If None, use a range from 1 to 1000 years.
        log_scale : bool, optional
            Whether to use a logarithmic x-axis. Default is True.
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        if return_periods is None:
            # Default return periods: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 years
            return_periods = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        
        # Results for each station
        station_results = []
        
        for station_code in station_codes:
            try:
                # Load results for this station
                results_df = self.load_tier1_results(station_code)
                
                # Extract extreme value results for the variable
                var_key = f"extreme_value_results.{variable}"
                
                if var_key + ".shape_parameter" not in results_df.columns:
                    print(f"Warning: {variable} results not found for station {station_code}")
                    continue
                
                # Extract GPD parameters
                shape = results_df[var_key + ".shape_parameter"].iloc[0]
                scale = results_df[var_key + ".scale_parameter"].iloc[0]
                threshold = results_df[var_key + ".threshold"].iloc[0]
                n_peaks = results_df[var_key + ".n_peaks"].iloc[0]
                station_name = results_df["metadata.station_name"].iloc[0]
                
                # Get total sample size (approximation)
                n_points = results_df["time_coverage.n_hours"].iloc[0]
                
                # Add to results list
                station_results.append({
                    "station_code": station_code,
                    "station_name": station_name,
                    "shape": shape,
                    "scale": scale,
                    "threshold": threshold,
                    "n_peaks": n_peaks,
                    "n_points": n_points
                })
            except Exception as e:
                print(f"Error loading results for station {station_code}: {e}")
        
        if not station_results:
            raise ValueError("No valid results found for any of the specified stations")
        
        # Create figure for plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot return level curves for each station
        for result in station_results:
            # Calculate rate of exceedance
            lambda_rate = result["n_peaks"] / result["n_points"]
            annual_rate = lambda_rate * 24 * 365.25  # Assuming hourly data
            
            # Calculate return levels for a range of return periods
            years = np.linspace(1, max(return_periods), 1000)
            
            if abs(result["shape"]) < 1e-6:
                # For shape ≈ 0, use exponential formula
                return_levels = result["threshold"] + result["scale"] * np.log(years * annual_rate)
            else:
                # For non-zero shape, use GPD formula
                return_levels = result["threshold"] + (result["scale"] / result["shape"]) * ((years * annual_rate) ** result["shape"] - 1)
            
            # Plot the return level curve
            line, = ax.plot(years, return_levels, label=f"{result['station_code']} - {result['station_name']}")
            
            # Add markers for specific return periods
            for period in return_periods:
                if period <= max(years):
                    if abs(result["shape"]) < 1e-6:
                        level = result["threshold"] + result["scale"] * np.log(period * annual_rate)
                    else:
                        level = result["threshold"] + (result["scale"] / result["shape"]) * ((period * annual_rate) ** result["shape"] - 1)
                    
                    ax.scatter(period, level, color=line.get_color(), s=40, zorder=5)
        
        # Set axis properties
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_xlabel('Return Period (years)', fontsize=12)
        ax.set_ylabel(f'{variable} (units)', fontsize=12)
        ax.set_title(f'Return Level Curves for {variable}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='best', fontsize=10)
        
        # Save the figure
        output_path = self.output_dir / f"return_level_curves_{variable}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_conditional_probability_comparison(
        self, 
        variable_pairs: List[Tuple[str, str]] = [("sea_level", "total_precipitation")]
    ) -> plt.Figure:
        """
        Plot conditional probability comparison across stations.
        
        Parameters
        ----------
        variable_pairs : List[Tuple[str, str]], optional
            List of variable pairs to compare. Default is [("sea_level", "total_precipitation")].
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        # Load all Tier 1 results
        try:
            results_df = self.load_tier1_results()
        except FileNotFoundError:
            raise ValueError("No Tier 1 results found. Run Tier 1 analysis first.")
        
        # Calculate the number of subplots needed
        n_pairs = len(variable_pairs)
        
        # Create figure for plotting
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 8), sharey=True)
        
        # Handle single pair case
        if n_pairs == 1:
            axes = [axes]
        
        # Plot for each variable pair
        for i, (var1, var2) in enumerate(variable_pairs):
            ax = axes[i]
            
            # Extract conditional probabilities
            col1 = f"joint_exceedance_results.conditional_probabilities.{var1}_given_{var2}"
            col2 = f"joint_exceedance_results.conditional_probabilities.{var2}_given_{var1}"
            
            if col1 not in results_df.columns or col2 not in results_df.columns:
                ax.text(0.5, 0.5, f"Data not found for {var1} and {var2}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                continue
            
            # Extract the data
            data = pd.DataFrame({
                'station_code': results_df['station_code'],
                'station_name': results_df['metadata.station_name'],
                f"{var1}_given_{var2}": results_df[col1],
                f"{var2}_given_{var1}": results_df[col2],
                'latitude': results_df['metadata.latitude']
            })
            
            # Sort by latitude for geographical ordering
            data = data.sort_values('latitude', ascending=False)
            
            # Remove stations with missing values
            data = data.dropna(subset=[f"{var1}_given_{var2}", f"{var2}_given_{var1}"])
            
            if len(data) == 0:
                ax.text(0.5, 0.5, "No valid data points",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                continue
            
            # Number of stations
            n_stations = len(data)
            
            # Set up the plot
            x = np.arange(n_stations)
            width = 0.35
            
            # Plot the bars
            ax.bar(x - width/2, data[f"{var1}_given_{var2}"], width, label=f"P({var1}|{var2})")
            ax.bar(x + width/2, data[f"{var2}_given_{var1}"], width, label=f"P({var2}|{var1})")
            
            # Customize the plot
            ax.set_ylabel('Conditional Probability')
            ax.set_title(f'{var1} vs {var2} Conditional Probabilities')
            ax.set_xticks(x)
            ax.set_xticklabels(data['station_code'], rotation=90, fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits
            ax.set_ylim(0, 1.0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / "conditional_probability_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_joint_exceedance_statistics(
        self,
        variable_pair: Tuple[str, str] = ("sea_level", "total_precipitation")
    ) -> plt.Figure:
        """
        Plot joint exceedance statistics across stations.
        
        Parameters
        ----------
        variable_pair : Tuple[str, str], optional
            Variable pair to plot. Default is ("sea_level", "total_precipitation").
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        var1, var2 = variable_pair
        
        # Load all Tier 1 results
        try:
            results_df = self.load_tier1_results()
        except FileNotFoundError:
            raise ValueError("No Tier 1 results found. Run Tier 1 analysis first.")
        
        # Extract relevant statistics
        cols = [
            "station_code",
            "metadata.station_name",
            "metadata.latitude",
            f"joint_exceedance_results.exceedance_probabilities.{var1}",
            f"joint_exceedance_results.exceedance_probabilities.{var2}",
            "joint_exceedance_results.exceedance_probabilities.joint_empirical",
            "joint_exceedance_results.exceedance_probabilities.joint_independence",
            "joint_exceedance_results.conditional_probability_ratio",
            f"joint_exceedance_results.event_counts.{var1}",
            f"joint_exceedance_results.event_counts.{var2}",
            "joint_exceedance_results.event_counts.joint"
        ]
        
        # Check if all columns exist
        missing_cols = [col for col in cols if col not in results_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in results: {missing_cols}")
        
        # Extract the data
        data = results_df[cols].copy()
        
        # Rename columns for clarity
        data.columns = [
            "station_code", "station_name", "latitude",
            f"{var1}_prob", f"{var2}_prob", "joint_empirical", "joint_independence",
            "cpr", f"{var1}_events", f"{var2}_events", "joint_events"
        ]
        
        # Sort by latitude for geographical ordering
        data = data.sort_values('latitude', ascending=False)
        
        # Remove stations with missing values
        data = data.dropna(subset=["cpr", "joint_empirical", "joint_independence"])
        
        if len(data) == 0:
            raise ValueError("No valid data points for plotting")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1.5]})
        
        # Plot CPR values
        ax = axes[0]
        x = np.arange(len(data))
        bars = ax.bar(x, data["cpr"], color='skyblue')
        
        # Add a horizontal line at CPR = 1 (independence)
        ax.axhline(y=1, color='r', linestyle='--', label='Independence (CPR=1)')
        
        # Customize the plot
        ax.set_ylabel('Conditional Probability Ratio (CPR)')
        ax.set_title('Compound Flooding Dependence by Station')
        ax.set_xticks(x)
        ax.set_xticklabels(data['station_code'], rotation=90, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Plot empirical vs independence joint probabilities
        ax = axes[1]
        x = np.arange(len(data))
        width = 0.35
        
        # Plot the bars
        ax.bar(x - width/2, data["joint_empirical"], width, label="Empirical Joint Probability", color='green')
        ax.bar(x + width/2, data["joint_independence"], width, label="Independence Joint Probability", color='orange')
        
        # Customize the plot
        ax.set_ylabel('Joint Exceedance Probability')
        ax.set_xlabel('Station')
        ax.set_title('Joint Exceedance Probabilities: Empirical vs. Independence Assumption')
        ax.set_xticks(x)
        ax.set_xticklabels(data['station_code'], rotation=90, fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / f"joint_exceedance_statistics_{var1}_{var2}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_lag_analysis(
        self,
        variable_pair: Tuple[str, str] = ("sea_level", "total_precipitation"),
        top_n_stations: int = 10
    ) -> plt.Figure:
        """
        Plot lag analysis results.
        
        Parameters
        ----------
        variable_pair : Tuple[str, str], optional
            Variable pair to plot. Default is ("sea_level", "total_precipitation").
        top_n_stations : int, optional
            Number of stations with highest CPR to include. Default is 10.
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        var1, var2 = variable_pair
        
        # Load all Tier 1 results
        try:
            results_df = self.load_tier1_results()
        except FileNotFoundError:
            raise ValueError("No Tier 1 results found. Run Tier 1 analysis first.")
        
        # Extract relevant columns
        cols = [
            "station_code",
            "metadata.station_name",
            "joint_exceedance_results.conditional_probability_ratio",
            "joint_exceedance_results.lag_analysis.mean_lag",
            "joint_exceedance_results.lag_analysis.median_lag",
            "joint_exceedance_results.lag_analysis.lag_std",
            "joint_exceedance_results.lag_analysis.n_lags"
        ]
        
        # Check if all columns exist
        missing_cols = [col for col in cols if col not in results_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in results: {missing_cols}")
        
        # Extract the data
        data = results_df[cols].copy()
        
        # Rename columns for clarity
        data.columns = [
            "station_code", "station_name", "cpr", 
            "mean_lag", "median_lag", "lag_std", "n_lags"
        ]
        
        # Remove stations with missing values or no lags
        data = data.dropna(subset=["cpr", "mean_lag"])
        data = data[data["n_lags"] > 0]
        
        if len(data) == 0:
            raise ValueError("No valid lag data for plotting")
        
        # Sort by CPR and take top N stations
        data = data.sort_values('cpr', ascending=False).head(top_n_stations)
        
        # Create figure for plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot mean lag by station
        ax = axes[0]
        y_pos = np.arange(len(data))
        
        # Plot with error bars (standard deviation)
        ax.barh(y_pos, data['mean_lag'], xerr=data['lag_std'], color='skyblue')
        
        # Add vertical line at 0 (simultaneous)
        ax.axvline(x=0, color='r', linestyle='--', label='Simultaneous')
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data['station_code'])
        ax.invert_yaxis()  # Stations with highest CPR at the top
        ax.set_xlabel(f'Mean Lag ({var2} relative to {var1}) [hours]')
        ax.set_title(f'Mean Lag Between {var1} and {var2} Extremes')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()
        
        # Plot lag distribution bins (aggregated across stations)
        ax = axes[1]
        
        # Extract lag bin columns if available
        lag_bin_cols = [col for col in results_df.columns if "joint_exceedance_results.lag_analysis.lag_bins" in col]
        
        if lag_bin_cols:
            # Extract bin counts for each station
            lag_bins = results_df.loc[results_df['station_code'].isin(data['station_code']), lag_bin_cols].sum()
            
            # Define bin edges and labels
            bin_edges = [-24, -12, -6, -3, 0, 3, 6, 12, 24]
            bin_labels = [f"{bin_edges[i]} to {bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
            
            # Get bin counts
            bin_values = []
            for i, label in enumerate(bin_labels):
                if f"joint_exceedance_results.lag_analysis.lag_bins.{'neg' if bin_edges[i] < 0 else 'pos'}_{abs(bin_edges[i])}_{abs(bin_edges[i+1])}" in lag_bin_cols:
                    key = f"joint_exceedance_results.lag_analysis.lag_bins.{'neg' if bin_edges[i] < 0 else 'pos'}_{abs(bin_edges[i])}_{abs(bin_edges[i+1])}"
                    bin_values.append(lag_bins[key])
                else:
                    bin_values.append(0)
            
            # Plot histogram
            ax.bar(bin_labels, bin_values, color='lightgreen')
            ax.set_xlabel('Lag Range [hours]')
            ax.set_ylabel('Number of Events')
            ax.set_title('Distribution of Lags Between Extreme Events')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, "Lag bin data not available",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = self.output_dir / f"lag_analysis_{var1}_{var2}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_summary_dashboard(
        self, 
        variable_pair: Tuple[str, str] = ("sea_level", "total_precipitation"),
        station_code: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a summary dashboard of Tier 1 results.
        
        Parameters
        ----------
        variable_pair : Tuple[str, str], optional
            Variable pair to plot. Default is ("sea_level", "total_precipitation").
        station_code : Optional[str], optional
            Station code to plot. If None, use the station with highest CPR.
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        var1, var2 = variable_pair
        
        # Load Tier 1 results
        try:
            if station_code:
                results_df = self.load_tier1_results(station_code)
            else:
                # Load all results and find station with highest CPR
                all_results = self.load_tier1_results()
                
                cpr_col = "joint_exceedance_results.conditional_probability_ratio"
                if cpr_col not in all_results.columns:
                    raise ValueError("CPR column not found in results")
                
                # Sort by CPR and take the highest
                all_results = all_results.sort_values(cpr_col, ascending=False)
                
                if len(all_results) == 0:
                    raise ValueError("No valid results found")
                
                station_code = all_results.iloc[0]["station_code"]
                results_df = all_results[all_results["station_code"] == station_code]
        except FileNotFoundError:
            raise ValueError("No Tier 1 results found. Run Tier 1 analysis first.")
        
        if len(results_df) == 0:
            raise ValueError(f"No results found for station {station_code}")
        
        # Extract key information
        station_name = results_df["metadata.station_name"].iloc[0]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Define grid layout
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # Plot 1: Station Information
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')  # No axes for text box
        
        # Gather station info
        info_text = (
            f"Station: {station_code} - {station_name}\n\n"
            f"Location: {results_df['metadata.latitude'].iloc[0]:.4f}°N, "
            f"{results_df['metadata.longitude'].iloc[0]:.4f}°W\n\n"
            f"Time Coverage: {results_df['time_coverage.start_date'].iloc[0]} to "
            f"{results_df['time_coverage.end_date'].iloc[0]}\n"
            f"({results_df['time_coverage.n_hours'].iloc[0] / (24*365.25):.2f} years)\n\n"
            f"Datum: {results_df['metadata.datum_information'].iloc[0]}"
        )
        
        ax1.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.set_title("Station Information", fontsize=12)
        
        # Plot 2: Threshold and Return Level Info for var1
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')  # No axes for text box
        
        # Check if extreme value results are available for var1
        var1_col = f"extreme_value_results.{var1}"
        if var1_col + ".shape_parameter" in results_df.columns:
            shape = results_df[var1_col + ".shape_parameter"].iloc[0]
            scale = results_df[var1_col + ".scale_parameter"].iloc[0]
            threshold = results_df[var1_col + ".threshold"].iloc[0]
            n_peaks = results_df[var1_col + ".n_peaks"].iloc[0]
            n_exceedances = results_df[var1_col + ".n_exceedances"].iloc[0]
            
            # Extract return levels if available
            if var1_col + ".return_levels" in results_df.columns:
                return_levels = results_df[var1_col + ".return_levels"].iloc[0]
                return_years = results_df[var1_col + ".return_years"].iloc[0]
                
                # Format return levels
                return_level_text = "\nReturn Levels:\n"
                for year, level in zip(return_years, return_levels):
                    return_level_text += f"  {year}-year: {level:.3f}\n"
            else:
                return_level_text = ""
            
            info_text = (
                f"{var1} Extreme Value Analysis:\n\n"
                f"Threshold: {threshold:.3f}\n"
                f"Shape Parameter (ξ): {shape:.3f}\n"
                f"Scale Parameter (σ): {scale:.3f}\n\n"
                f"Number of Exceedances: {n_exceedances}\n"
                f"Number of Independent Peaks: {n_peaks}"
                f"{return_level_text}"
            )
        else:
            info_text = f"No extreme value results for {var1}"
        
        ax2.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.set_title(f"{var1} Extreme Value Results", fontsize=12)
        
        # Plot 3: Threshold and Return Level Info for var2
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')  # No axes for text box
        
        # Check if extreme value results are available for var2
        var2_col = f"extreme_value_results.{var2}"
        if var2_col + ".shape_parameter" in results_df.columns:
            shape = results_df[var2_col + ".shape_parameter"].iloc[0]
            scale = results_df[var2_col + ".scale_parameter"].iloc[0]
            threshold = results_df[var2_col + ".threshold"].iloc[0]
            n_peaks = results_df[var2_col + ".n_peaks"].iloc[0]
            n_exceedances = results_df[var2_col + ".n_exceedances"].iloc[0]
            
            # Extract return levels if available
            if var2_col + ".return_levels" in results_df.columns:
                return_levels = results_df[var2_col + ".return_levels"].iloc[0]
                return_years = results_df[var2_col + ".return_years"].iloc[0]
                
                # Format return levels
                return_level_text = "\nReturn Levels:\n"
                for year, level in zip(return_years, return_levels):
                    return_level_text += f"  {year}-year: {level:.3f}\n"
            else:
                return_level_text = ""
            
            info_text = (
                f"{var2} Extreme Value Analysis:\n\n"
                f"Threshold: {threshold:.3f}\n"
                f"Shape Parameter (ξ): {shape:.3f}\n"
                f"Scale Parameter (σ): {scale:.3f}\n\n"
                f"Number of Exceedances: {n_exceedances}\n"
                f"Number of Independent Peaks: {n_peaks}"
                f"{return_level_text}"
            )
        else:
            info_text = f"No extreme value results for {var2}"
        
        ax3.text(0.05, 0.95, info_text, fontsize=10, va='top', ha='left',
                transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.set_title(f"{var2} Extreme Value Results", fontsize=12)
        
        # Plot 4: Joint Exceedance Statistics
        ax4 = fig.add_subplot(gs[1, 0:2])
        
        # Check if joint exceedance results are available
        joint_col = "joint_exceedance_results"
        if joint_col + ".exceedance_probabilities.joint_empirical" in results_df.columns:
            # Extract probabilities
            p1 = results_df[joint_col + f".exceedance_probabilities.{var1}"].iloc[0]
            p2 = results_df[joint_col + f".exceedance_probabilities.{var2}"].iloc[0]
            p_joint = results_df[joint_col + ".exceedance_probabilities.joint_empirical"].iloc[0]
            p_ind = results_df[joint_col + ".exceedance_probabilities.joint_independence"].iloc[0]
            cpr = results_df[joint_col + ".conditional_probability_ratio"].iloc[0]
            
            # Extract conditional probabilities
            p1_given_2 = results_df[joint_col + f".conditional_probabilities.{var1}_given_{var2}"].iloc[0]
            p2_given_1 = results_df[joint_col + f".conditional_probabilities.{var2}_given_{var1}"].iloc[0]
            
            # Extract event counts
            n1 = results_df[joint_col + f".event_counts.{var1}"].iloc[0]
            n2 = results_df[joint_col + f".event_counts.{var2}"].iloc[0]
            n_joint = results_df[joint_col + ".event_counts.joint"].iloc[0]
            
            # Plot a bar chart
            x = ['P(var1)', 'P(var2)', 'P(joint) empirical', 'P(joint) independence']
            y = [p1, p2, p_joint, p_ind]
            
            bars = ax4.bar(x, y, color=['blue', 'orange', 'green', 'red'])
            
            # Add text labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            # Set axis properties
            ax4.set_ylabel('Probability')
            ax4.set_title(f'Joint Exceedance Probabilities (CPR = {cpr:.3f})')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add text annotation with additional information
            info_text = (
                f"Conditional Probabilities:\n"
                f"P({var1} | {var2}) = {p1_given_2:.4f}\n"
                f"P({var2} | {var1}) = {p2_given_1:.4f}\n\n"
                f"Event Counts:\n"
                f"{var1} events: {n1}\n"
                f"{var2} events: {n2}\n"
                f"Joint events: {n_joint}"
            )
            
            # Add text box in the top right
            ax4.text(0.98, 0.98, info_text, fontsize=10, va='top', ha='right',
                    transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, "No joint exceedance results available",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax4.transAxes)
        
        # Plot 5: Lag Analysis
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Check if lag analysis results are available
        lag_col = joint_col + ".lag_analysis"
        if lag_col + ".mean_lag" in results_df.columns:
            # Extract lag statistics
            mean_lag = results_df[lag_col + ".mean_lag"].iloc[0]
            median_lag = results_df[lag_col + ".median_lag"].iloc[0]
            lag_std = results_df[lag_col + ".lag_std"].iloc[0]
            n_lags = results_df[lag_col + ".n_lags"].iloc[0]
            
            # Extract lag bins if available
            lag_bin_cols = [col for col in results_df.columns if lag_col + ".lag_bins" in col]
            
            if lag_bin_cols and n_lags > 0:
                # Define bin edges and labels
                bin_edges = [-24, -12, -6, -3, 0, 3, 6, 12, 24]
                bin_labels = [f"{bin_edges[i]} to {bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
                
                # Get bin counts
                bin_values = []
                for i, label in enumerate(bin_labels):
                    if f"{lag_col}.lag_bins.{'neg' if bin_edges[i] < 0 else 'pos'}_{abs(bin_edges[i])}_{abs(bin_edges[i+1])}" in lag_bin_cols:
                        key = f"{lag_col}.lag_bins.{'neg' if bin_edges[i] < 0 else 'pos'}_{abs(bin_edges[i])}_{abs(bin_edges[i+1])}"
                        bin_values.append(results_df[key].iloc[0])
                    else:
                        bin_values.append(0)
                
                # Plot histogram
                ax5.bar(bin_labels, bin_values, color='lightgreen')
                ax5.set_xlabel('Lag Range [hours]')
                ax5.set_ylabel('Number of Events')
                ax5.set_title(f'Lag Distribution (Mean: {mean_lag:.1f} hours)')
                ax5.grid(True, alpha=0.3, axis='y')
                plt.setp(ax5.get_xticklabels(), rotation=90, ha='center')
                
                # Add text annotation with lag statistics
                info_text = (
                    f"Mean Lag: {mean_lag:.1f} hours\n"
                    f"Median Lag: {median_lag:.1f} hours\n"
                    f"Std Dev: {lag_std:.1f} hours\n"
                    f"Number of Joint Events: {n_lags}"
                )
                
                # Add text box in the top right
                ax5.text(0.98, 0.98, info_text, fontsize=9, va='top', ha='right',
                        transform=ax5.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax5.text(0.5, 0.5, "No lag data available or insufficient joint events",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, "No lag analysis results available",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax5.transAxes)
        
        # Plot 6: Return Level Curves for var1
        ax6 = fig.add_subplot(gs[2, 0])
        
        # Check if extreme value results are available for var1
        if var1_col + ".shape_parameter" in results_df.columns:
            shape = results_df[var1_col + ".shape_parameter"].iloc[0]
            scale = results_df[var1_col + ".scale_parameter"].iloc[0]
            threshold = results_df[var1_col + ".threshold"].iloc[0]
            n_peaks = results_df[var1_col + ".n_peaks"].iloc[0]
            n_hours = results_df["time_coverage.n_hours"].iloc[0]
            
            # Calculate rate of exceedance
            lambda_rate = n_peaks / n_hours
            annual_rate = lambda_rate * 24 * 365.25  # Assuming hourly data
            
            # Calculate return levels for a range of return periods
            years = np.logspace(0, 3, 100)  # 1 to 1000 years
            
            if abs(shape) < 1e-6:
                # For shape ≈ 0, use exponential formula
                return_levels = threshold + scale * np.log(years * annual_rate)
            else:
                # For non-zero shape, use GPD formula
                return_levels = threshold + (scale / shape) * ((years * annual_rate) ** shape - 1)
            
            # Plot the return level curve
            ax6.semilogx(years, return_levels, 'b-')
            
            # Add markers for common return periods
            for period in [2, 10, 50, 100]:
                if abs(shape) < 1e-6:
                    level = threshold + scale * np.log(period * annual_rate)
                else:
                    level = threshold + (scale / shape) * ((period * annual_rate) ** shape - 1)
                
                ax6.scatter(period, level, color='red', s=40, zorder=5)
                ax6.text(period, level, f'{period}yr', fontsize=8, ha='center', va='bottom')
            
            # Set axis properties
            ax6.set_xlabel('Return Period (years)')
            ax6.set_ylabel(f'{var1}')
            ax6.set_title(f'{var1} Return Level Curve')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, f"No extreme value results for {var1}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax6.transAxes)
        
        # Plot 7: Return Level Curves for var2
        ax7 = fig.add_subplot(gs[2, 1])
        
        # Check if extreme value results are available for var2
        if var2_col + ".shape_parameter" in results_df.columns:
            shape = results_df[var2_col + ".shape_parameter"].iloc[0]
            scale = results_df[var2_col + ".scale_parameter"].iloc[0]
            threshold = results_df[var2_col + ".threshold"].iloc[0]
            n_peaks = results_df[var2_col + ".n_peaks"].iloc[0]
            n_hours = results_df["time_coverage.n_hours"].iloc[0]
            
            # Calculate rate of exceedance
            lambda_rate = n_peaks / n_hours
            annual_rate = lambda_rate * 24 * 365.25  # Assuming hourly data
            
            # Calculate return levels for a range of return periods
            years = np.logspace(0, 3, 100)  # 1 to 1000 years
            
            if abs(shape) < 1e-6:
                # For shape ≈ 0, use exponential formula
                return_levels = threshold + scale * np.log(years * annual_rate)
            else:
                # For non-zero shape, use GPD formula
                return_levels = threshold + (scale / shape) * ((years * annual_rate) ** shape - 1)
            
            # Plot the return level curve
            ax7.semilogx(years, return_levels, 'orange')
            
            # Add markers for common return periods
            for period in [2, 10, 50, 100]:
                if abs(shape) < 1e-6:
                    level = threshold + scale * np.log(period * annual_rate)
                else:
                    level = threshold + (scale / shape) * ((period * annual_rate) ** shape - 1)
                
                ax7.scatter(period, level, color='red', s=40, zorder=5)
                ax7.text(period, level, f'{period}yr', fontsize=8, ha='center', va='bottom')
            
            # Set axis properties
            ax7.set_xlabel('Return Period (years)')
            ax7.set_ylabel(f'{var2}')
            ax7.set_title(f'{var2} Return Level Curve')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, f"No extreme value results for {var2}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax7.transAxes)
        
        # Plot 8: Summary information and interpretation
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')  # No axes for text box
        
        # Prepare summary text
        if joint_col + ".conditional_probability_ratio" in results_df.columns:
            cpr = results_df[joint_col + ".conditional_probability_ratio"].iloc[0]
            
            if cpr > 2:
                interpretation = "Strong dependence"
            elif cpr > 1.5:
                interpretation = "Moderate dependence"
            elif cpr > 1.1:
                interpretation = "Weak dependence"
            else:
                interpretation = "No significant dependence"
            
            # Add lag interpretation if available
            if lag_col + ".mean_lag" in results_df.columns:
                mean_lag = results_df[lag_col + ".mean_lag"].iloc[0]
                
                if abs(mean_lag) < 3:
                    lag_interp = "extremes typically occur nearly simultaneously."
                elif mean_lag > 0:
                    lag_interp = f"{var2} extremes typically lag {var1} extremes by {abs(mean_lag):.1f} hours."
                else:
                    lag_interp = f"{var1} extremes typically lag {var2} extremes by {abs(mean_lag):.1f} hours."
            else:
                lag_interp = "lag information not available."
            
            summary_text = (
                f"Summary of Compound Flooding Analysis\n"
                f"--------------------------------------\n\n"
                f"Interpretation: {interpretation}\n\n"
                f"The Conditional Probability Ratio (CPR) of {cpr:.2f} indicates "
                f"{'a significant ' if cpr > 1.5 else 'a '}"
                f"statistical dependence between extreme {var1} and {var2} at this station.\n\n"
                f"When extreme {var1} occurs, the probability of extreme {var2} is "
                f"{p2_given_1:.1%}, compared to a baseline probability of {p2:.1%}.\n\n"
                f"Similarly, when extreme {var2} occurs, the probability of extreme {var1} is "
                f"{p1_given_2:.1%}, compared to a baseline probability of {p1:.1%}.\n\n"
                f"Regarding timing, {lag_interp}\n\n"
                f"These results suggest that compound flooding from combined {var1} and {var2} "
                f"should be considered in flood risk assessment for this location."
            )
        else:
            summary_text = "Insufficient data for summary interpretation."
        
        ax8.text(0.05, 0.95, summary_text, fontsize=10, va='top', ha='left',
                transform=ax8.transAxes, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax8.set_title("Interpretation", fontsize=12)
        
        # Add main title
        plt.suptitle(f"Compound Flooding Analysis Summary - Station {station_code} ({station_name})", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        
        # Save the figure
        output_path = self.output_dir / f"summary_dashboard_{station_code}.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        # Show the plot if requested
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig


# Example Usage
if __name__ == "__main__":
    # Create a visualizer instance
    visualizer = CompoundFloodingVisualizer(
        results_dir=Path("outputs"),
        output_dir=Path("outputs/plots"),
        show_plots=True
    )
    
    # Generate example plots
    try:
        # Try to create a station map (requires Cartopy)
        visualizer.plot_station_map()
    except Exception as e:
        print(f"Error creating station map: {e}")
    
    try:
        # Generate a summary dashboard for a station
        visualizer.plot_summary_dashboard(station_code="240A")
    except Exception as e:
        print(f"Error creating summary dashboard: {e}")