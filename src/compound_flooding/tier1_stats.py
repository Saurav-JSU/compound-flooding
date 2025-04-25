"""
Tier 1 statistical analysis for compound flooding.

This module implements the Tier 1 analysis for compound flooding, including:
- Univariate extreme value analysis (POT/GPD)
- Threshold selection and diagnostics
- Empirical joint exceedance analysis
- Conditional probability ratio (CPR) calculation
- Lead/lag analysis
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import scipy.stats as stats
import warnings
import matplotlib.pyplot as plt
from scipy.stats import genpareto
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Type aliases
PathLike = Union[str, Path]
ThresholdDiagnostics = Dict[str, Dict[str, Union[np.ndarray, float]]]
ExtremeValueResults = Dict[str, Dict[str, Any]]
JointExceedanceResults = Dict[str, Any]


class Tier1Analyzer:
    """
    Class for performing Tier 1 statistical analysis on compound flooding data.
    
    This class implements the methods for univariate extreme value analysis,
    threshold selection, and joint exceedance analysis as described in the
    methodology document.
    """
    
    def __init__(
        self,
        preprocessed_dir: PathLike = Path("outputs/netcdf"),
        output_dir: PathLike = Path("outputs/tier1"),
        threshold_percentile: float = 99.0,
        lag_window: int = 24,
        min_cluster_separation: int = 72,
        save_diagnostics: bool = True
    ):
        """
        Initialize the Tier 1 analyzer.
        
        Parameters
        ----------
        preprocessed_dir : PathLike, optional
            Directory containing preprocessed NetCDF files.
        output_dir : PathLike, optional
            Directory to save Tier 1 analysis results.
        threshold_percentile : float, optional
            Percentile threshold for extreme value analysis. Default is 99.0.
        lag_window : int, optional
            Window (in hours) for lag analysis. Default is 24.
        min_cluster_separation : int, optional
            Minimum separation (in hours) between extreme events to be considered
            independent clusters. Default is 72 (3 days).
        save_diagnostics : bool, optional
            Whether to save diagnostic plots. Default is True.
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.output_dir = Path(output_dir)
        self.threshold_percentile = threshold_percentile
        self.lag_window = lag_window
        self.min_cluster_separation = min_cluster_separation
        self.save_diagnostics = save_diagnostics
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # If saving diagnostics, create plots directory
        if self.save_diagnostics:
            self.plots_dir = self.output_dir / "plots"
            self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def load_station_data(self, station_code: str) -> Optional[xr.Dataset]:
        """
        Load preprocessed data for a station.
        
        Parameters
        ----------
        station_code : str
            Station code.
        
        Returns
        -------
        Optional[xr.Dataset]
            Dataset containing station data, or None if file not found.
        """
        file_path = self.preprocessed_dir / f"{station_code}_preprocessed.nc"
        
        if not file_path.exists():
            print(f"Warning: Preprocessed file not found for station {station_code}: {file_path}")
            return None
        
        try:
            ds = xr.open_dataset(file_path)
            return ds
        except Exception as e:
            print(f"Error loading preprocessed data for station {station_code}: {e}")
            return None
    
    def analyze_station(
        self, 
        station_code: str, 
        variables: List[str] = ['sea_level', 'total_precipitation']
    ) -> Dict[str, Any]:
        """
        Perform Tier 1 analysis for a station.
        
        Parameters
        ----------
        station_code : str
            Station code.
        variables : List[str], optional
            List of variables to analyze. Default is ['sea_level', 'total_precipitation'].
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing Tier 1 analysis results.
        """
        print(f"Starting Tier 1 analysis for station {station_code}")
        
        # Load station data
        ds = self.load_station_data(station_code)
        if ds is None:
            return {"station_code": station_code, "status": "error", "message": "Data not found"}
        
        # Initialize results dictionary
        results = {
            "station_code": station_code,
            "metadata": {key: ds.attrs.get(key, None) for key in 
                        ['station_name', 'latitude', 'longitude', 'datum_information']},
            "time_coverage": {
                "start_date": pd.Timestamp(ds.time.values[0]).strftime('%Y-%m-%d'),
                "end_date": pd.Timestamp(ds.time.values[-1]).strftime('%Y-%m-%d'),
                "n_hours": len(ds.time)
            },
            "status": "success"
        }
        
        # Check if variables exist in dataset
        for var in variables:
            if var not in ds.data_vars:
                return {
                    "station_code": station_code, 
                    "status": "error", 
                    "message": f"Variable {var} not found in dataset"
                }
        
        # Perform univariate extreme value analysis
        threshold_diagnostics = {}
        extreme_results = {}
        
        for var in variables:
            print(f"  Performing univariate extreme value analysis for {var}")
            
            # Get variable data
            var_data = ds[var].values
            
            # Calculate threshold diagnostics
            var_diag = self._calculate_threshold_diagnostics(var_data, var, station_code)
            threshold_diagnostics[var] = var_diag
            
            # Get threshold from diagnostics or percentile
            threshold = var_diag.get('selected_threshold', 
                                    np.nanpercentile(var_data, self.threshold_percentile))
            
            # Perform POT extreme value analysis
            var_results = self._perform_pot_analysis(var_data, threshold, var, station_code)
            extreme_results[var] = var_results
        
        # Perform joint exceedance analysis
        print(f"  Performing joint exceedance analysis")
        joint_results = self._perform_joint_exceedance_analysis(
            ds, variables[0], variables[1], 
            extreme_results[variables[0]]['threshold'],
            extreme_results[variables[1]]['threshold']
        )
        
        # Package results
        results["threshold_diagnostics"] = threshold_diagnostics
        results["extreme_value_results"] = extreme_results
        results["joint_exceedance_results"] = joint_results
        
        # Save results
        self._save_results(results, station_code)
        
        print(f"Completed Tier 1 analysis for station {station_code}")
        return results
    
    def _calculate_threshold_diagnostics(
        self, 
        data: np.ndarray, 
        variable: str, 
        station_code: str
    ) -> Dict[str, Any]:
        """
        Calculate threshold selection diagnostics for extreme value analysis.
        
        This includes:
        - Mean residual life plot
        - Parameter stability plots (shape and modified scale)
        
        Parameters
        ----------
        data : np.ndarray
            Array of variable values.
        variable : str
            Variable name.
        station_code : str
            Station code.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing threshold diagnostics.
        """
        # Remove NaN values
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {"status": "error", "message": "No valid data points"}
        
        # For precipitation, focus on non-zero values
        if variable == 'total_precipitation' or variable == 'ground_precipitation':
            valid_data = valid_data[valid_data > 0]
            if len(valid_data) == 0:
                return {"status": "error", "message": "No non-zero precipitation values"}
        
        # Calculate percentiles for potential thresholds
        percentiles = np.arange(90, 99.5, 0.5)
        thresholds = np.percentile(valid_data, percentiles)
        
        # Initialize results
        mrl_values = []
        shape_values = []
        mod_scale_values = []
        n_exceedances = []
        
        # Calculate diagnostics for each threshold
        for threshold in thresholds:
            exceedances = valid_data[valid_data > threshold] - threshold
            n_exceedances.append(len(exceedances))
            
            if len(exceedances) > 10:  # Need sufficient data for reliable estimates
                # Mean residual life (mean excess)
                mrl = np.mean(exceedances)
                mrl_values.append(mrl)
                
                # Fit GPD and get shape parameter
                try:
                    shape, loc, scale = genpareto.fit(exceedances, floc=0)
                    shape_values.append(shape)
                    
                    # Modified scale (xi * sigma)
                    mod_scale = scale - shape * threshold
                    mod_scale_values.append(mod_scale)
                except Exception:
                    # If fitting fails, append NaN
                    shape_values.append(np.nan)
                    mod_scale_values.append(np.nan)
            else:
                mrl_values.append(np.nan)
                shape_values.append(np.nan)
                mod_scale_values.append(np.nan)
        
        # Determine threshold based on diagnostics
        # This is a simplified heuristic - in practice, this requires human judgment
        valid_indices = ~np.isnan(shape_values)
        if np.sum(valid_indices) >= 5:  # Need at least 5 valid points for trend estimation
            # Find where shape parameter stabilizes
            shape_array = np.array(shape_values)[valid_indices]
            thresh_array = thresholds[valid_indices]
            
            # Use LOWESS to smooth the shape parameter series
            try:
                smoothed = lowess(shape_array, thresh_array, frac=0.3, return_sorted=True)
                thresh_smooth, shape_smooth = smoothed[:, 0], smoothed[:, 1]
                
                # Calculate the standard deviation in a sliding window
                window_size = max(3, len(shape_smooth) // 5)
                stdevs = []
                
                for i in range(len(shape_smooth) - window_size + 1):
                    stdevs.append(np.std(shape_smooth[i:i+window_size]))
                
                # Find where the standard deviation is small and stable
                stdevs = np.array(stdevs)
                if len(stdevs) > 0:
                    stable_idx = np.argmin(stdevs) + window_size // 2
                    if stable_idx < len(thresh_smooth):
                        selected_threshold = thresh_smooth[stable_idx]
                    else:
                        # Fallback: use the threshold corresponding to given percentile
                        selected_threshold = np.percentile(valid_data, self.threshold_percentile)
                else:
                    selected_threshold = np.percentile(valid_data, self.threshold_percentile)
            except Exception:
                # Fallback if LOWESS fails
                selected_threshold = np.percentile(valid_data, self.threshold_percentile)
        else:
            # Not enough valid data points, use percentile threshold
            selected_threshold = np.percentile(valid_data, self.threshold_percentile)
        
        # Ensure we have enough exceedances for POT analysis
        min_exceedances = 50  # Minimum number of exceedances desired
        if sum(valid_data > selected_threshold) < min_exceedances:
            # Find threshold that gives at least min_exceedances
            for p in np.arange(self.threshold_percentile - 0.5, 90, -0.5):
                test_threshold = np.percentile(valid_data, p)
                if sum(valid_data > test_threshold) >= min_exceedances:
                    selected_threshold = test_threshold
                    break
        
        # Save diagnostics plot
        if self.save_diagnostics:
            self._plot_threshold_diagnostics(
                thresholds, mrl_values, shape_values, mod_scale_values, n_exceedances,
                selected_threshold, variable, station_code
            )
        
        return {
            "thresholds": thresholds,
            "mrl_values": mrl_values,
            "shape_values": shape_values,
            "mod_scale_values": mod_scale_values,
            "n_exceedances": n_exceedances,
            "selected_threshold": selected_threshold,
            "status": "success"
        }
    
    def _plot_threshold_diagnostics(
        self, 
        thresholds: np.ndarray, 
        mrl_values: List[float],
        shape_values: List[float],
        mod_scale_values: List[float],
        n_exceedances: List[int],
        selected_threshold: float,
        variable: str,
        station_code: str
    ) -> None:
        """
        Plot threshold selection diagnostics.
        
        Parameters
        ----------
        thresholds : np.ndarray
            Array of threshold values.
        mrl_values : List[float]
            List of mean residual life values.
        shape_values : List[float]
            List of shape parameter values.
        mod_scale_values : List[float]
            List of modified scale parameter values.
        n_exceedances : List[int]
            List of number of exceedances.
        selected_threshold : float
            Selected threshold value.
        variable : str
            Variable name.
        station_code : str
            Station code.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Convert lists to arrays for plotting
        thresholds = np.array(thresholds)
        mrl_values = np.array(mrl_values)
        shape_values = np.array(shape_values)
        mod_scale_values = np.array(mod_scale_values)
        n_exceedances = np.array(n_exceedances)
        
        # Mean residual life plot
        ax = axes[0, 0]
        valid = ~np.isnan(mrl_values)
        ax.plot(thresholds[valid], mrl_values[valid], 'o-')
        ax.axvline(selected_threshold, color='r', linestyle='--')
        ax.set_title('Mean Residual Life Plot')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Mean Excess')
        ax.grid(True, alpha=0.3)
        
        # Shape parameter stability plot
        ax = axes[0, 1]
        valid = ~np.isnan(shape_values)
        ax.plot(thresholds[valid], shape_values[valid], 'o-')
        ax.axvline(selected_threshold, color='r', linestyle='--')
        ax.set_title('Shape Parameter Stability')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Shape Parameter')
        ax.grid(True, alpha=0.3)
        
        # Modified scale parameter stability plot
        ax = axes[1, 0]
        valid = ~np.isnan(mod_scale_values)
        ax.plot(thresholds[valid], mod_scale_values[valid], 'o-')
        ax.axvline(selected_threshold, color='r', linestyle='--')
        ax.set_title('Modified Scale Parameter Stability')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Modified Scale Parameter')
        ax.grid(True, alpha=0.3)
        
        # Number of exceedances
        ax = axes[1, 1]
        ax.plot(thresholds, n_exceedances, 'o-')
        ax.axvline(selected_threshold, color='r', linestyle='--', 
                   label=f'Selected threshold: {selected_threshold:.3f}')
        ax.set_title('Number of Exceedances')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Number of Exceedances')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Threshold Diagnostics for {variable} - Station {station_code}', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{station_code}_{variable}_threshold_diagnostics.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
    
    def _perform_pot_analysis(
        self, 
        data: np.ndarray, 
        threshold: float, 
        variable: str, 
        station_code: str
    ) -> Dict[str, Any]:
        """
        Perform Peaks-Over-Threshold (POT) extreme value analysis.
        
        Parameters
        ----------
        data : np.ndarray
            Array of variable values.
        threshold : float
            Threshold for extreme value analysis.
        variable : str
            Variable name.
        station_code : str
            Station code.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing POT analysis results.
        """
        # Remove NaN values
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {"status": "error", "message": "No valid data points"}
        
        # Extract exceedances above threshold
        exceedances_idx = np.where(valid_data > threshold)[0]
        
        if len(exceedances_idx) == 0:
            return {"status": "error", "message": "No exceedances above threshold"}
        
        # Identify clusters (consecutive or nearby exceedances)
        # This is a simple declustering to ensure independence
        clusters = []
        current_cluster = [exceedances_idx[0]]
        
        for i in range(1, len(exceedances_idx)):
            if exceedances_idx[i] - exceedances_idx[i-1] <= self.min_cluster_separation:
                # Add to current cluster if close enough
                current_cluster.append(exceedances_idx[i])
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [exceedances_idx[i]]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        # Extract peak exceedance from each cluster
        peak_indices = []
        for cluster in clusters:
            peak_idx = max(cluster, key=lambda idx: valid_data[idx])
            peak_indices.append(peak_idx)
        
        # Sort peak indices to keep original order
        peak_indices.sort()
        
        # Extract peak exceedances for GPD fitting
        peak_exceedances = valid_data[peak_indices] - threshold
        
        # Fit Generalized Pareto Distribution
        try:
            shape, loc, scale = genpareto.fit(peak_exceedances, floc=0)
            
            # Calculate return levels
            years = np.array([2, 5, 10, 20, 50, 100])
            return_levels = self._calculate_return_levels(
                threshold, shape, scale, len(peak_exceedances), len(valid_data), years
            )
            
            # Diagnostic plots
            if self.save_diagnostics:
                self._plot_gpd_diagnostics(
                    peak_exceedances, shape, scale, threshold, variable, station_code
                )
            
            return {
                "threshold": threshold,
                "shape_parameter": shape,
                "scale_parameter": scale,
                "n_exceedances": len(exceedances_idx),
                "n_peaks": len(peak_indices),
                "return_years": years.tolist(),
                "return_levels": return_levels.tolist(),
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": f"GPD fitting failed: {str(e)}"}
    
    def _calculate_return_levels(
        self, 
        threshold: float, 
        shape: float, 
        scale: float, 
        n_peaks: int, 
        n_points: int, 
        years: np.ndarray
    ) -> np.ndarray:
        """
        Calculate return levels for specified return periods.
        
        Parameters
        ----------
        threshold : float
            Threshold used for POT analysis.
        shape : float
            GPD shape parameter.
        scale : float
            GPD scale parameter.
        n_peaks : int
            Number of peaks used in GPD fitting.
        n_points : int
            Total number of data points.
        years : np.ndarray
            Array of return periods in years.
        
        Returns
        -------
        np.ndarray
            Array of return levels.
        """
        # Compute rate of exceedance (lambda)
        lambda_rate = n_peaks / n_points
        
        # Assuming hourly data, adjust lambda to annual rate
        annual_rate = lambda_rate * 24 * 365.25
        
        # Calculate return levels
        if abs(shape) < 1e-6:
            # For shape ≈ 0, use exponential formula
            return_levels = threshold + scale * np.log(years * annual_rate)
        else:
            # For non-zero shape, use GPD formula
            return_levels = threshold + (scale / shape) * ((years * annual_rate) ** shape - 1)
        
        return return_levels
    
    def _plot_gpd_diagnostics(
        self, 
        exceedances: np.ndarray, 
        shape: float, 
        scale: float, 
        threshold: float, 
        variable: str, 
        station_code: str
    ) -> None:
        """
        Plot GPD model diagnostics.
        
        Parameters
        ----------
        exceedances : np.ndarray
            Array of exceedances above threshold.
        shape : float
            GPD shape parameter.
        scale : float
            GPD scale parameter.
        threshold : float
            Threshold used for POT analysis.
        variable : str
            Variable name.
        station_code : str
            Station code.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram and fitted PDF
        ax = axes[0, 0]
        bins = min(50, max(10, int(len(exceedances) / 10)))
        hist, bin_edges = np.histogram(exceedances, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, hist, width=bin_centers[1] - bin_centers[0], alpha=0.5)
        
        # Generate fitted PDF
        x = np.linspace(0, max(exceedances) * 1.1, 1000)
        y = genpareto.pdf(x, shape, scale=scale)
        ax.plot(x, y, 'r-', label=f'GPD(ξ={shape:.3f}, σ={scale:.3f})')
        
        ax.set_title('Histogram and Fitted PDF')
        ax.set_xlabel('Exceedance')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # QQ-plot
        ax = axes[0, 1]
        # Empirical quantiles
        sorted_exceedances = np.sort(exceedances)
        p = np.arange(1, len(exceedances) + 1) / (len(exceedances) + 1)
        # Theoretical quantiles
        theoretical_quantiles = genpareto.ppf(p, shape, scale=scale)
        
        ax.scatter(theoretical_quantiles, sorted_exceedances)
        ax.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--')
        ax.set_title('QQ-Plot')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Empirical Quantiles')
        ax.grid(True, alpha=0.3)
        
        # Return level plot
        ax = axes[1, 0]
        years = np.logspace(0, 2, 100)  # 1 to 100 years
        return_levels = self._calculate_return_levels(
            threshold, shape, scale, len(exceedances), 
            len(exceedances) * (max(exceedances) / np.mean(exceedances)), years
        )
        
        ax.semilogx(years, return_levels)
        ax.set_title('Return Level Plot')
        ax.set_xlabel('Return Period (years)')
        ax.set_ylabel(f'{variable} ({threshold} + excess)')
        ax.grid(True, alpha=0.3)
        
        # Excess vs Threshold plot
        ax = axes[1, 1]
        # Generate some equally spaced quantiles
        quantiles = np.linspace(0, 0.9, 10)
        thresholds = np.quantile(exceedances, quantiles)
        mean_excess = []
        
        for t in thresholds:
            excess_t = exceedances[exceedances > t] - t
            if len(excess_t) > 10:
                mean_excess.append(np.mean(excess_t))
            else:
                mean_excess.append(np.nan)
        
        ax.plot(thresholds, mean_excess, 'o-')
        ax.set_title('Mean Excess Plot')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Mean Excess')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'GPD Model Diagnostics for {variable} - Station {station_code}', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{station_code}_{variable}_gpd_diagnostics.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
    
    def _perform_joint_exceedance_analysis(
        self, 
        ds: xr.Dataset, 
        var1: str, 
        var2: str, 
        threshold1: float, 
        threshold2: float
    ) -> Dict[str, Any]:
        """
        Perform joint exceedance analysis.
        
        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing station data.
        var1 : str
            First variable name.
        var2 : str
            Second variable name.
        threshold1 : float
            Threshold for first variable.
        threshold2 : float
            Threshold for second variable.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing joint exceedance analysis results.
        """
        # Extract time series
        time = ds.time.values
        data1 = ds[var1].values
        data2 = ds[var2].values
        
        # Create a DataFrame with aligned time series
        df = pd.DataFrame({
            'time': time,
            var1: data1,
            var2: data2
        })
        
        # Remove rows with NaN values
        df_valid = df.dropna(subset=[var1, var2])
        
        if len(df_valid) == 0:
            return {"status": "error", "message": "No valid data points after removing NaNs"}
        
        # Calculate empirical exceedance probabilities
        p1 = np.mean(df_valid[var1] > threshold1)
        p2 = np.mean(df_valid[var2] > threshold2)
        
        # Calculate joint exceedance probability
        p_joint = np.mean((df_valid[var1] > threshold1) & (df_valid[var2] > threshold2))
        
        # Calculate joint probability assuming independence
        p_ind = p1 * p2
        
        # Calculate conditional probability ratio (CPR)
        cpr = p_joint / p_ind if p_ind > 0 else np.nan
        
        # Calculate conditional probabilities
        p_1_given_2 = np.mean(df_valid[var1][df_valid[var2] > threshold2] > threshold1) if any(df_valid[var2] > threshold2) else np.nan
        p_2_given_1 = np.mean(df_valid[var2][df_valid[var1] > threshold1] > threshold2) if any(df_valid[var1] > threshold1) else np.nan
        
        # Perform lag analysis
        lag_results = self._perform_lag_analysis(
            df_valid, var1, var2, threshold1, threshold2
        )
        
        # Count unique events (accounting for autocorrelation)
        # For var1
        events1 = self._identify_independent_events(
            df_valid, var1, threshold1, self.min_cluster_separation
        )
        n_events1 = len(events1)
        
        # For var2
        events2 = self._identify_independent_events(
            df_valid, var2, threshold2, self.min_cluster_separation
        )
        n_events2 = len(events2)
        
        # Count joint events
        joint_events = self._identify_joint_events(
            events1, events2, df_valid, self.lag_window
        )
        n_joint_events = len(joint_events)
        
        # Create plot of joint exceedances
        if self.save_diagnostics:
            self._plot_joint_exceedances(
                df_valid, var1, var2, threshold1, threshold2, 
                p_joint, cpr, ds.attrs.get('station_code', 'unknown')
            )
        
        return {
            "thresholds": {
                var1: threshold1,
                var2: threshold2
            },
            "exceedance_probabilities": {
                var1: float(p1),
                var2: float(p2),
                "joint_empirical": float(p_joint),
                "joint_independence": float(p_ind)
            },
            "conditional_probabilities": {
                f"{var1}_given_{var2}": float(p_1_given_2),
                f"{var2}_given_{var1}": float(p_2_given_1)
            },
            "conditional_probability_ratio": float(cpr),
            "event_counts": {
                var1: int(n_events1),
                var2: int(n_events2),
                "joint": int(n_joint_events)
            },
            "lag_analysis": lag_results,
            "status": "success"
        }
    
    def _identify_independent_events(
        self, 
        df: pd.DataFrame, 
        variable: str, 
        threshold: float, 
        min_separation: int
    ) -> List[Dict[str, Any]]:
        """
        Identify independent extreme events.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time series data.
        variable : str
            Variable name.
        threshold : float
            Threshold for extreme events.
        min_separation : int
            Minimum separation (in hours) between extreme events.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of identified independent events.
        """
        # Create binary exceedance series
        exceedance = df[variable] > threshold
        
        if not exceedance.any():
            return []
        
        # Convert to datetime index for easier time calculations
        df_with_datetime = df.copy()
        df_with_datetime['time'] = pd.to_datetime(df_with_datetime['time'])
        df_with_datetime.set_index('time', inplace=True)
        
        # Find runs of exceedances
        events = []
        in_event = False
        event_start = None
        event_peak_value = None
        event_peak_time = None
        
        for idx, row in df_with_datetime.iterrows():
            if row[variable] > threshold:
                if not in_event:
                    # Start a new event
                    in_event = True
                    event_start = idx
                    event_peak_value = row[variable]
                    event_peak_time = idx
                else:
                    # Continue the current event, update peak if needed
                    if row[variable] > event_peak_value:
                        event_peak_value = row[variable]
                        event_peak_time = idx
            elif in_event:
                # End of an event
                events.append({
                    'start_time': event_start,
                    'end_time': idx,
                    'peak_time': event_peak_time,
                    'peak_value': event_peak_value,
                    'duration': (idx - event_start).total_seconds() / 3600  # Duration in hours
                })
                in_event = False
        
        # Add the last event if still in progress
        if in_event:
            events.append({
                'start_time': event_start,
                'end_time': df_with_datetime.index[-1],
                'peak_time': event_peak_time,
                'peak_value': event_peak_value,
                'duration': (df_with_datetime.index[-1] - event_start).total_seconds() / 3600
            })
        
        # Merge events that are too close together
        if not events:
            return []
        
        events.sort(key=lambda e: e['start_time'])
        merged_events = [events[0]]
        
        for event in events[1:]:
            prev_event = merged_events[-1]
            hours_since_prev = (event['start_time'] - prev_event['end_time']).total_seconds() / 3600
            
            if hours_since_prev < min_separation:
                # Merge with previous event
                prev_event['end_time'] = max(prev_event['end_time'], event['end_time'])
                prev_event['duration'] = (prev_event['end_time'] - prev_event['start_time']).total_seconds() / 3600
                
                # Update peak if current event has higher peak
                if event['peak_value'] > prev_event['peak_value']:
                    prev_event['peak_value'] = event['peak_value']
                    prev_event['peak_time'] = event['peak_time']
            else:
                # Add as a new event
                merged_events.append(event)
        
        return merged_events
    
    def _identify_joint_events(
        self, 
        events1: List[Dict[str, Any]], 
        events2: List[Dict[str, Any]], 
        df: pd.DataFrame, 
        window_hours: int
    ) -> List[Dict[str, Any]]:
        """
        Identify joint extreme events within a time window.
        
        Parameters
        ----------
        events1 : List[Dict[str, Any]]
            List of events for first variable.
        events2 : List[Dict[str, Any]]
            List of events for second variable.
        df : pd.DataFrame
            DataFrame containing time series data.
        window_hours : int
            Time window (in hours) for considering joint events.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of identified joint events.
        """
        joint_events = []
        
        for e1 in events1:
            for e2 in events2:
                # Calculate time difference between peaks in hours
                time_diff = abs((e1['peak_time'] - e2['peak_time']).total_seconds() / 3600)
                
                if time_diff <= window_hours:
                    # This is a joint event
                    joint_events.append({
                        'event1_peak_time': e1['peak_time'],
                        'event1_peak_value': e1['peak_value'],
                        'event2_peak_time': e2['peak_time'],
                        'event2_peak_value': e2['peak_value'],
                        'time_lag': (e2['peak_time'] - e1['peak_time']).total_seconds() / 3600  # Lag in hours, positive if e2 occurs after e1
                    })
        
        return joint_events
    
    def _perform_lag_analysis(
        self, 
        df: pd.DataFrame, 
        var1: str, 
        var2: str, 
        threshold1: float, 
        threshold2: float
    ) -> Dict[str, Any]:
        """
        Perform lag analysis to identify temporal relationships.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time series data.
        var1 : str
            First variable name.
        var2 : str
            Second variable name.
        threshold1 : float
            Threshold for first variable.
        threshold2 : float
            Threshold for second variable.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing lag analysis results.
        """
        # Convert to datetime index for easier lag calculations
        df_with_datetime = df.copy()
        df_with_datetime['time'] = pd.to_datetime(df_with_datetime['time'])
        df_with_datetime.set_index('time', inplace=True)
        
        # Identify extreme events
        events1 = self._identify_independent_events(
            df_with_datetime.reset_index(), var1, threshold1, self.min_cluster_separation
        )
        events2 = self._identify_independent_events(
            df_with_datetime.reset_index(), var2, threshold2, self.min_cluster_separation
        )
        
        # Calculate time lags between events
        lags = []
        
        for e1 in events1:
            # Find the closest event2 in time
            closest_e2 = None
            min_diff = float('inf')
            
            for e2 in events2:
                time_diff = abs((e1['peak_time'] - e2['peak_time']).total_seconds() / 3600)
                
                if time_diff <= self.lag_window and time_diff < min_diff:
                    min_diff = time_diff
                    closest_e2 = e2
            
            if closest_e2 is not None:
                # Calculate lag (positive if var2 occurs after var1)
                lag = (closest_e2['peak_time'] - e1['peak_time']).total_seconds() / 3600
                lags.append(lag)
        
        if not lags:
            return {
                "status": "warning", 
                "message": "No joint events found within lag window"
            }
        
        # Calculate lag statistics
        lags = np.array(lags)
        mean_lag = np.mean(lags)
        median_lag = np.median(lags)
        lag_std = np.std(lags)
        
        # Count lags in different windows
        lag_bins = {
            "neg_24_12": np.sum((lags >= -24) & (lags < -12)),
            "neg_12_6": np.sum((lags >= -12) & (lags < -6)),
            "neg_6_3": np.sum((lags >= -6) & (lags < -3)),
            "neg_3_0": np.sum((lags >= -3) & (lags < 0)),
            "pos_0_3": np.sum((lags >= 0) & (lags < 3)),
            "pos_3_6": np.sum((lags >= 3) & (lags < 6)),
            "pos_6_12": np.sum((lags >= 6) & (lags < 12)),
            "pos_12_24": np.sum((lags >= 12) & (lags <= 24))
        }
        
        return {
            "mean_lag": float(mean_lag),
            "median_lag": float(median_lag),
            "lag_std": float(lag_std),
            "n_lags": len(lags),
            "lag_bins": lag_bins,
            "status": "success"
        }
    
    def _plot_joint_exceedances(
        self, 
        df: pd.DataFrame, 
        var1: str, 
        var2: str, 
        threshold1: float, 
        threshold2: float, 
        p_joint: float, 
        cpr: float, 
        station_code: str
    ) -> None:
        """
        Plot joint exceedances.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time series data.
        var1 : str
            First variable name.
        var2 : str
            Second variable name.
        threshold1 : float
            Threshold for first variable.
        threshold2 : float
            Threshold for second variable.
        p_joint : float
            Joint exceedance probability.
        cpr : float
            Conditional probability ratio.
        station_code : str
            Station code.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        colors = np.zeros(len(df), dtype=int)
        colors[(df[var1] > threshold1) & (df[var2] <= threshold2)] = 1  # var1 exceedance only
        colors[(df[var1] <= threshold1) & (df[var2] > threshold2)] = 2  # var2 exceedance only
        colors[(df[var1] > threshold1) & (df[var2] > threshold2)] = 3   # joint exceedance
        
        # Create color map
        cmap = plt.cm.get_cmap('viridis', 4)
        
        # Create scatter plot
        scatter = ax1.scatter(df[var1], df[var2], c=colors, cmap=cmap, alpha=0.5, s=10)
        
        # Add threshold lines
        ax1.axvline(threshold1, color='r', linestyle='--', label=f'{var1} threshold')
        ax1.axhline(threshold2, color='r', linestyle='--', label=f'{var2} threshold')
        
        # Add labels and legend
        ax1.set_xlabel(var1)
        ax1.set_ylabel(var2)
        ax1.set_title(f'Joint Exceedances (CPR: {cpr:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_ticks([0.4, 1.2, 2.0, 2.8])
        cbar.set_ticklabels(['No exceedance', f'{var1} only', f'{var2} only', 'Joint'])
        
        # Create lag histogram
        # Convert to datetime index for easier lag calculations
        df_with_datetime = df.copy()
        df_with_datetime['time'] = pd.to_datetime(df_with_datetime['time'])
        df_with_datetime.set_index('time', inplace=True)
        
        # Identify extreme events
        events1 = self._identify_independent_events(
            df_with_datetime.reset_index(), var1, threshold1, self.min_cluster_separation
        )
        events2 = self._identify_independent_events(
            df_with_datetime.reset_index(), var2, threshold2, self.min_cluster_separation
        )
        
        # Calculate lags between nearby events
        lags = []
        for e1 in events1:
            for e2 in events2:
                time_diff = abs((e1['peak_time'] - e2['peak_time']).total_seconds() / 3600)
                if time_diff <= self.lag_window:
                    # Calculate lag (positive if var2 occurs after var1)
                    lag = (e2['peak_time'] - e1['peak_time']).total_seconds() / 3600
                    lags.append(lag)
        
        if lags:
            # Plot lag histogram
            bins = np.linspace(-self.lag_window, self.lag_window, 25)
            ax2.hist(lags, bins=bins, alpha=0.7)
            ax2.axvline(0, color='r', linestyle='--')
            ax2.set_xlabel(f'Lag ({var2} relative to {var1}) [hours]')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Lag Distribution of Joint Extreme Events')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No joint events found within lag window',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
        
        plt.suptitle(f'Joint Exceedance Analysis - Station {station_code}', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{station_code}_joint_exceedances.png"
        plt.savefig(plot_path, dpi=100)
        plt.close()
    
    def _save_results(self, results: Dict[str, Any], station_code: str) -> None:
        """
        Save analysis results to a file.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Dictionary containing analysis results.
        station_code : str
            Station code.
        """
        # Create a flat DataFrame from the nested dictionary for easier saving
        flat_dict = self._flatten_dict(results)
        df = pd.DataFrame([flat_dict])
        
        # Save to parquet
        parquet_path = self.output_dir / f"{station_code}_tier1_results.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved results to {parquet_path}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary to flatten.
        parent_key : str, optional
            Parent key for nested dictionary. Default is ''.
        
        Returns
        -------
        Dict[str, Any]
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # For lists of dictionaries, just keep the count
                items.append((f"{new_key}.count", len(v)))
                # Maybe save the first item as an example
                if len(v) > 0:
                    items.extend(self._flatten_dict(v[0], f"{new_key}.sample").items())
            else:
                items.append((new_key, v))
        
        return dict(items)


# Example Usage
if __name__ == "__main__":
    import tempfile
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create a mock preprocessed NetCDF file
        nc_dir = tmp_path / "netcdf"
        nc_dir.mkdir()
        
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        
        time = pd.date_range('2020-01-01', periods=1000, freq='H')
        
        # Generate correlated sea level and precipitation data
        z = np.random.normal(0, 1, len(time))
        sea_level = 1.5 + 0.5 * np.sin(np.linspace(0, 10, len(time))) + 0.3 * z + 0.2 * np.random.normal(0, 1, len(time))
        precip = 0.001 * np.exp(0.2 * z + 0.8 * np.random.normal(0, 1, len(time)))
        
        # Create a few spikes that exceed extreme thresholds
        sea_level[100:110] += 2.0
        sea_level[500:505] += 1.5
        sea_level[800:810] += 1.8
        
        precip[105:115] += 0.05
        precip[300:310] += 0.03
        precip[805:815] += 0.04
        
        # Create a mock Dataset
        ds = xr.Dataset(
            data_vars={
                'sea_level': ('time', sea_level),
                'total_precipitation': ('time', precip)
            },
            coords={
                'time': time
            },
            attrs={
                'station_code': 'TEST1',
                'station_name': 'Test Station',
                'latitude': 40.0,
                'longitude': -74.0,
                'datum_information': 'Relative to local mean sea level'
            }
        )
        
        # Save mock NetCDF file
        nc_file = nc_dir / "TEST1_preprocessed.nc"
        ds.to_netcdf(nc_file)
        
        # Create output directory
        output_dir = tmp_path / "tier1"
        
        # Initialize the analyzer
        analyzer = Tier1Analyzer(
            preprocessed_dir=nc_dir,
            output_dir=output_dir,
            threshold_percentile=95.0,
            lag_window=12
        )
        
        # Analyze the station
        results = analyzer.analyze_station('TEST1')
        
        # Print some key results
        print("\nTier 1 Analysis Results:")
        print(f"Station: {results['station_code']} - {results['metadata']['station_name']}")
        print(f"Status: {results['status']}")
        print("\nExtreme Value Results:")
        for var, res in results['extreme_value_results'].items():
            if res['status'] == 'success':
                print(f"  {var}: shape={res['shape_parameter']:.3f}, scale={res['scale_parameter']:.3f}")
                print(f"     100-year return level: {res['return_levels'][-1]:.3f}")
        
        print("\nJoint Exceedance Results:")
        joint_res = results['joint_exceedance_results']
        if joint_res['status'] == 'success':
            print(f"  CPR: {joint_res['conditional_probability_ratio']:.3f}")
            print(f"  Joint events: {joint_res['event_counts']['joint']}")
            print(f"  Mean lag: {joint_res['lag_analysis']['mean_lag']:.1f} hours")