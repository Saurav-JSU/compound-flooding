# src/compound_flooding/joint_empirical.py
"""
Module: joint_empirical.py
Responsibilities:
- Compute empirical joint exceedance statistics for two variables
- Support optional lead/lag window for co-occurrence
- Compute marginal exceedance rates, joint rate, conditional probabilities, and CPR
- Calculate confidence intervals using bootstrap resampling
- Save results to Parquet or JSON formats
"""
import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_data_arrays(
    da1: xr.DataArray,
    da2: xr.DataArray,
    threshold1: float,
    threshold2: float
) -> Tuple[bool, str]:
    """
    Validate input DataArrays and thresholds.
    
    Parameters
    ----------
    da1, da2 : xr.DataArray
        Data arrays to validate
    threshold1, threshold2 : float
        Thresholds for exceedance
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check input types
    if not isinstance(da1, xr.DataArray) or not isinstance(da2, xr.DataArray):
        return False, "Inputs must be xarray DataArrays"
    
    # Check coordinate compatibility
    if 'datetime' not in da1.coords or 'datetime' not in da2.coords:
        return False, "Both DataArrays must have 'datetime' coordinate"
    
    # Check for overlapping time periods
    time1 = da1.coords['datetime'].values
    time2 = da2.coords['datetime'].values
    
    if len(np.intersect1d(time1, time2)) == 0:
        return False, "DataArrays have no overlapping timestamps"
    
    # Check thresholds
    if not np.isfinite(threshold1) or not np.isfinite(threshold2):
        return False, "Thresholds must be finite values"
    
    # Check if arrays have any finite values
    if np.isnan(da1.values).all() or np.isnan(da2.values).all():
        return False, "At least one DataArray contains only NaN values"
    
    return True, ""


def compute_empirical_stats(
    da1: xr.DataArray,
    da2: xr.DataArray,
    threshold1: float,
    threshold2: float,
    lag_hours: int = 0,
    min_joint_events: int = 5
) -> Dict[str, Any]:
    """
    Compute empirical exceedance statistics for two DataArrays.

    Parameters
    ----------
    da1, da2 : xr.DataArray
        Time-aligned data arrays (must share same datetime coordinate).
    threshold1, threshold2 : float
        Exceedance thresholds for da1 and da2.
    lag_hours : int, default 0
        If >0, allow da2 exceedances within ±lag_hours around each da1 timestamp.
    min_joint_events : int, default 5
        Minimum number of joint events required for reliable statistics.

    Returns
    -------
    stats : dict
        Keys: 
            n_total: Total number of valid observations
            n_exc1: Number of exceedances in variable 1
            n_exc2: Number of exceedances in variable 2
            n_joint: Number of joint exceedances
            p_exc1: Probability of exceedance in variable 1
            p_exc2: Probability of exceedance in variable 2
            p_joint: Joint exceedance probability
            p_independent: Joint probability assuming independence
            cpr: Conditional probability ratio
            p2_given_1: Conditional probability of var2 exceedance given var1 exceedance
            p1_given_2: Conditional probability of var1 exceedance given var2 exceedance
            significance: Dictionary with bootstrap confidence intervals
        
    Raises
    ------
    ValueError
        If inputs are invalid or no overlapping non-NaN observations
    """
    # Validate inputs
    is_valid, error_msg = validate_data_arrays(da1, da2, threshold1, threshold2)
    if not is_valid:
        raise ValueError(f"Invalid inputs: {error_msg}")
    
    if lag_hours < 0:
        raise ValueError(f"lag_hours must be non-negative, got {lag_hours}")
    
    logger.info(f"Computing empirical statistics with thresholds {threshold1:.4f} and {threshold2:.4f}")
    
    # Convert to pandas Series and align on index
    sx = da1.to_series()
    sy = da2.to_series()
    
    # Combine into DataFrame and drop NaN values
    df = pd.DataFrame({'x': sx, 'y': sy}).dropna()
    n_total = len(df)
    
    if n_total == 0:
        raise ValueError("No overlapping non-NaN observations.")
    
    logger.info(f"Found {n_total} valid observations")
    
    # If lag window, compute rolling max of y
    if lag_hours > 0:
        logger.info(f"Using lag window of ±{lag_hours} hours")
        window = 2 * lag_hours + 1
        y_roll = df['y'].rolling(window=window, center=True, min_periods=1).max()
    else:
        y_roll = df['y']
    
    # Exceedance indicators
    exc1 = df['x'] > threshold1
    exc2 = y_roll > threshold2
    
    n_exc1 = int(exc1.sum())
    n_exc2 = int(exc2.sum())
    n_joint = int((exc1 & exc2).sum())
    
    logger.info(f"Exceedances: var1={n_exc1}, var2={n_exc2}, joint={n_joint}")
    
    # Check if we have enough joint events
    if n_joint < min_joint_events:
        logger.warning(f"Only {n_joint} joint events found, which is less than the "
                      f"recommended minimum of {min_joint_events}")
    
    # Compute probabilities
    p_exc1 = n_exc1 / n_total
    p_exc2 = n_exc2 / n_total
    p_joint = n_joint / n_total
    p_ind = p_exc1 * p_exc2
    
    # Compute CPR
    cpr = p_joint / p_ind if p_ind > 0 else np.nan
    
    # Compute conditional probabilities
    p2_given_1 = (n_joint / n_exc1) if n_exc1 > 0 else np.nan
    p1_given_2 = (n_joint / n_exc2) if n_exc2 > 0 else np.nan
    
    # Return statistics
    stats = {
        'n_total': n_total,
        'n_exc1': n_exc1,
        'n_exc2': n_exc2,
        'n_joint': n_joint,
        'p_exc1': p_exc1,
        'p_exc2': p_exc2,
        'p_joint': p_joint,
        'p_independent': p_ind,
        'cpr': cpr,
        'p2_given_1': p2_given_1,
        'p1_given_2': p1_given_2
    }
    
    return stats


def compute_time_lag_dependency(
    da1: xr.DataArray,
    da2: xr.DataArray,
    threshold1: float,
    threshold2: float,
    max_lag_hours: int = 24
) -> Dict[str, Any]:
    """
    Compute dependency measures for different lead/lag configurations.
    
    Parameters
    ----------
    da1, da2 : xr.DataArray
        Time-aligned data arrays
    threshold1, threshold2 : float
        Exceedance thresholds
    max_lag_hours : int
        Maximum lag to consider in hours
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with lag analyses
    """
    # Validate inputs
    is_valid, error_msg = validate_data_arrays(da1, da2, threshold1, threshold2)
    if not is_valid:
        raise ValueError(f"Invalid inputs: {error_msg}")
    
    # Convert to pandas Series
    sx = da1.to_series()
    sy = da2.to_series()
    
    # Align on index and drop NaNs
    df = pd.DataFrame({'x': sx, 'y': sy}).dropna()
    n_total = len(df)
    
    if n_total == 0:
        raise ValueError("No overlapping non-NaN observations.")
    
    logger.info(f"Computing lag dependency up to ±{max_lag_hours} hours")
    
    # Set up lags to compute
    lags = list(range(-max_lag_hours, max_lag_hours + 1))
    
    # Results for each lag
    results = []
    
    # Base exceedances for variable 1
    exc1 = df['x'] > threshold1
    n_exc1 = int(exc1.sum())
    p_exc1 = n_exc1 / n_total
    
    # Compute statistics for each lag
    from scipy.stats import kendalltau
    
    for lag in lags:
        # Shift variable 2 by lag (positive lag means var2 follows var1)
        y_shifted = df['y'].shift(-lag)
        
        # Compute exceedances for shifted var2
        exc2 = y_shifted > threshold2
        n_exc2 = int(exc2.sum())
        p_exc2 = n_exc2 / n_valid if (n_valid := (~np.isnan(y_shifted)).sum()) > 0 else np.nan
        
        # Joint exceedances
        n_joint = int((exc1 & exc2).sum())
        p_joint = n_joint / n_valid if n_valid > 0 else np.nan
        
        # Dependence measures
        p_ind = p_exc1 * p_exc2
        cpr = p_joint / p_ind if p_ind > 0 else np.nan
        
        # Correlation
        try:
            # Use only finite values for correlation
            mask = ~np.isnan(df['x']) & ~np.isnan(y_shifted)
            if mask.sum() > 10:  # Need reasonable sample for correlation
                tau, pval = kendalltau(df['x'][mask], y_shifted[mask])
            else:
                tau, pval = np.nan, np.nan
        except:
            tau, pval = np.nan, np.nan
        
        results.append({
            'lag': lag,
            'n_joint': n_joint,
            'p_joint': p_joint,
            'p_independent': p_ind,
            'cpr': cpr,
            'kendall_tau': tau,
            'pvalue': pval
        })
    
    # Find the lag with the strongest dependence
    valid_results = [r for r in results if np.isfinite(r['cpr'])]
    if valid_results:
        max_cpr_result = max(valid_results, key=lambda r: r['cpr'])
        optimal_lag = max_cpr_result['lag']
    else:
        optimal_lag = 0
    
    return {
        'optimal_lag': optimal_lag,
        'lag_analysis': results
    }


def bootstrap_joint_stats(
    da1: xr.DataArray,
    da2: xr.DataArray,
    threshold1: float,
    threshold2: float,
    lag_hours: int = 0,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Dict[str, Any]:
    """
    Perform bootstrap resampling to estimate uncertainties in joint statistics.
    
    Parameters
    ----------
    da1, da2 : xr.DataArray
        Time-aligned data arrays
    threshold1, threshold2 : float
        Exceedance thresholds
    lag_hours : int
        Lag window for co-occurrence
    n_bootstrap : int
        Number of bootstrap samples
    ci_level : float
        Confidence level (0-1)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with bootstrap results
    """
    # Validate inputs
    is_valid, error_msg = validate_data_arrays(da1, da2, threshold1, threshold2)
    if not is_valid:
        raise ValueError(f"Invalid inputs: {error_msg}")
    
    # Convert to pandas Series
    sx = da1.to_series()
    sy = da2.to_series()
    
    # Align on index and drop NaNs
    df = pd.DataFrame({'x': sx, 'y': sy}).dropna()
    n_total = len(df)
    
    if n_total == 0:
        raise ValueError("No overlapping non-NaN observations.")
        
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap should be at least 100, got {n_bootstrap}")
        
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be between 0 and 1, got {ci_level}")
    
    logger.info(f"Performing bootstrap with {n_bootstrap} samples")
    
    # If lag window, compute rolling max of y
    if lag_hours > 0:
        window = 2 * lag_hours + 1
        df['y_roll'] = df['y'].rolling(window=window, center=True, min_periods=1).max()
    else:
        df['y_roll'] = df['y']
    
    # Arrays to store bootstrap results
    p_joint_boot = np.zeros(n_bootstrap)
    cpr_boot = np.zeros(n_bootstrap)
    p2_given_1_boot = np.zeros(n_bootstrap)
    p1_given_2_boot = np.zeros(n_bootstrap)
    
    # Exceedance indicators for original data (for reference)
    exc1_orig = df['x'] > threshold1
    exc2_orig = df['y_roll'] > threshold2
    n_exc1_orig = int(exc1_orig.sum())
    n_exc2_orig = int(exc2_orig.sum())
    n_joint_orig = int((exc1_orig & exc2_orig).sum())
    
    # Perform bootstrap resampling
    np.random.seed(42)  # For reproducibility
    indices = np.arange(n_total)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        boot_idx = np.random.choice(indices, size=n_total, replace=True)
        boot_df = df.iloc[boot_idx]
        
        # Compute exceedances
        exc1 = boot_df['x'] > threshold1
        exc2 = boot_df['y_roll'] > threshold2
        
        n_exc1 = int(exc1.sum())
        n_exc2 = int(exc2.sum())
        n_joint = int((exc1 & exc2).sum())
        
        # Compute statistics
        p_exc1 = n_exc1 / n_total
        p_exc2 = n_exc2 / n_total
        p_joint = n_joint / n_total
        p_ind = p_exc1 * p_exc2
        
        p_joint_boot[i] = p_joint
        cpr_boot[i] = p_joint / p_ind if p_ind > 0 else np.nan
        p2_given_1_boot[i] = n_joint / n_exc1 if n_exc1 > 0 else np.nan
        p1_given_2_boot[i] = n_joint / n_exc2 if n_exc2 > 0 else np.nan
    
    # Compute confidence intervals
    alpha = (1 - ci_level) / 2
    ci_level_pct = [100 * alpha, 100 * (1 - alpha)]
    
    # Function to compute percentiles, handling potential NaNs
    def get_percentiles(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return [np.nan, np.nan]
        return np.percentile(valid, ci_level_pct).tolist()
    
    # Compute percentiles
    p_joint_ci = get_percentiles(p_joint_boot)
    cpr_ci = get_percentiles(cpr_boot)
    p2_given_1_ci = get_percentiles(p2_given_1_boot)
    p1_given_2_ci = get_percentiles(p1_given_2_boot)
    
    # Compute bootstrap means
    p_joint_mean = float(np.nanmean(p_joint_boot))
    cpr_mean = float(np.nanmean(cpr_boot))
    p2_given_1_mean = float(np.nanmean(p2_given_1_boot))
    p1_given_2_mean = float(np.nanmean(p1_given_2_boot))
    
    # Compute bootstrap standard errors
    p_joint_se = float(np.nanstd(p_joint_boot))
    cpr_se = float(np.nanstd(cpr_boot))
    p2_given_1_se = float(np.nanstd(p2_given_1_boot))
    p1_given_2_se = float(np.nanstd(p1_given_2_boot))
    
    # Test for independence
    # H0: CPR = 1 (independence)
    cpr_original = n_joint_orig / (n_exc1_orig * n_exc2_orig / n_total) if n_exc1_orig > 0 and n_exc2_orig > 0 else np.nan
    
    # Compute p-value for independence test
    p_value = np.mean(cpr_boot <= 1) if np.isfinite(cpr_original) and cpr_original > 1 else np.nan
    
    return {
        'bootstrap_statistics': {
            'p_joint': {
                'mean': p_joint_mean,
                'se': p_joint_se,
                'ci': p_joint_ci,
                'values': p_joint_boot.tolist()  # Add the raw values
            },
            'cpr': {
                'mean': cpr_mean,
                'se': cpr_se,
                'ci': cpr_ci,
                'values': cpr_boot.tolist()  # Add the raw values
            },
            'p2_given_1': {
                'mean': p2_given_1_mean,
                'se': p2_given_1_se,
                'ci': p2_given_1_ci,
                'values': p2_given_1_boot.tolist()  # Add the raw values
            },
            'p1_given_2': {
                'mean': p1_given_2_mean,
                'se': p1_given_2_se,
                'ci': p1_given_2_ci,
                'values': p1_given_2_boot.tolist()  # Add the raw values
            }
        },
        # Rest of the structure remains the same
        'independence_test': {
            'cpr': cpr_original,
            'p_value': p_value,
            'reject_independence': p_value < 0.05 if np.isfinite(p_value) else None
        },
        'n_bootstrap': n_bootstrap,
        'ci_level': ci_level
    }


def save_stats(stats: Dict, filename: str, format: str = 'parquet') -> None:
    """
    Save statistics to file in specified format.
    
    Parameters
    ----------
    stats : Dict
        Statistics to save
    filename : str
        Output filename
    format : str, default 'parquet'
        Output format: 'parquet' or 'json'
    """
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    if format.lower() == 'parquet':
        # For Parquet, we need a flat DataFrame
        df = pd.DataFrame([stats])
        df.to_parquet(filename)
        logger.info(f"Saved statistics to {filename} in Parquet format")
    elif format.lower() == 'json':
        # For JSON we can save the nested dictionary directly
        import json
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {filename} in JSON format")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'json'")


if __name__ == '__main__':
    import argparse
    import sys
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Test joint empirical module')
    parser.add_argument('--netcdf', help='NetCDF file to analyze (optional)')
    parser.add_argument('--var1', default='sea_level', help='First variable name')
    parser.add_argument('--var2', default='total_precipitation', help='Second variable name')
    parser.add_argument('--threshold1', type=float, help='Threshold for var1 (optional)')
    parser.add_argument('--threshold2', type=float, help='Threshold for var2 (optional)')
    parser.add_argument('--percentile1', type=float, default=0.95, help='Percentile for var1 threshold (if threshold1 not provided)')
    parser.add_argument('--percentile2', type=float, default=0.95, help='Percentile for var2 threshold (if threshold2 not provided)')
    parser.add_argument('--lag-hours', type=int, default=0, help='Lag window for co-occurrence')
    parser.add_argument('--lag-analysis', action='store_true', help='Perform lag analysis')
    parser.add_argument('--max-lag', type=int, default=24, help='Maximum lag for analysis')
    parser.add_argument('--bootstrap', action='store_true', help='Perform bootstrap uncertainty quantification')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Number of bootstrap samples')
    parser.add_argument('--plot', action='store_true', help='Generate diagnostic plots')
    args = parser.parse_args()
    
    try:
        if args.netcdf:
            # Real data test
            print(f"Testing with real data from {args.netcdf}")
            ds = xr.open_dataset(args.netcdf)
            
            if args.var1 not in ds or args.var2 not in ds:
                available_vars = list(ds.data_vars)
                print(f"Error: Variables not found in NetCDF. Available variables: {available_vars}")
                sys.exit(1)
                
            da1 = ds[args.var1]
            da2 = ds[args.var2]
            
            print(f"Variables: {args.var1}, {args.var2}")
            print(f"Data shapes: {da1.shape}, {da2.shape}")
            print(f"Data ranges: {float(da1.min()):.4f} to {float(da1.max()):.4f}, {float(da2.min()):.4f} to {float(da2.max()):.4f}")
            print(f"Missing values: {np.isnan(da1.values).sum()}, {np.isnan(da2.values).sum()}")
            
            # Determine thresholds
            threshold1 = args.threshold1
            threshold2 = args.threshold2
            
            if threshold1 is None:
                arr1 = da1.values.flatten()
                arr1 = arr1[~np.isnan(arr1)]
                threshold1 = float(np.quantile(arr1, args.percentile1))
                print(f"Using threshold for {args.var1} at {args.percentile1:.2%} percentile: {threshold1:.4f}")
            
            if threshold2 is None:
                arr2 = da2.values.flatten()
                arr2 = arr2[~np.isnan(arr2)]
                threshold2 = float(np.quantile(arr2, args.percentile2))
                print(f"Using threshold for {args.var2} at {args.percentile2:.2%} percentile: {threshold2:.4f}")
            
            # Compute empirical statistics
            try:
                emp_stats = compute_empirical_stats(
                    da1=da1,
                    da2=da2,
                    threshold1=threshold1,
                    threshold2=threshold2,
                    lag_hours=args.lag_hours
                )
                
                print("\nEmpirical statistics:")
                print(f"  Total observations: {emp_stats['n_total']}")
                print(f"  {args.var1} exceedances: {emp_stats['n_exc1']} ({emp_stats['p_exc1']:.4f})")
                print(f"  {args.var2} exceedances: {emp_stats['n_exc2']} ({emp_stats['p_exc2']:.4f})")
                print(f"  Joint exceedances: {emp_stats['n_joint']} ({emp_stats['p_joint']:.4f})")
                print(f"  Independent probability: {emp_stats['p_independent']:.4f}")
                print(f"  Conditional probability ratio (CPR): {emp_stats['cpr']:.4f}")
                print(f"  P({args.var2} | {args.var1}): {emp_stats['p2_given_1']:.4f}")
                print(f"  P({args.var1} | {args.var2}): {emp_stats['p1_given_2']:.4f}")
                
                # Perform lag analysis if requested
                if args.lag_analysis:
                    print(f"\nPerforming lag analysis up to ±{args.max_lag} hours...")
                    lag_results = compute_time_lag_dependency(
                        da1=da1,
                        da2=da2,
                        threshold1=threshold1,
                        threshold2=threshold2,
                        max_lag_hours=args.max_lag
                    )
                    
                    optimal_lag = lag_results['optimal_lag']
                    print(f"  Optimal lag: {optimal_lag} hours")
                    print(f"  (Positive lag means {args.var2} follows {args.var1})")
                
                # Perform bootstrap if requested
                if args.bootstrap:
                    print(f"\nPerforming bootstrap with {args.n_bootstrap} samples...")
                    bootstrap_results = bootstrap_joint_stats(
                        da1=da1,
                        da2=da2,
                        threshold1=threshold1,
                        threshold2=threshold2,
                        lag_hours=args.lag_hours,
                        n_bootstrap=args.n_bootstrap
                    )
                    
                    print("\nBootstrap results:")
                    print(f"  CPR: {bootstrap_results['bootstrap_statistics']['cpr']['mean']:.4f} ± {bootstrap_results['bootstrap_statistics']['cpr']['se']:.4f}")
                    print(f"  95% CI: [{bootstrap_results['bootstrap_statistics']['cpr']['ci'][0]:.4f}, {bootstrap_results['bootstrap_statistics']['cpr']['ci'][1]:.4f}]")
                    
                    # Test for independence
                    p_value = bootstrap_results['independence_test']['p_value']
                    if np.isfinite(p_value):
                        print(f"  Independence test p-value: {p_value:.4f}")
                        print(f"  Reject independence at α=0.05: {p_value < 0.05}")
                    else:
                        print("  Independence test: insufficient data")
                
                # Generate plots if requested
                if args.plot:
                    # Create joint exceedance plot
                    plt.figure(figsize=(12, 10))
                    
                    # Convert to pandas for plotting
                    df = pd.DataFrame({
                        args.var1: da1.to_series(),
                        args.var2: da2.to_series()
                    }).dropna()
                    
                    # Scatter plot with thresholds
                    plt.subplot(2, 2, 1)
                    plt.scatter(df[args.var1], df[args.var2], alpha=0.3, s=5)
                    plt.axvline(threshold1, color='r', linestyle='--', label=f'{args.var1} threshold')
                    plt.axhline(threshold2, color='g', linestyle='--', label=f'{args.var2} threshold')
                    plt.xlabel(args.var1)
                    plt.ylabel(args.var2)
                    plt.title('Joint Scatter Plot')
                    plt.legend()
                    
                    if args.lag_analysis:
                        # Lag dependency plot
                        plt.subplot(2, 2, 2)
                        
                        lag_df = pd.DataFrame(lag_results['lag_analysis'])
                        plt.plot(lag_df['lag'], lag_df['cpr'], 'o-')
                        plt.axhline(1.0, color='r', linestyle='--', label='Independence (CPR=1)')
                        plt.axvline(optimal_lag, color='g', linestyle=':', label=f'Optimal lag={optimal_lag}h')
                        plt.xlabel('Lag (hours)')
                        plt.ylabel('CPR')
                        plt.title('Lag Dependency Analysis')
                        plt.grid(True)
                        plt.legend()
                        
                        # Kendall's tau by lag
                        plt.subplot(2, 2, 3)
                        plt.plot(lag_df['lag'], lag_df['kendall_tau'], 'o-')
                        plt.axhline(0.0, color='r', linestyle='--', label='No correlation')
                        plt.axvline(optimal_lag, color='g', linestyle=':', label=f'Optimal lag={optimal_lag}h')
                        plt.xlabel('Lag (hours)')
                        plt.ylabel("Kendall's tau")
                        plt.title('Rank Correlation by Lag')
                        plt.grid(True)
                        plt.legend()
                    
                    if args.bootstrap:
                        if args.lag_analysis:
                            plt.subplot(2, 2, 4)
                        else:
                            plt.subplot(2, 2, 2)
                        
                        # Extract bootstrap samples and filter out NaNs
                        cpr_samples = np.array(bootstrap_results['bootstrap_statistics']['cpr']['values'])
                        cpr_samples = cpr_samples[~np.isnan(cpr_samples)]
                        
                        plt.hist(cpr_samples, bins=30, alpha=0.7)
                        plt.axvline(emp_stats['cpr'], color='r', linestyle='--',
                                   label=f'Sample CPR: {emp_stats["cpr"]:.4f}')
                        plt.axvline(1.0, color='g', linestyle=':',
                                   label='Independence (CPR=1)')
                        ci = bootstrap_results['bootstrap_statistics']['cpr']['ci']
                        if np.isfinite(ci[0]) and np.isfinite(ci[1]):
                            plt.axvline(ci[0], color='b', linestyle='-.',
                                      label=f'95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]')
                            plt.axvline(ci[1], color='b', linestyle='-.')
                        plt.xlabel('CPR value')
                        plt.ylabel('Frequency')
                        plt.title('Bootstrap CPR Distribution')
                        plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Error in empirical analysis: {type(e).__name__}: {e}")
        else:
            # Synthetic data test
            print("Testing with synthetic data")
            
            # Create synthetic correlated data
            np.random.seed(42)
            n = 1000
            
            # Generate correlated random variables with known dependence
            means = [0, 0]
            corr = 0.7  # Set correlation strength
            cov = np.array([[1, corr], [corr, 1]])
            
            # Generate bivariate normal
            data = np.random.multivariate_normal(means, cov, n)
            x = data[:, 0]
            y = data[:, 1]
            
            # Create xarray DataArrays
            time_coords = pd.date_range('2000-01-01', periods=n, freq='h')
            da1 = xr.DataArray(x, dims=['datetime'], coords={'datetime': time_coords})
            da2 = xr.DataArray(y, dims=['datetime'], coords={'datetime': time_coords})
            
            print(f"Generated {n} synthetic data points with correlation {corr}")
            
            # Set thresholds at 95th percentile
            threshold1 = float(np.quantile(x, 0.95))
            threshold2 = float(np.quantile(y, 0.95))
            
            print(f"Using thresholds: {threshold1:.4f}, {threshold2:.4f}")
            
            # Compute empirical statistics
            try:
                emp_stats = compute_empirical_stats(
                    da1=da1,
                    da2=da2,
                    threshold1=threshold1,
                    threshold2=threshold2,
                    lag_hours=args.lag_hours
                )
                
                print("\nEmpirical statistics:")
                print(f"  Total observations: {emp_stats['n_total']}")
                print(f"  Variable 1 exceedances: {emp_stats['n_exc1']} ({emp_stats['p_exc1']:.4f})")
                print(f"  Variable 2 exceedances: {emp_stats['n_exc2']} ({emp_stats['p_exc2']:.4f})")
                print(f"  Joint exceedances: {emp_stats['n_joint']} ({emp_stats['p_joint']:.4f})")
                print(f"  Independent probability: {emp_stats['p_independent']:.4f}")
                print(f"  Conditional probability ratio (CPR): {emp_stats['cpr']:.4f}")
                
                # Perform bootstrap
                bootstrap_results = bootstrap_joint_stats(
                    da1=da1,
                    da2=da2,
                    threshold1=threshold1,
                    threshold2=threshold2,
                    lag_hours=args.lag_hours,
                    n_bootstrap=500  # Use fewer samples for synthetic test
                )
                
                print("\nBootstrap results:")
                print(f"  CPR: {bootstrap_results['bootstrap_statistics']['cpr']['mean']:.4f} ± {bootstrap_results['bootstrap_statistics']['cpr']['se']:.4f}")
                print(f"  95% CI: [{bootstrap_results['bootstrap_statistics']['cpr']['ci'][0]:.4f}, {bootstrap_results['bootstrap_statistics']['cpr']['ci'][1]:.4f}]")
                
                # Test for independence
                p_value = bootstrap_results['independence_test']['p_value']
                if np.isfinite(p_value):
                    print(f"  Independence test p-value: {p_value:.4f}")
                    print(f"  Reject independence at α=0.05: {p_value < 0.05}")
                
                # Generate plots if requested
                if args.plot:
                    plt.figure(figsize=(12, 8))
                    
                    # Scatter plot with thresholds
                    plt.subplot(2, 2, 1)
                    plt.scatter(x, y, alpha=0.3, s=5)
                    plt.axvline(threshold1, color='r', linestyle='--', label='Var1 threshold')
                    plt.axhline(threshold2, color='g', linestyle='--', label='Var2 threshold')
                    plt.xlabel('Variable 1')
                    plt.ylabel('Variable 2')
                    plt.title(f'Joint Scatter Plot (ρ={corr})')
                    plt.legend()
                    
                    # Highlight joint exceedances
                    plt.subplot(2, 2, 2)
                    joint_mask = (x > threshold1) & (y > threshold2)
                    x1_only = (x > threshold1) & (y <= threshold2)
                    x2_only = (x <= threshold1) & (y > threshold2)
                    neither = (x <= threshold1) & (y <= threshold2)
                    
                    plt.scatter(x[neither], y[neither], alpha=0.2, s=5, color='gray', label='Neither')
                    plt.scatter(x[x1_only], y[x1_only], alpha=0.5, s=10, color='blue', label='Var1 only')
                    plt.scatter(x[x2_only], y[x2_only], alpha=0.5, s=10, color='green', label='Var2 only')
                    plt.scatter(x[joint_mask], y[joint_mask], alpha=0.7, s=20, color='red', label='Joint')
                    
                    plt.axvline(threshold1, color='r', linestyle='--')
                    plt.axhline(threshold2, color='g', linestyle='--')
                    plt.xlabel('Variable 1')
                    plt.ylabel('Variable 2')
                    plt.title('Exceedance Categories')
                    plt.legend()
                    
                    # Bootstrap histogram for CPR
                    plt.subplot(2, 2, 3)
                    cpr_samples = np.array(bootstrap_results['bootstrap_statistics']['cpr']['values'])
                    cpr_samples = cpr_samples[~np.isnan(cpr_samples)]
                    
                    plt.hist(cpr_samples, bins=30, alpha=0.7)
                    plt.axvline(emp_stats['cpr'], color='r', linestyle='--',
                               label=f'Sample CPR: {emp_stats["cpr"]:.4f}')
                    plt.axvline(1.0, color='g', linestyle=':',
                               label='Independence (CPR=1)')
                    ci = bootstrap_results['bootstrap_statistics']['cpr']['ci']
                    plt.axvline(ci[0], color='b', linestyle='-.',
                              label=f'95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]')
                    plt.axvline(ci[1], color='b', linestyle='-.')
                    plt.xlabel('CPR value')
                    plt.ylabel('Frequency')
                    plt.title('Bootstrap CPR Distribution')
                    plt.legend()
                    
                    # Theoretical vs. empirical joint probability
                    plt.subplot(2, 2, 4)
                    p_values = np.linspace(0.5, 0.99, 50)
                    theo_joint = np.zeros_like(p_values)
                    emp_joint = np.zeros_like(p_values)
                    
                    for i, p in enumerate(p_values):
                        thr1 = float(np.quantile(x, p))
                        thr2 = float(np.quantile(y, p))
                        
                        # Empirical joint probability
                        emp_joint[i] = np.mean((x > thr1) & (y > thr2))
                        
                        # Theoretical joint probability under independence
                        theo_joint[i] = (1 - p) ** 2
                    
                    plt.plot(p_values, emp_joint, 'o-', label='Empirical joint')
                    plt.plot(p_values, theo_joint, 'r--', label='Independent case')
                    plt.xlabel('Threshold probability')
                    plt.ylabel('Joint exceedance probability')
                    plt.title('Empirical vs. Independent Joint Probability')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Error in empirical analysis: {type(e).__name__}: {e}")
        
        print("\nJoint empirical module smoke test completed successfully.")
    except Exception as e:
        print(f"Error during test: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)