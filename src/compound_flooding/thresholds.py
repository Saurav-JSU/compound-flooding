# src/compound_flooding/thresholds.py
"""
Module: thresholds.py
Responsibilities:
- Compute candidate POT thresholds (percentile-based)
- Compute mean residual life (mean excess) diagnostic data
- Provide helper to select threshold at a given percentile
- Find optimal threshold using mean residual life plot analysis
"""
import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Optional, Union, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_threshold(
    da: xr.DataArray,
    percentile: float = 0.99,
    min_sample_size: int = 30
) -> float:
    """
    Select threshold at the given percentile from the DataArray, ignoring NaNs.

    Parameters
    ----------
    da : xr.DataArray
        Input data array of values (e.g., sea level or precipitation).
    percentile : float
        Percentile for threshold selection (e.g., 0.99 for 99th percentile).
        Must be between 0 and 1.
    min_sample_size : int
        Minimum required non-NaN values. Default is 30.

    Returns
    -------
    float
        Threshold value at specified percentile.
        
    Raises
    ------
    ValueError
        If percentile is outside the range [0, 1] or DataArray has insufficient data.
    TypeError
        If input is not an xarray DataArray.
    """
    # Validate input type
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray")
    
    # Validate percentile
    if not 0 <= percentile <= 1:
        raise ValueError(f"Percentile must be between 0 and 1, got {percentile}")
    
    # Extract values, ignoring NaNs
    arr = da.values.flatten()
    arr = arr[~np.isnan(arr)]
    
    # Check if we have enough data
    if arr.size == 0:
        raise ValueError("DataArray contains only NaNs; cannot compute threshold.")
    if arr.size < min_sample_size:
        logger.warning(f"DataArray contains only {arr.size} non-NaN values, which is less than "
                      f"recommended minimum of {min_sample_size}")
    
    # Compute threshold
    threshold = float(np.quantile(arr, percentile))
    logger.info(f"Selected threshold at {percentile:.2%} percentile: {threshold:.4f}")
    
    # Check if threshold produces reasonable number of exceedances
    exceedances = arr[arr > threshold]
    n_exceed = len(exceedances)
    if n_exceed < min_sample_size:
        logger.warning(f"Threshold {threshold:.4f} yields only {n_exceed} exceedances, "
                      f"which is less than recommended minimum of {min_sample_size}")
    
    return threshold


def mean_residual_life(
    da: xr.DataArray,
    thresholds: Union[np.ndarray, List[float]],
    min_exceedances: int = 10
) -> pd.DataFrame:
    """
    Compute mean residual life (mean excess over threshold) for each threshold.
    
    The mean residual life plot is a diagnostic tool for selecting threshold u for
    POT/GPD analysis. A valid threshold is one above which the mean excess is
    approximately linear in u.

    Parameters
    ----------
    da : xr.DataArray
        Input data array of values.
    thresholds : np.ndarray or List[float]
        Array of threshold candidates in ascending order.
    min_exceedances : int
        Minimum number of exceedances required to compute mean excess.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['threshold', 'mean_excess', 'n_exceed', 'std_err']
    
    Raises
    ------
    ValueError
        If no thresholds are provided or all values in da are NaN.
    TypeError
        If input is not an xarray DataArray or thresholds is not an array-like.
    """
    # Validate inputs
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray")
        
    if not isinstance(thresholds, (np.ndarray, list)):
        raise TypeError("thresholds must be a numpy array or list")
        
    if len(thresholds) == 0:
        raise ValueError("thresholds must not be empty")
    
    # Ensure thresholds are a numpy array
    thresholds = np.asarray(thresholds)
    
    # Extract values, ignoring NaNs
    arr = da.values.flatten()
    arr = arr[~np.isnan(arr)]
    
    # Check if we have any data
    if arr.size == 0:
        raise ValueError("DataArray contains only NaNs; cannot compute mean residual life.")
    
    # Compute mean residual life for each threshold
    results = []
    for u in thresholds:
        exceedances = arr[arr > u]
        n_exceed = len(exceedances)
        
        if n_exceed >= min_exceedances:
            # Calculate excess values
            excess = exceedances - u
            # Compute mean excess
            mean_exc = np.mean(excess)
            # Compute standard error
            std_err = np.std(excess, ddof=1) / np.sqrt(n_exceed)
        else:
            mean_exc = np.nan
            std_err = np.nan
        
        results.append({
            'threshold': float(u),
            'mean_excess': float(mean_exc),
            'n_exceed': int(n_exceed),
            'std_err': float(std_err)
        })
    
    return pd.DataFrame(results)


def generate_threshold_candidates(
    da: xr.DataArray,
    percentiles: Optional[List[float]] = None,
    n_points: int = 20,
    min_percentile: float = 0.8,
    max_percentile: float = 0.995
) -> np.ndarray:
    """
    Generate candidate thresholds based on percentiles of the data.

    Parameters
    ----------
    da : xr.DataArray
        Input data array.
    percentiles : List[float], optional
        List of percentiles to use (must be between 0 and 1).
        If not provided, generates n_points evenly spaced percentiles.
    n_points : int
        Number of threshold candidates to generate if percentiles is None.
    min_percentile : float
        Minimum percentile to consider if generating evenly spaced percentiles.
    max_percentile : float
        Maximum percentile to consider if generating evenly spaced percentiles.

    Returns
    -------
    np.ndarray
        Array of threshold candidates.
    
    Raises
    ------
    ValueError
        If percentile values are invalid.
    """
    # Validate inputs
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray")
    
    # Generate percentiles if not provided
    if percentiles is None:
        percentiles = np.linspace(min_percentile, max_percentile, n_points)
    else:
        # Validate provided percentiles
        if not all(0 <= p <= 1 for p in percentiles):
            raise ValueError("All percentiles must be between 0 and 1")
    
    # Extract values, ignoring NaNs
    arr = da.values.flatten()
    arr = arr[~np.isnan(arr)]
    
    # Check if we have any data
    if arr.size == 0:
        raise ValueError("DataArray contains only NaNs; cannot compute thresholds.")
    
    # Compute thresholds
    thresholds = np.array([np.quantile(arr, p) for p in percentiles])
    
    return thresholds


def find_optimal_threshold(
    da: xr.DataArray,
    min_percentile: float = 0.8,
    max_percentile: float = 0.995,
    n_points: int = 20,
    min_exceedances: int = 30
) -> Tuple[float, pd.DataFrame]:
    """
    Find an optimal threshold for GPD modeling using mean residual life plot analysis.
    
    The optimal threshold is one above which the mean excess plot is approximately linear,
    indicating the asymptotic GPD assumption is valid.

    Parameters
    ----------
    da : xr.DataArray
        Input data array.
    min_percentile : float
        Minimum percentile to consider.
    max_percentile : float
        Maximum percentile to consider.
    n_points : int
        Number of threshold candidates to generate.
    min_exceedances : int
        Minimum number of exceedances required for valid GPD fit.

    Returns
    -------
    Tuple[float, pd.DataFrame]
        Optimal threshold and mean residual life data.
        
    Notes
    -----
    This is a heuristic method using linear regions in the mean residual life plot.
    The returned threshold is not guaranteed to be optimal in all cases.
    Visual inspection of the mean residual life plot is recommended.
    """
    # Generate threshold candidates
    thresholds = generate_threshold_candidates(
        da, 
        n_points=n_points,
        min_percentile=min_percentile,
        max_percentile=max_percentile
    )
    
    # Compute mean residual life
    mrl_data = mean_residual_life(da, thresholds, min_exceedances=min_exceedances)
    
    # Filter out thresholds with too few exceedances
    valid_mrl = mrl_data[mrl_data['n_exceed'] >= min_exceedances]
    
    if len(valid_mrl) < 3:
        # Not enough valid points for analysis
        logger.warning(f"Not enough valid thresholds with {min_exceedances} exceedances.")
        if len(mrl_data) > 0:
            # Return the threshold with the most exceedances
            optimal_idx = mrl_data['n_exceed'].idxmax()
            optimal_threshold = mrl_data.loc[optimal_idx, 'threshold']
            logger.info(f"Using threshold with most exceedances: {optimal_threshold:.4f}")
            return optimal_threshold, mrl_data
        else:
            # No valid thresholds at all
            raise ValueError("No valid thresholds found for mean residual life analysis.")
    
    # Compute slopes between consecutive points
    valid_thresholds = valid_mrl['threshold'].values
    valid_mrl_values = valid_mrl['mean_excess'].values
    
    slopes = np.zeros(len(valid_thresholds) - 1)
    for i in range(len(slopes)):
        dx = valid_thresholds[i+1] - valid_thresholds[i]
        dy = valid_mrl_values[i+1] - valid_mrl_values[i]
        slopes[i] = dy / dx if dx != 0 else np.nan
    
    # Find regions where slope stabilizes (becomes more linear)
    slope_diffs = np.abs(np.diff(slopes))
    
    # If slope differences are available, find where they are small (linear region)
    if len(slope_diffs) > 0:
        # Find the point where the slope starts to stabilize
        stable_idx = np.argmin(slope_diffs) + 1  # +1 because we want the second point of the diff
        optimal_threshold = valid_thresholds[stable_idx]
        
        # Ensure this threshold has enough exceedances
        n_exceed = valid_mrl.loc[valid_mrl['threshold'] == optimal_threshold, 'n_exceed'].iloc[0]
        
        logger.info(f"Found optimal threshold: {optimal_threshold:.4f} with {n_exceed} exceedances")
    else:
        # If we can't do slope analysis, use the middle point
        middle_idx = len(valid_thresholds) // 2
        optimal_threshold = valid_thresholds[middle_idx]
        n_exceed = valid_mrl.loc[valid_mrl['threshold'] == optimal_threshold, 'n_exceed'].iloc[0]
        
        logger.info(f"Using middle threshold as optimal: {optimal_threshold:.4f} with {n_exceed} exceedances")
    
    return optimal_threshold, mrl_data


if __name__ == '__main__':
    import argparse
    import sys
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Test thresholds module')
    parser.add_argument('--netcdf', help='NetCDF file to analyze (optional)')
    parser.add_argument('--variable', default='sea_level', help='Variable to analyze')
    parser.add_argument('--percentile', type=float, default=0.99, help='Percentile for threshold selection')
    parser.add_argument('--min-percentile', type=float, default=0.8, help='Minimum percentile for MRL plot')
    parser.add_argument('--max-percentile', type=float, default=0.995, help='Maximum percentile for MRL plot')
    parser.add_argument('--n-points', type=int, default=20, help='Number of points for MRL plot')
    parser.add_argument('--plot', action='store_true', help='Generate and show diagnostic plots')
    args = parser.parse_args()
    
    try:
        if args.netcdf:
            # Real data test
            print(f"Testing with real data from {args.netcdf}")
            ds = xr.open_dataset(args.netcdf)
            
            if args.variable not in ds:
                print(f"Error: Variable '{args.variable}' not found in NetCDF. Available variables: {list(ds.data_vars)}")
                sys.exit(1)
                
            da = ds[args.variable]
            print(f"Variable: {args.variable}")
            print(f"Data shape: {da.shape}")
            print(f"Data range: {float(da.min())} to {float(da.max())}")
            print(f"Missing values: {np.isnan(da.values).sum()}")
            
            # Simple threshold at percentile
            simple_thr = select_threshold(da, args.percentile)
            print(f"\nSimple threshold at {args.percentile:.2%}: {simple_thr:.4f}")
            
            # Generate candidate thresholds
            thresholds = generate_threshold_candidates(
                da, 
                n_points=args.n_points,
                min_percentile=args.min_percentile,
                max_percentile=args.max_percentile
            )
            
            # Compute mean residual life
            print(f"\nComputing mean residual life for {len(thresholds)} thresholds...")
            mrl_data = mean_residual_life(da, thresholds)
            print(mrl_data.head())
            
            # Find optimal threshold
            print("\nFinding optimal threshold using mean residual life plot...")
            optimal_thr, _ = find_optimal_threshold(
                da,
                min_percentile=args.min_percentile,
                max_percentile=args.max_percentile,
                n_points=args.n_points
            )
            
            print(f"Optimal threshold: {optimal_thr:.4f}")
            
            # Generate plots if requested
            if args.plot:
                try:
                    # Mean residual life plot
                    plt.figure(figsize=(10, 6))
                    plt.errorbar(
                        mrl_data['threshold'],
                        mrl_data['mean_excess'],
                        yerr=mrl_data['std_err'],
                        fmt='o-',
                        capsize=5
                    )
                    plt.axvline(x=optimal_thr, color='r', linestyle='--', label=f'Optimal threshold: {optimal_thr:.4f}')
                    plt.axvline(x=simple_thr, color='g', linestyle=':', label=f'{args.percentile:.2%} threshold: {simple_thr:.4f}')
                    plt.xlabel('Threshold')
                    plt.ylabel('Mean Excess')
                    plt.title(f'Mean Residual Life Plot for {args.variable}')
                    plt.grid(True)
                    plt.legend()
                    
                    # Number of exceedances
                    plt.figure(figsize=(10, 6))
                    plt.plot(mrl_data['threshold'], mrl_data['n_exceed'], 'o-')
                    plt.axvline(x=optimal_thr, color='r', linestyle='--', label=f'Optimal threshold: {optimal_thr:.4f}')
                    plt.axvline(x=simple_thr, color='g', linestyle=':', label=f'{args.percentile:.2%} threshold: {simple_thr:.4f}')
                    plt.xlabel('Threshold')
                    plt.ylabel('Number of Exceedances')
                    plt.title(f'Number of Exceedances for {args.variable}')
                    plt.grid(True)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error creating plots: {e}")
            
        else:
            # Synthetic data test
            print("Testing with synthetic data")
            
            # Create synthetic data with known threshold
            true_threshold = 1.5
            np.random.seed(42)
            
            # Generate regular data below threshold
            n_regular = 1000
            regular_data = np.random.normal(0, 0.5, n_regular)
            
            # Generate GPD data above threshold
            # Parameters: shape (xi) = 0.2, scale (sigma) = 0.5
            n_extreme = 100
            xi = 0.2
            sigma = 0.5
            
            # Generate GPD variates
            q = np.random.uniform(0, 1, n_extreme)
            extreme_data = true_threshold + (sigma/xi) * ((1-q)**(-xi) - 1)
            
            # Combine data
            data = np.concatenate([regular_data[regular_data <= true_threshold], extreme_data])
            
            # Convert to xarray
            da = xr.DataArray(data, dims=['time'], coords={'time': np.arange(len(data))})
            
            print(f"Generated {len(data)} synthetic data points with true threshold {true_threshold}")
            print(f"Data range: {data.min():.4f} to {data.max():.4f}")
            
            # Test simple threshold selection
            for p in [0.9, 0.95, 0.99]:
                thr = select_threshold(da, p)
                print(f"Threshold at {p:.2%} percentile: {thr:.4f}")
            
            # Test mean residual life
            thresholds = np.linspace(0.0, 3.0, 30)
            mrl_data = mean_residual_life(da, thresholds)
            print("\nMean Residual Life Data:")
            print(mrl_data.head())
            
            # Test optimal threshold finding
            optimal_thr, _ = find_optimal_threshold(da)
            print(f"\nOptimal threshold: {optimal_thr:.4f} (true: {true_threshold:.4f})")
            
            # Create diagnostic plots
            if args.plot:
                try:
                    # Data histogram
                    plt.figure(figsize=(12, 8))
                    
                    plt.subplot(2, 2, 1)
                    plt.hist(data, bins=30, alpha=0.7, density=True)
                    plt.axvline(x=true_threshold, color='r', linestyle='--', label=f'True threshold: {true_threshold}')
                    plt.axvline(x=optimal_thr, color='g', linestyle=':', label=f'Optimal threshold: {optimal_thr:.4f}')
                    plt.xlabel('Value')
                    plt.ylabel('Density')
                    plt.title('Synthetic Data Histogram')
                    plt.legend()
                    
                    # Mean residual life plot
                    plt.subplot(2, 2, 2)
                    plt.errorbar(
                        mrl_data['threshold'],
                        mrl_data['mean_excess'],
                        yerr=mrl_data['std_err'],
                        fmt='o-',
                        capsize=5
                    )
                    plt.axvline(x=true_threshold, color='r', linestyle='--', label=f'True threshold: {true_threshold}')
                    plt.axvline(x=optimal_thr, color='g', linestyle=':', label=f'Optimal threshold: {optimal_thr:.4f}')
                    plt.xlabel('Threshold')
                    plt.ylabel('Mean Excess')
                    plt.title('Mean Residual Life Plot')
                    plt.grid(True)
                    plt.legend()
                    
                    # Number of exceedances
                    plt.subplot(2, 2, 3)
                    plt.plot(mrl_data['threshold'], mrl_data['n_exceed'], 'o-')
                    plt.axvline(x=true_threshold, color='r', linestyle='--', label=f'True threshold: {true_threshold}')
                    plt.axvline(x=optimal_thr, color='g', linestyle=':', label=f'Optimal threshold: {optimal_thr:.4f}')
                    plt.xlabel('Threshold')
                    plt.ylabel('Number of Exceedances')
                    plt.title('Number of Exceedances')
                    plt.grid(True)
                    plt.legend()
                    
                    # QQ plot of exceedances above optimal threshold
                    plt.subplot(2, 2, 4)
                    excess = data[data > optimal_thr] - optimal_thr
                    excess.sort()
                    n = len(excess)
                    
                    # Theoretical quantiles from fitted GPD
                    pp = np.arange(1, n+1) / (n+1)  # plotting positions
                    theoretical_quantiles = (sigma/xi) * ((1-pp)**(-xi) - 1)
                    
                    plt.plot(theoretical_quantiles, excess, 'o')
                    max_val = max(theoretical_quantiles.max(), excess.max())
                    plt.plot([0, max_val], [0, max_val], 'r--')
                    plt.xlabel('Theoretical Quantiles')
                    plt.ylabel('Empirical Quantiles')
                    plt.title('QQ Plot of Excesses over Optimal Threshold')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error creating plots: {e}")
        
        print("\nThresholds module smoke test completed successfully.")
    except Exception as e:
        print(f"Error during test: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)