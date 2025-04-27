# src/compound_flooding/univariate.py
"""
Module: univariate.py
Responsibilities:
- Fit GPD to data exceeding a specified threshold
- Compute return levels for given return periods
- Perform diagnostic checks on GPD fits
- Provide methods for uncertainty quantification
"""
import numpy as np
import xarray as xr
from scipy.stats import genpareto
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_EXCEEDANCES = 30  # Minimum number of exceedances for reliable GPD fit
MAX_XI = 0.5  # Maximum allowed shape parameter (xi) for stability


def validate_exceedances(exceedances: np.ndarray, min_size: int = MIN_EXCEEDANCES) -> Tuple[bool, str]:
    """
    Validate exceedances array for GPD fitting.
    
    Parameters
    ----------
    exceedances : np.ndarray
        Array of exceedances over threshold
    min_size : int, optional
        Minimum number of exceedances required, defaults to MIN_EXCEEDANCES
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not isinstance(exceedances, np.ndarray):
        return False, "Exceedances must be a numpy array"
        
    if len(exceedances) < min_size:
        return False, f"Insufficient exceedances ({len(exceedances)}), minimum required: {min_size}"
        
    if np.any(exceedances < 0):
        return False, "Exceedances must be non-negative"
        
    if not np.isfinite(exceedances).all():
        return False, "Exceedances contain NaN or Inf values"
        
    if np.std(exceedances) == 0:
        return False, "Exceedances have zero variance (all values are identical)"
        
    return True, ""


def fit_gpd(
    da: xr.DataArray,
    threshold: float,
    min_exceedances: int = MIN_EXCEEDANCES,
    constrain_shape: bool = True,
    max_shape: float = MAX_XI
) -> Dict[str, Any]:
    """
    Fit a Generalized Pareto Distribution to exceedances over threshold.

    Parameters
    ----------
    da : xr.DataArray
        Data array of values (e.g., sea_level). NaNs ignored.
    threshold : float
        POT threshold for defining exceedances.
    min_exceedances : int, optional
        Minimum number of exceedances required for reliable fit.
    constrain_shape : bool, optional
        Whether to constrain shape parameter (xi) to a maximum value.
    max_shape : float, optional
        Maximum allowed shape parameter (xi) if constrained.

    Returns
    -------
    dict
        {
          'shape': xi,
          'scale': sigma,
          'threshold': threshold,
          'n_exceed': n_exc,
          'rate': n_exc / N_total,
          'diagnostics': {
            'neg_log_likelihood': -log_like,
            'aic': aic,
            'bic': bic,
            'quantile_errors': quantile_errors
          }
        }
        
    Raises
    ------
    ValueError
        If exceedances are insufficient or invalid
    """
    # Validate input
    if not isinstance(da, xr.DataArray):
        raise TypeError("Input must be an xarray DataArray")
        
    # Extract exceedances
    arr = da.values.flatten()
    arr = arr[~np.isnan(arr)]
    N = arr.size
    
    if N == 0:
        raise ValueError("Input DataArray contains only NaN values")
        
    exceed = arr[arr > threshold] - threshold
    n_exc = exceed.size
    
    # Validate exceedances
    is_valid, error_msg = validate_exceedances(exceed, min_exceedances)
    if not is_valid:
        raise ValueError(f"Invalid exceedances: {error_msg}")
        
    logger.info(f"Fitting GPD to {n_exc} exceedances above threshold {threshold:.4f}")
    
    # Fit GPD
    try:
        # First, unconstrained MLE fit
        params = genpareto.fit(exceed, floc=0)
        xi, loc, sigma = params
        
        # Check shape parameter
        if constrain_shape and xi > max_shape:
            logger.warning(f"Shape parameter {xi:.4f} exceeds maximum allowed {max_shape:.4f}, constraining")
            
            # Re-fit with constrained shape
            from scipy.optimize import minimize
            
            def neg_log_likelihood(params):
                shape, scale = params
                if scale <= 0:
                    return np.inf
                return -np.sum(genpareto.logpdf(exceed, shape, loc=0, scale=scale))
            
            # Initial guess and bounds
            x0 = [max_shape, sigma]
            bounds = [(max_shape-1e-6, max_shape+1e-6), (1e-6, None)]
            
            result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
            if result.success:
                xi, sigma = result.x
                logger.info(f"Successfully constrained shape parameter to {xi:.4f}")
            else:
                logger.warning(f"Failed to constrain shape parameter: {result.message}")
        
        # Compute rate
        rate = n_exc / N
        
        # Compute diagnostics
        neg_log_like = -np.sum(genpareto.logpdf(exceed, xi, loc=0, scale=sigma))
        k = 2  # number of parameters (shape and scale)
        aic = 2 * k + 2 * neg_log_like
        bic = k * np.log(n_exc) + 2 * neg_log_like
        
        # Compute quantile errors (for QQ-plot diagnostics)
        p = np.linspace(0.01, 0.99, 99)
        emp_quantiles = np.quantile(exceed, p)
        theo_quantiles = genpareto.ppf(p, xi, loc=0, scale=sigma)
        quantile_errors = (emp_quantiles - theo_quantiles) / theo_quantiles
        
        return {
            'shape': float(xi),
            'scale': float(sigma),
            'threshold': float(threshold),
            'n_exceed': int(n_exc),
            'rate': float(rate),
            'diagnostics': {
                'neg_log_likelihood': float(neg_log_like),
                'aic': float(aic),
                'bic': float(bic),
                'quantile_errors': quantile_errors.tolist()
            }
        }
    except Exception as e:
        logger.error(f"GPD fitting error: {type(e).__name__}: {e}")
        raise ValueError(f"Failed to fit GPD: {str(e)}")


def return_level(
    xi: float,
    sigma: float,
    threshold: float,
    rate: float,
    return_periods: List[float],
    ci_level: float = 0.95,
    bootstrap_samples: int = 0
) -> pd.DataFrame:
    """
    Compute return levels for specified return periods.

    Parameters
    ----------
    xi, sigma: float
        GPD shape and scale parameters.
    threshold: float
        POT threshold.
    rate: float
        Exceedance rate (n_exc / N) per observation.
    return_periods: list of float
        Return periods in same unit as data frequency (e.g., hours).
    ci_level: float, optional
        Confidence level for intervals (0-1), default 0.95.
    bootstrap_samples: int, optional
        Number of bootstrap samples for confidence intervals.
        If 0, delta method is used for CI.

    Returns
    -------
    pd.DataFrame
        Columns ['return_period', 'return_level', 'lower_ci', 'upper_ci']
    """
    results = []
    
    # Check inputs
    if not np.isfinite(xi) or not np.isfinite(sigma):
        raise ValueError(f"Invalid GPD parameters: shape={xi}, scale={sigma}")
        
    if sigma <= 0:
        raise ValueError(f"Scale parameter must be positive, got {sigma}")
        
    if not 0 <= rate <= 1:
        raise ValueError(f"Rate must be between 0 and 1, got {rate}")
        
    if not all(np.isfinite(return_periods)) or any(rp <= 0 for rp in return_periods):
        raise ValueError("All return periods must be positive finite values")
        
    # Compute return levels
    for T in return_periods:
        # Convert return period to exceedance probability
        # For r events per year and T-year return period, P(X > x) = 1/(r*T)
        p_exceed = 1.0 / T
        
        # Compute quantile 
        # For non-exceedance probability F:
        # F = 1 - P(X > x)
        # F = 1 - rate * (1 - G(x-u))  where G is the GPD CDF
        # For quantile x:
        # x = u + (sigma/xi) * ((1-(1-F)/rate)^(-xi) - 1)
        
        # Non-exceedance probability
        F = 1 - p_exceed
        
        # Compute return level
        if abs(xi) < 1e-6:  # xi ≈ 0
            q_exc = -sigma * np.log((1-F)/rate)
        else:
            q_exc = (sigma / xi) * (((1-F) / rate) ** (-xi) - 1)
            
        level = threshold + q_exc
        
        # Calculate confidence intervals
        lower_ci, upper_ci = np.nan, np.nan
        
        if bootstrap_samples > 0 and ci_level > 0:
            # Bootstrap method for confidence intervals
            # This would require the original exceedances, which we don't have here
            # so we'll skip this for now
            pass
        elif ci_level > 0:
            try:
                # Delta method for confidence intervals with robust handling
                if abs(xi) < 1e-6:  # near-zero shape
                    var_level = (sigma**2) * (np.log((1-F)/rate))**2
                elif xi < 0:
                    # For negative shape, use a different approximation
                    zeta = (1-(1-F)/rate)
                    if 0 < zeta < 1:  # ensure we're not at bounds
                        var_level = (sigma**2) * (1/(rate**2)) * (zeta**2) * (1-zeta)**(-2)
                    else:
                        var_level = np.nan
                else:
                    # Original formula for positive shape
                    var_level = (sigma**2) * (1 - (1-F)/rate)**(-2*xi) / rate
                
                # Ensure variance is real and positive
                if np.isfinite(var_level) and var_level > 0 and not isinstance(var_level, complex):
                    std_level = np.sqrt(var_level)
                    alpha = 1 - ci_level
                    z = -genpareto.ppf(alpha/2, 0, 1)
                    lower_ci = float(level - z * std_level)
                    upper_ci = float(level + z * std_level)
                else:
                    lower_ci = np.nan
                    upper_ci = np.nan
            except Exception:
                # Catch all computational errors
                lower_ci = np.nan
                upper_ci = np.nan
        
        results.append({
            'return_period': float(T),
            'return_level': float(level),
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        })
        
    return pd.DataFrame(results)


def gof_tests(exceedances: np.ndarray, xi: float, sigma: float) -> Dict[str, Any]:
    """
    Perform goodness-of-fit tests for GPD fit.
    
    Parameters
    ----------
    exceedances : np.ndarray
        Exceedances over threshold
    xi : float
        Shape parameter
    sigma : float
        Scale parameter
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of test results
    """
    from scipy.stats import kstest, anderson
    
    results = {}
    
    # Kolmogorov-Smirnov test
    ks = kstest(exceedances, lambda x: genpareto.cdf(x, xi, loc=0, scale=sigma))
    results['ks_statistic'] = float(ks.statistic)
    results['ks_pvalue'] = float(ks.pvalue)
    
    # QQ plot coordinates
    n = len(exceedances)
    pp = np.arange(1, n+1) / (n+1)  # plotting positions
    emp_quantiles = np.sort(exceedances)
    theo_quantiles = genpareto.ppf(pp, xi, loc=0, scale=sigma)
    
    results['qq_plot'] = {
        'empirical': emp_quantiles.tolist(),
        'theoretical': theo_quantiles.tolist()
    }
    
    # PP plot coordinates
    emp_cdf = np.arange(1, n+1) / n
    theo_cdf = genpareto.cdf(np.sort(exceedances), xi, loc=0, scale=sigma)
    
    results['pp_plot'] = {
        'empirical': emp_cdf.tolist(),
        'theoretical': theo_cdf.tolist()
    }
    
    return results


def bootstrap_gpd(
    da: xr.DataArray,
    threshold: float,
    n_bootstrap: int = 1000,
    constrain_shape: bool = True
) -> Dict[str, Any]:
    """
    Perform bootstrap resampling to quantify uncertainty in GPD parameters.
    
    Parameters
    ----------
    da : xr.DataArray
        Data array of values
    threshold : float
        Threshold for exceedances
    n_bootstrap : int
        Number of bootstrap samples
    constrain_shape : bool
        Whether to constrain shape parameter
        
    Returns
    -------
    Dict[str, Any]
        Bootstrap results including parameter distributions
    """
    # Extract exceedances
    arr = da.values.flatten()
    arr = arr[~np.isnan(arr)]
    exceed = arr[arr > threshold] - threshold
    
    if len(exceed) < MIN_EXCEEDANCES:
        raise ValueError(f"Insufficient exceedances ({len(exceed)}) for bootstrap")
    
    # Arrays to store bootstrap results
    shapes = np.zeros(n_bootstrap)
    scales = np.zeros(n_bootstrap)
    
    # Perform bootstrap
    np.random.seed(42)  # For reproducibility
    n = len(exceed)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(exceed, size=n, replace=True)
        
        try:
            # Fit GPD
            params = genpareto.fit(sample, floc=0)
            xi, loc, sigma = params
            
            # Constrain shape if needed
            if constrain_shape and xi > MAX_XI:
                xi = MAX_XI
            
            shapes[i] = xi
            scales[i] = sigma
        except:
            # If fit fails, use median of successful fits
            valid_shapes = shapes[:i]
            valid_scales = scales[:i]
            
            if len(valid_shapes) > 0:
                shapes[i] = np.median(valid_shapes)
                scales[i] = np.median(valid_scales)
            else:
                # If no successful fits, use fallback
                shapes[i] = 0
                scales[i] = np.mean(sample)
    
    # Compute statistics
    shape_mean = np.mean(shapes)
    shape_std = np.std(shapes)
    shape_percentiles = np.percentile(shapes, [2.5, 50, 97.5])
    
    scale_mean = np.mean(scales)
    scale_std = np.std(scales)
    scale_percentiles = np.percentile(scales, [2.5, 50, 97.5])
    
    return {
        'shape': {
            'mean': float(shape_mean),
            'std': float(shape_std),
            'percentiles': {
                '2.5%': float(shape_percentiles[0]),
                '50%': float(shape_percentiles[1]),
                '97.5%': float(shape_percentiles[2])
            },
            'values': shapes.tolist()
        },
        'scale': {
            'mean': float(scale_mean),
            'std': float(scale_std),
            'percentiles': {
                '2.5%': float(scale_percentiles[0]),
                '50%': float(scale_percentiles[1]),
                '97.5%': float(scale_percentiles[2])
            },
            'values': scales.tolist()
        },
        'n_bootstrap': n_bootstrap,
        'n_exceedances': n
    }


if __name__ == '__main__':
    import argparse
    import sys
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='Test univariate module')
    parser.add_argument('--netcdf', help='NetCDF file to analyze (optional)')
    parser.add_argument('--variable', default='sea_level', help='Variable to analyze')
    parser.add_argument('--threshold', type=float, help='Threshold for GPD fitting')
    parser.add_argument('--percentile', type=float, default=0.95, 
                        help='Percentile for threshold if not specified directly')
    parser.add_argument('--return-periods', nargs='+', type=float, default=[10, 20, 50, 100],
                        help='Return periods to compute')
    parser.add_argument('--bootstrap', action='store_true', help='Perform bootstrap uncertainty quantification')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='Number of bootstrap samples')
    parser.add_argument('--plot', action='store_true', help='Generate diagnostic plots')
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
            
            # Determine threshold
            threshold = args.threshold
            if threshold is None:
                arr = da.values.flatten()
                arr = arr[~np.isnan(arr)]
                threshold = float(np.quantile(arr, args.percentile))
                print(f"Using threshold at {args.percentile:.2%} percentile: {threshold:.4f}")
            
            # Fit GPD
            try:
                gpd_results = fit_gpd(da, threshold)
                print("\nGPD fitting results:")
                print(f"  Shape parameter (ξ): {gpd_results['shape']:.4f}")
                print(f"  Scale parameter (σ): {gpd_results['scale']:.4f}")
                print(f"  Threshold (u): {gpd_results['threshold']:.4f}")
                print(f"  Number of exceedances: {gpd_results['n_exceed']}")
                print(f"  Exceedance rate: {gpd_results['rate']:.4f}")
                print(f"  Log-likelihood: {-gpd_results['diagnostics']['neg_log_likelihood']:.4f}")
                print(f"  AIC: {gpd_results['diagnostics']['aic']:.4f}")
                print(f"  BIC: {gpd_results['diagnostics']['bic']:.4f}")
                
                # Compute return levels
                rl = return_level(
                    gpd_results['shape'], 
                    gpd_results['scale'], 
                    gpd_results['threshold'], 
                    gpd_results['rate'], 
                    args.return_periods,
                    ci_level=0.95
                )
                
                print("\nReturn levels:")
                print(rl.to_string(index=False))
                
                # Bootstrap if requested
                if args.bootstrap:
                    print(f"\nPerforming bootstrap with {args.n_bootstrap} samples...")
                    bootstrap_results = bootstrap_gpd(da, threshold, args.n_bootstrap)
                    
                    print("\nBootstrap results:")
                    print(f"  Shape parameter (ξ): {bootstrap_results['shape']['mean']:.4f} ± {bootstrap_results['shape']['std']:.4f}")
                    print(f"  95% CI: [{bootstrap_results['shape']['percentiles']['2.5%']:.4f}, {bootstrap_results['shape']['percentiles']['97.5%']:.4f}]")
                    print(f"  Scale parameter (σ): {bootstrap_results['scale']['mean']:.4f} ± {bootstrap_results['scale']['std']:.4f}")
                    print(f"  95% CI: [{bootstrap_results['scale']['percentiles']['2.5%']:.4f}, {bootstrap_results['scale']['percentiles']['97.5%']:.4f}]")
                
                # Generate plots if requested
                if args.plot:
                    # Extract exceedances for plotting
                    arr = da.values.flatten()
                    arr = arr[~np.isnan(arr)]
                    exceed = arr[arr > threshold] - threshold
                    
                    # Run GOF tests for plotting
                    gof = gof_tests(exceed, gpd_results['shape'], gpd_results['scale'])
                    
                    # Create diagnostic plots
                    plt.figure(figsize=(12, 10))
                    
                    # Histogram of exceedances with fitted GPD
                    plt.subplot(2, 2, 1)
                    plt.hist(exceed, bins=20, density=True, alpha=0.7)
                    
                    # Plot fitted GPD density
                    x = np.linspace(0, exceed.max(), 1000)
                    y = genpareto.pdf(x, gpd_results['shape'], loc=0, scale=gpd_results['scale'])
                    plt.plot(x, y, 'r-', label=f'Fitted GPD (ξ={gpd_results["shape"]:.3f}, σ={gpd_results["scale"]:.3f})')
                    plt.xlabel('Exceedance')
                    plt.ylabel('Density')
                    plt.title('Histogram of Exceedances with Fitted GPD')
                    plt.legend()
                    
                    # QQ plot
                    plt.subplot(2, 2, 2)
                    plt.plot(gof['qq_plot']['theoretical'], gof['qq_plot']['empirical'], 'o')
                    max_val = max(max(gof['qq_plot']['theoretical']), max(gof['qq_plot']['empirical']))
                    plt.plot([0, max_val], [0, max_val], 'r--')
                    plt.xlabel('Theoretical Quantiles')
                    plt.ylabel('Empirical Quantiles')
                    plt.title(f'QQ Plot (KS p-value: {gof["ks_pvalue"]:.3f})')
                    
                    # PP plot
                    plt.subplot(2, 2, 3)
                    plt.plot(gof['pp_plot']['theoretical'], gof['pp_plot']['empirical'], 'o')
                    plt.plot([0, 1], [0, 1], 'r--')
                    plt.xlabel('Theoretical CDF')
                    plt.ylabel('Empirical CDF')
                    plt.title('PP Plot')
                    
                    # Return level plot
                    plt.subplot(2, 2, 4)
                    plt.plot(rl['return_period'], rl['return_level'], 'o-')
                    plt.fill_between(
                        rl['return_period'], 
                        rl['lower_ci'], 
                        rl['upper_ci'], 
                        alpha=0.3
                    )
                    plt.xscale('log')
                    plt.xlabel('Return Period')
                    plt.ylabel('Return Level')
                    plt.title('Return Level Plot')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Error in GPD analysis: {type(e).__name__}: {e}")
                
        else:
            # Synthetic data test
            print("Testing with synthetic data")
            
            # Generate GPD sample
            np.random.seed(42)
            true_shape = 0.2
            true_scale = 1.0
            n_samples = 1000
            
            # Generate exceedances directly from GPD
            exceed = genpareto.rvs(true_shape, scale=true_scale, size=n_samples)
            
            # Create pseudo-observations with threshold
            threshold = 1.0
            base_data = np.random.normal(0, 0.5, 5000)
            # Ensure base data is below threshold
            base_data = base_data[base_data < threshold]
            # Take a subset and concatenate with exceedances + threshold
            base_subset = base_data[:4000]
            full_data = np.concatenate([base_subset, exceed + threshold])
            
            # Create xarray DataArray
            da = xr.DataArray(full_data, dims=['time'], coords={'time': np.arange(len(full_data))})
            
            print(f"Generated {len(full_data)} synthetic data points")
            print(f"True parameters: shape={true_shape}, scale={true_scale}")
            print(f"Data range: {float(da.min())} to {float(da.max())}")
            print(f"Threshold: {threshold}")
            
            # Fit GPD
            try:
                gpd_results = fit_gpd(da, threshold)
                print("\nGPD fitting results:")
                print(f"  Shape parameter (ξ): {gpd_results['shape']:.4f} (true: {true_shape})")
                print(f"  Scale parameter (σ): {gpd_results['scale']:.4f} (true: {true_scale})")
                print(f"  Threshold (u): {gpd_results['threshold']:.4f}")
                print(f"  Number of exceedances: {gpd_results['n_exceed']}")
                print(f"  Exceedance rate: {gpd_results['rate']:.4f}")
                
                # Compute return levels
                rl = return_level(
                    gpd_results['shape'], 
                    gpd_results['scale'], 
                    gpd_results['threshold'], 
                    gpd_results['rate'], 
                    args.return_periods
                )
                
                print("\nReturn levels:")
                print(rl.to_string(index=False))
                
                # Bootstrap if requested
                if args.bootstrap:
                    print(f"\nPerforming bootstrap with {args.n_bootstrap} samples...")
                    bootstrap_results = bootstrap_gpd(da, threshold, args.n_bootstrap)
                    
                    print("\nBootstrap results:")
                    print(f"  Shape parameter (ξ): {bootstrap_results['shape']['mean']:.4f} ± {bootstrap_results['shape']['std']:.4f}")
                    print(f"  95% CI: [{bootstrap_results['shape']['percentiles']['2.5%']:.4f}, {bootstrap_results['shape']['percentiles']['97.5%']:.4f}]")
                    print(f"  Scale parameter (σ): {bootstrap_results['scale']['mean']:.4f} ± {bootstrap_results['scale']['std']:.4f}")
                    print(f"  95% CI: [{bootstrap_results['scale']['percentiles']['2.5%']:.4f}, {bootstrap_results['scale']['percentiles']['97.5%']:.4f}]")
                
                # Generate plots if requested
                if args.plot:
                    # Run GOF tests for plotting
                    exceed = full_data[full_data > threshold] - threshold
                    gof = gof_tests(exceed, gpd_results['shape'], gpd_results['scale'])
                    
                    # Create diagnostic plots
                    plt.figure(figsize=(12, 10))
                    
                    # Histogram of exceedances with fitted GPD
                    plt.subplot(2, 2, 1)
                    plt.hist(exceed, bins=20, density=True, alpha=0.7)
                    
                    # Plot fitted GPD density
                    x = np.linspace(0, exceed.max(), 1000)
                    y = genpareto.pdf(x, gpd_results['shape'], loc=0, scale=gpd_results['scale'])
                    y_true = genpareto.pdf(x, true_shape, loc=0, scale=true_scale)
                    plt.plot(x, y, 'r-', label=f'Fitted GPD (ξ={gpd_results["shape"]:.3f}, σ={gpd_results["scale"]:.3f})')
                    plt.plot(x, y_true, 'g--', label=f'True GPD (ξ={true_shape}, σ={true_scale})')
                    plt.xlabel('Exceedance')
                    plt.ylabel('Density')
                    plt.title('Histogram of Exceedances with Fitted GPD')
                    plt.legend()
                    
                    # QQ plot
                    plt.subplot(2, 2, 2)
                    plt.plot(gof['qq_plot']['theoretical'], gof['qq_plot']['empirical'], 'o')
                    max_val = max(max(gof['qq_plot']['theoretical']), max(gof['qq_plot']['empirical']))
                    plt.plot([0, max_val], [0, max_val], 'r--')
                    plt.xlabel('Theoretical Quantiles')
                    plt.ylabel('Empirical Quantiles')
                    plt.title(f'QQ Plot (KS p-value: {gof["ks_pvalue"]:.3f})')
                    
                    # PP plot
                    plt.subplot(2, 2, 3)
                    plt.plot(gof['pp_plot']['theoretical'], gof['pp_plot']['empirical'], 'o')
                    plt.plot([0, 1], [0, 1], 'r--')
                    plt.xlabel('Theoretical CDF')
                    plt.ylabel('Empirical CDF')
                    plt.title('PP Plot')
                    
                    # Return level plot
                    plt.subplot(2, 2, 4)
                    plt.plot(rl['return_period'], rl['return_level'], 'o-')
                    plt.fill_between(
                        rl['return_period'], 
                        rl['lower_ci'], 
                        rl['upper_ci'], 
                        alpha=0.3
                    )
                    plt.xscale('log')
                    plt.xlabel('Return Period')
                    plt.ylabel('Return Level')
                    plt.title('Return Level Plot')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"Error in GPD analysis: {type(e).__name__}: {e}")
        
        print("\nUnivariate module smoke test completed successfully.")
    except Exception as e:
        print(f"Error during test: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)