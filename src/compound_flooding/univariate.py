# src/compound_flooding/univariate.py
"""
Module: univariate.py
Responsibilities:
- Fit GPD to data exceeding a specified threshold
- Compute return levels for given return periods
- Provide diagnostic statistics
"""
import numpy as np
import xarray as xr
from scipy.stats import genpareto
import pandas as pd


def fit_gpd(
    da: xr.DataArray,
    threshold: float
) -> dict:
    """
    Fit a Generalized Pareto Distribution to exceedances over threshold.

    Parameters
    ----------
    da : xr.DataArray
        Data array of values (e.g., sea_level). NaNs ignored.
    threshold : float
        POT threshold for defining exceedances.

    Returns
    -------
    dict
        {
          'shape': xi,
          'scale': sigma,
          'threshold': threshold,
          'n_exceed': n_exc,
          'rate': n_exc / N_total
        }
    """
    # Extract exceedances
    arr = da.values.flatten()
    arr = arr[~np.isnan(arr)]
    N = arr.size
    exceed = arr[arr > threshold] - threshold
    n_exc = exceed.size
    if n_exc < 1:
        raise ValueError("No exceedances above threshold.")
    # Fit GPD
    params = genpareto.fit(exceed, loc=0)
    # scipy returns (c, loc, scale)
    xi, loc, sigma = params
    rate = n_exc / N
    return {
        'shape': float(xi),
        'scale': float(sigma),
        'threshold': float(threshold),
        'n_exceed': int(n_exc),
        'rate': float(rate)
    }


def return_level(
    xi: float,
    sigma: float,
    threshold: float,
    rate: float,
    return_periods: list[float]
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

    Returns
    -------
    pd.DataFrame
        Columns ['return_period', 'return_level']
    """
    results = []
    for T in return_periods:
        # Probability of exceedance in one obs = 1 - 1/T
        prob = 1 - 1.0 / T
        # Quantile of GPD: q = threshold + (sigma/xi)*((prob/rate)**(-xi) - 1)
        if xi != 0:
            q_exc = (sigma / xi) * ((prob / rate) ** (-xi) - 1)
        else:
            q_exc = -sigma * np.log(prob / rate)
        level = threshold + q_exc
        results.append({'return_period': T, 'return_level': float(level)})
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Smoke-test GPD fitting
    print("Running univariate module smoke-test...")
    import xarray as xr
    # Create synthetic data: exponential tail
    np.random.seed(0)
    data = np.concatenate([np.random.rand(1000), 5 + np.random.exponential(scale=2, size=100)])
    da = xr.DataArray(data, dims=['i'])
    thr = float(np.quantile(data, 0.8))
    res = fit_gpd(da, thr)
    print(f"  Fit params: shape={res['shape']:.2f}, scale={res['scale']:.2f}, rate={res['rate']:.3f}")
    rl = return_level(
        res['shape'], res['scale'], res['threshold'], res['rate'], [10, 20, 50]
    )
    print("  Return levels (10,20,50):")
    print(rl)
    print("univariate module smoke-test completed.")
