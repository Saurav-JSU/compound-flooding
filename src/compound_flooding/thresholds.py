# src/compound_flooding/thresholds.py
"""
Module: thresholds.py
Responsibilities:
- Compute candidate POT thresholds (percentile-based)
- Compute mean residual life (mean excess) diagnostic data
- Provide helper to select threshold at a given percentile
"""
import numpy as np
import xarray as xr
import pandas as pd


def select_threshold(
    da: xr.DataArray,
    percentile: float = 0.99
) -> float:
    """
    Select threshold at the given percentile from the DataArray, ignoring NaNs.

    Parameters
    ----------
    da : xr.DataArray
        Input data array of values (e.g., sea level or precipitation).
    percentile : float (0 < p < 1)
        Percentile for threshold selection (e.g., 0.99 for 99th percentile).

    Returns
    -------
    float
        Threshold value at specified percentile.
    """
    arr = da.values.flatten()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("DataArray contains only NaNs; cannot compute threshold.")
    return float(np.quantile(arr, percentile))


def mean_residual_life(
    da: xr.DataArray,
    thresholds: np.ndarray
) -> pd.DataFrame:
    """
    Compute mean residual life (mean excess over threshold) for each threshold.

    Parameters
    ----------
    da : xr.DataArray
        Input data array of values.
    thresholds : np.ndarray
        Array of threshold candidates.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['threshold', 'mean_excess', 'n_exceed']
    """
    arr = da.values.flatten()
    df = []
    for u in thresholds:
        exceed = arr[arr > u]
        if exceed.size > 0:
            mean_exc = np.mean(exceed - u)
            count = exceed.size
        else:
            mean_exc = np.nan
            count = 0
        df.append({'threshold': float(u), 'mean_excess': mean_exc, 'n_exceed': int(count)})
    return pd.DataFrame(df)



if __name__ == '__main__':
    # Manual smoke-test for threshold functions
    import xarray as xr
    import numpy as np

    print("Running thresholds module smoke-test...")
    # Test select_threshold
    da = xr.DataArray([0, 1, 2, 3, 4], dims=['time'])
    thr = select_threshold(da, percentile=0.6)
    expected = float(np.quantile([0,1,2,3,4], 0.6))
    assert thr == expected, f"select_threshold failed: got {thr}, expected {expected}"
    print(f"  select_threshold OK: {thr}")

    # Test mean_residual_life
    da2 = xr.DataArray([0, 2, 4, 6, 8], dims=['t'])
    thresholds = np.array([2, 4])
    mrl = mean_residual_life(da2, thresholds)
    # For threshold 2: exceed [4,6,8] mean_excess = (2+4+6)/3 = 4.0
    me2 = mrl.loc[mrl['threshold']==2, 'mean_excess'].iloc[0]
    assert np.isclose(me2, 4.0), f"MRL at 2 incorrect: {me2}"
    print(f"  mean_residual_life at 2 OK: {me2}")
    # For threshold 4: exceed [6,8] mean_excess = (2+4)/2 = 3.0? Actually values [6,8]-4 = [2,4] mean=3.0
    me4 = mrl.loc[mrl['threshold']==4, 'mean_excess'].iloc[0]
    assert np.isclose(me4, 3.0), f"MRL at 4 incorrect: {me4}"
    print(f"  mean_residual_life at 4 OK: {me4}")

    print("thresholds module smoke-test completed successfully.")
