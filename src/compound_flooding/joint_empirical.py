# src/compound_flooding/joint_empirical.py
"""
Module: joint_empirical.py
Responsibilities:
- Compute empirical joint exceedance statistics for two variables
- Support optional lead/lag window for co-occurrence
- Compute marginal exceedance rates, joint rate, conditional probabilities, and CPR
- Save results to Parquet
"""
import numpy as np
import xarray as xr
import pandas as pd


def compute_empirical_stats(
    da1: xr.DataArray,
    da2: xr.DataArray,
    threshold1: float,
    threshold2: float,
    lag_hours: int = 0
) -> dict:
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

    Returns
    -------
    stats : dict
        Keys: n_total, n_exc1, n_exc2, n_joint, p_exc1, p_exc2,
              p_joint, p_independent, cpr, p2_given_1, p1_given_2
    """
    # Convert to pandas Series and align on index
    sx = da1.to_series()
    sy = da2.to_series()
    df = pd.DataFrame({'x': sx, 'y': sy}).dropna()
    n_total = len(df)
    if n_total == 0:
        raise ValueError("No overlapping non-NaN observations.")

    # If lag window, compute rolling max of y
    if lag_hours > 0:
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

    p_exc1 = n_exc1 / n_total
    p_exc2 = n_exc2 / n_total
    p_joint = n_joint / n_total
    p_ind = p_exc1 * p_exc2
    cpr = p_joint / p_ind if p_ind > 0 else np.nan
    p2_given_1 = (n_joint / n_exc1) if n_exc1 > 0 else np.nan
    p1_given_2 = (n_joint / n_exc2) if n_exc2 > 0 else np.nan

    return {
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


def save_stats(stats: dict, filename: str) -> None:
    """
    Save statistics dict to Parquet file.
    """
    df = pd.DataFrame([stats])
    df.to_parquet(filename)


if __name__ == '__main__':
    print("Running joint_empirical module smoke-test...")
    import xarray as xr
    import numpy as np

    # Create synthetic correlated data
    np.random.seed(0)
    n = 1000
    x = np.random.randn(n)
    # y correlated with x
    y = 0.5 * x + 0.5 * np.random.randn(n)
    # Choose thresholds at high quantile
    da1 = xr.DataArray(x, dims=['time'], coords={'time': pd.date_range('2000-01-01', periods=n, freq='H')})
    da2 = xr.DataArray(y, dims=['time'], coords=da1.coords)
    thr1 = float(np.quantile(x, 0.9))
    thr2 = float(np.quantile(y, 0.9))

    stats = compute_empirical_stats(da1, da2, thr1, thr2, lag_hours=0)
    print(f"Statistics: {stats}")
    # Basic sanity checks
    assert stats['n_total'] == n
    assert stats['n_exc1'] > 0 and stats['n_exc2'] > 0
    assert 0 <= stats['p_joint'] <= 1
    print("joint_empirical module smoke-test completed successfully.")
