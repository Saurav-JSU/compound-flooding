# src/compound_flooding/tier1_stats.py
"""
Module: tier1_stats.py
Responsibilities:
- Orchestrate Tier-1 analysis per station
  * Threshold selection
  * Univariate GPD fits
  * Return level estimation
  * Empirical joint exceedance stats
- Parallel processing over cleaned NetCDFs
- Save per-station Parquet outputs
"""
import os
import glob
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from src.compound_flooding.thresholds import select_threshold
from src.compound_flooding.univariate import fit_gpd, return_level
from src.compound_flooding.joint_empirical import compute_empirical_stats


def process_station(
    nc_file: str,
    output_dir: str,
    pct_sl: float,
    pct_pr: float,
    return_periods: list[float],
    lag_hours: int
) -> str:
    """
    Process one station NetCDF: compute thresholds, fits, joint stats, and save to Parquet.

    Returns station code on success or skip message.
    """
    code = os.path.splitext(os.path.basename(nc_file))[0]
    try:
        ds = xr.open_dataset(nc_file)
        sea = ds['sea_level']
        pr = ds['total_precipitation']

        # Select sea-level threshold
        try:
            thr_sl = select_threshold(sea, pct_sl)
        except Exception as e:
            ds.close()
            return f"SKIP {code}: sea_level threshold error ({e})"

        # Select precipitation threshold
        try:
            thr_pr = select_threshold(pr, pct_pr)
        except Exception as e:
            ds.close()
            return f"SKIP {code}: precipitation threshold error ({e})"

        # Univariate GPD fits
        res_sl = fit_gpd(sea, thr_sl)
        res_pr = fit_gpd(pr, thr_pr)

        # Return levels
        rl_sl = return_level(
            res_sl['shape'], res_sl['scale'], res_sl['threshold'], res_sl['rate'], return_periods
        )
        rl_pr = return_level(
            res_pr['shape'], res_pr['scale'], res_pr['threshold'], res_pr['rate'], return_periods
        )

        # Empirical joint stats
        stats = compute_empirical_stats(sea, pr, thr_sl, thr_pr, lag_hours=lag_hours)

        # Combine results
        combined = {
            'station': code,
            'thr_sl': thr_sl,
            'thr_pr': thr_pr,
            'xi_sl': res_sl['shape'],
            'sigma_sl': res_sl['scale'],
            'rate_sl': res_sl['rate'],
            'xi_pr': res_pr['shape'],
            'sigma_pr': res_pr['scale'],
            'rate_pr': res_pr['rate'],
            **stats
        }
        # Add return levels
        for rp in return_periods:
            lv_sl = rl_sl.loc[rl_sl['return_period']==rp, 'return_level'].iloc[0]
            lv_pr = rl_pr.loc[rl_pr['return_period']==rp, 'return_level'].iloc[0]
            combined[f'rl_sl_{int(rp)}'] = float(lv_sl)
            combined[f'rl_pr_{int(rp)}'] = float(lv_pr)

        # Save to Parquet
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{code}.parquet")
        pd.DataFrame([combined]).to_parquet(out_file)
        ds.close()
        return code

    except Exception as e:
        try:
            ds.close()
        except:
            pass
        return f"ERROR {code}: {e}"


def run_tier1(
    netcdf_dir: str,
    output_dir: str,
    pct_sl: float = 0.99,
    pct_pr: float = 0.99,
    return_periods: list[float] = [10, 20, 50, 100],
    lag_hours: int = 0,
    workers: int = None
) -> None:
    """
    Run Tier-1 analysis for all station NetCDFs in directory, in parallel.
    """
    files = glob.glob(os.path.join(netcdf_dir, '*.nc'))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {netcdf_dir}")

    n_workers = workers or os.cpu_count() or 1
    print(f"Running Tier-1 on {len(files)} stations with {n_workers} workers...")

    func = partial(
        process_station,
        output_dir=output_dir,
        pct_sl=pct_sl,
        pct_pr=pct_pr,
        return_periods=return_periods,
        lag_hours=lag_hours
    )
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(func, files):
            print(res)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tier-1 stats CLI')
    parser.add_argument('--netcdf-dir', required=True, help='Directory of cleaned NetCDFs')
    parser.add_argument('--output-dir', required=True, help='Directory to save Tier-1 Parquet files')
    parser.add_argument('--pct-sl', type=float, default=0.99, help='Sea-level threshold percentile')
    parser.add_argument('--pct-pr', type=float, default=0.99, help='Precip threshold percentile')
    parser.add_argument('--return-periods', nargs='+', type=float, default=[10,20,50,100], help='Return periods')
    parser.add_argument('--lag-hours', type=int, default=0, help='Lag hours for joint exceedance window')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (0=all cores)')

    args = parser.parse_args()
    run_tier1(
        netcdf_dir=args.netcdf_dir,
        output_dir=args.output_dir,
        pct_sl=args.pct_sl,
        pct_pr=args.pct_pr,
        return_periods=args.return_periods,
        lag_hours=args.lag_hours,
        workers=(args.workers or None)
    )