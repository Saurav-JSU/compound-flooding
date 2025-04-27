# src/compound_flooding/cli.py

"""
CLI wrapper for the Compound Flooding toolkit.

Sub-commands:
  ingest  : Ingest and preprocess all stations (parallel)
  tier1   : Run Tier-1 statistical analysis (parallel)
  copula  : Fit copulas to one or many cleaned NetCDFs (parallel)
"""

import argparse
import os
import glob
import multiprocessing
import concurrent.futures
import json
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import kendalltau

from src.compound_flooding.data_io import validate_paths, load_metadata, load_station_data
from src.compound_flooding.preprocess import preprocess_dataframe
from src.compound_flooding.tier1_stats import run_tier1
from src.compound_flooding.copula_utils import (
    fit_gumbel, fit_frank, fit_student, fit_gaussian, select_best_copula
)


def _process_station(
    code: str,
    station_dir: str,
    output_dir: str,
    detrend: bool,
    max_gap: int,
    spike: float
) -> str:
    """
    Ingest, preprocess, and save one station to NetCDF.
    """
    try:
        df = load_station_data(code, station_dir)
        if hasattr(df, 'compute'):
            df = df.compute()
        ds = preprocess_dataframe(
            df,
            detrend=detrend,
            max_gap_hours=max_gap,
            spike_threshold=spike
        )
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"{code}.nc")
        ds.to_netcdf(outpath)
        return f"[OK]   {code} → {outpath}"
    except Exception as e:
        return f"[ERR]  {code}: {e}"


def ingest_all(
    metadata: str,
    station_dir: str,
    output_dir: str,
    detrend: bool,
    max_gap: int,
    spike: float,
    workers: int
) -> None:
    """
    Parallel ingest & preprocess for all stations.
    """
    validate_paths(metadata, station_dir)
    meta = load_metadata(metadata)
    codes = meta['SITE CODE'].tolist()

    num_workers = workers or multiprocessing.cpu_count()
    print(f"Running ingest for {len(codes)} stations with {num_workers} worker(s)...")

    func = partial(
        _process_station,
        station_dir=station_dir,
        output_dir=output_dir,
        detrend=detrend,
        max_gap=max_gap,
        spike=spike
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for msg in executor.map(func, codes):
            print(msg)


def fit_copula(
    input_nc: str,
    u_var: str,
    v_var: str,
    method: str,
    output_json: str
) -> None:
    """
    Open a single NetCDF, extract u_var and v_var, fit a copula, and write JSON.
    """
    print(f"Opening dataset {input_nc}...")
    ds = xr.open_dataset(input_nc)

    var1 = ds[u_var].values
    var2 = ds[v_var].values

    # drop NaNs and Infs
    valid = np.isfinite(var1) & np.isfinite(var2)
    n_drop = len(var1) - valid.sum()
    if n_drop:
        print(f"Warning: dropping {n_drop} missing or invalid values before copula fitting.")
    u, v = var1[valid], var2[valid]

    # pseudo‐obs conversion
    df = pd.DataFrame({'u': u, 'v': v})
    u_p = df['u'].rank(method='average') / (len(df) + 1)
    v_p = df['v'].rank(method='average') / (len(df) + 1)
    print("Converting to pseudo-observations...")

    # fit copula
    print(f"Fitting copula (method={method})...")
    if method == 'auto':
        cop = select_best_copula(u_p.values, v_p.values)
    else:
        func_map = {
            'gumbel': fit_gumbel,
            'frank': fit_frank,
            'student': fit_student,
            'gaussian': fit_gaussian
        }
        cop = func_map[method](u_p.values, v_p.values)

    # compute metrics
    uv = np.column_stack((u_p, v_p))
    log_lik = float(cop.logpdf(uv).sum())
    n = len(uv)

    # extract copula parameters
    params = {}
    if hasattr(cop, 'theta'):
        params['theta'] = cop.theta
    if hasattr(cop, 'df'):
        params['df'] = cop.df
    if hasattr(cop, 'corr'):
        params['corr'] = cop.corr.tolist()

    # information criterion
    aic = 2 * len(params) - 2 * log_lik

    result = {
        'copula': cop.__class__.__name__,
        'params': params,
        'kendall_tau': float(kendalltau(u, v)[0]),
        'n_obs': n,
        'log_likelihood': log_lik,
        'aic': float(aic)
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {output_json}")


def _run_one_copula(
    nc_path: str,
    u_var: str,
    v_var: str,
    method: str,
    output_json: str,
    out_is_dir: bool
) -> None:
    """
    Top-level helper for ProcessPoolExecutor. Fits a copula for one file.
    """
    if out_is_dir:
        base = os.path.splitext(os.path.basename(nc_path))[0]
        outpath = os.path.join(output_json, f"{base}_copula.json")
    else:
        outpath = output_json

    fit_copula(
        input_nc=nc_path,
        u_var=u_var,
        v_var=v_var,
        method=method,
        output_json=outpath
    )
    print(f"[OK] {nc_path} → {outpath}")


def main():
    parser = argparse.ArgumentParser(
        prog='compound_flooding',
        description='Compound Flooding Toolkit'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ingest sub-command
    p_ingest = sub.add_parser('ingest', help='Ingest & preprocess all stations')
    p_ingest.add_argument('--metadata', default='compound_flooding/data/GESLA/usa_metadata.csv')
    p_ingest.add_argument('--station-dir', default='compound_flooding/GESLA_ERA5_with_sea_level')
    p_ingest.add_argument('--output-dir', default='outputs/cleaned')
    p_ingest.add_argument('--detrend', action='store_true')
    p_ingest.add_argument('--max-gap', type=int, default=2)
    p_ingest.add_argument('--spike', type=float, default=None)
    p_ingest.add_argument('--workers', type=int, default=0,
                         help='Number of parallel workers (0=all cores)')

    # tier1 sub-command
    p_tier1 = sub.add_parser('tier1', help='Run Tier-1 analysis for all cleaned NetCDFs')
    p_tier1.add_argument('--netcdf-dir', required=True)
    p_tier1.add_argument('--output-dir', required=True)
    p_tier1.add_argument('--pct-sl', type=float, default=0.99)
    p_tier1.add_argument('--pct-pr', type=float, default=0.99)
    p_tier1.add_argument('--return-periods', nargs='+', type=float, default=[10,20,50,100])
    p_tier1.add_argument('--lag-hours', type=int, default=0)
    p_tier1.add_argument('--workers', type=int, default=0,
                         help='Number of parallel workers (0=all cores)')

    # copula sub-command
    p_copula = sub.add_parser('copula', help='Fit copulas to one or many cleaned NetCDFs')
    group = p_copula.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-nc',  help='Single cleaned NetCDF file')
    group.add_argument('--input-dir', help='Directory of cleaned NetCDF files')
    p_copula.add_argument('--u-var', default='sea_level',
                          help='Name of the first variable (u)')
    p_copula.add_argument('--v-var', default='total_precipitation',
                          help='Name of the second variable (v)')
    p_copula.add_argument('--method', choices=['gumbel','frank','student','gaussian','auto'],
                          default='auto', help='Which copula to fit')
    p_copula.add_argument('--output-json', required=True,
                          help='Output JSON file (or directory if --input-dir)')
    p_copula.add_argument('--workers', type=int, default=0,
                          help='Number of parallel workers (0=all cores)')

    args = parser.parse_args()

    if args.command == 'ingest':
        ingest_all(
            metadata=args.metadata,
            station_dir=args.station_dir,
            output_dir=args.output_dir,
            detrend=args.detrend,
            max_gap=args.max_gap,
            spike=args.spike,
            workers=args.workers
        )

    elif args.command == 'tier1':
        run_tier1(
            netcdf_dir=args.netcdf_dir,
            output_dir=args.output_dir,
            pct_sl=args.pct_sl,
            pct_pr=args.pct_pr,
            return_periods=args.return_periods,
            lag_hours=args.lag_hours,
            workers=(args.workers or None)
        )

    elif args.command == 'copula':
        # gather input files
        if args.input_dir:
            files = sorted(glob.glob(os.path.join(args.input_dir, '*.nc')))
            os.makedirs(args.output_json, exist_ok=True)
            out_is_dir = True
        else:
            files = [args.input_nc]
            out_is_dir = False

        num_workers = args.workers or multiprocessing.cpu_count()
        print(f"Running copula on {len(files)} file(s) with {num_workers} worker(s)...")

        # build a picklable function with all parameters baked in
        func = partial(
            _run_one_copula,
            u_var=args.u_var,
            v_var=args.v_var,
            method=args.method,
            output_json=args.output_json,
            out_is_dir=out_is_dir
        )

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(func, files))


if __name__ == '__main__':
    main()
