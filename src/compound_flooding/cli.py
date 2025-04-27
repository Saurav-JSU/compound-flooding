# src/compound_flooding/cli.py

"""
CLI wrapper for the Compound Flooding toolkit.

Sub-commands:
  ingest    : Ingest and preprocess all stations (parallel)
  thresholds: Analyze and find optimal thresholds for variables (parallel)
  tier1     : Run Tier-1 statistical analysis (parallel)
  copula    : Fit copulas to one or many cleaned NetCDFs (parallel)
"""

import argparse
import os
import glob
import multiprocessing
import concurrent.futures
import json
import logging
from functools import partial
import pandas as pd
import xarray as xr
import numpy as np
from scipy.stats import kendalltau

from src.compound_flooding.data_io import validate_paths, load_metadata, load_station_data
from src.compound_flooding.preprocess import preprocess_dataframe
from src.compound_flooding.thresholds import select_threshold, find_optimal_threshold
from src.compound_flooding.tier1_stats import run_tier1
from src.compound_flooding.copula_utils import (
    fit_gumbel, fit_frank, fit_student, fit_gaussian, select_best_copula
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        logger.info(f"Processing station {code}")
        df = load_station_data(code, station_dir)
        if hasattr(df, 'compute'):
            logger.info(f"Computing dask dataframe for {code}")
            df = df.compute()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"{code}.nc")
        
        # Skip if already exists and not forced to reprocess
        if os.path.exists(outpath):
            logger.info(f"Output file {outpath} already exists, skipping")
            return f"[SKIP] {code} - already processed"
            
        ds = preprocess_dataframe(
            df,
            detrend=detrend,
            max_gap_hours=max_gap,
            spike_threshold=spike
        )
        ds.to_netcdf(outpath)
        return f"[OK]   {code} → {outpath}"
    except Exception as e:
        logger.error(f"Error processing station {code}: {type(e).__name__}: {e}")
        return f"[ERR]  {code}: {type(e).__name__}: {e}"


def ingest_all(
    metadata: str,
    station_dir: str,
    output_dir: str,
    detrend: bool,
    max_gap: int,
    spike: float,
    workers: int,
    max_stations: int = None
) -> None:
    """
    Parallel ingest & preprocess for all stations.
    
    Parameters
    ----------
    metadata : str
        Path to metadata CSV
    station_dir : str
        Directory containing station CSVs
    output_dir : str
        Directory to save NetCDF files
    detrend : bool
        Whether to detrend sea_level
    max_gap : int
        Maximum gap size in hours to interpolate
    spike : float
        Threshold for clipping sea_level spikes
    workers : int
        Number of parallel workers (0=all cores)
    max_stations : int, optional
        Maximum number of stations to process (for testing)
    """
    validate_paths(metadata, station_dir)
    meta = load_metadata(metadata)
    codes = meta['SITE CODE'].tolist()
    
    # Limit number of stations if specified
    if max_stations and max_stations > 0 and max_stations < len(codes):
        logger.info(f"Limiting to {max_stations} stations (out of {len(codes)})")
        codes = codes[:max_stations]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Decide on number of workers
    num_workers = workers if workers > 0 else multiprocessing.cpu_count()
    logger.info(f"Running ingest for {len(codes)} stations with {num_workers} worker(s)...")

    func = partial(
        _process_station,
        station_dir=station_dir,
        output_dir=output_dir,
        detrend=detrend,
        max_gap=max_gap,
        spike=spike
    )
    
    # Initialize counters
    success_count = 0
    error_count = 0
    skip_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, msg in enumerate(executor.map(func, codes, chunksize=max(1, len(codes) // (num_workers * 4)))):
            print(f"[{i+1}/{len(codes)}] {msg}")
            if msg.startswith("[OK]"):
                success_count += 1
            elif msg.startswith("[ERR]"):
                error_count += 1
            elif msg.startswith("[SKIP]"):
                skip_count += 1
    
    logger.info(f"Ingest complete: {success_count} succeeded, {error_count} failed, {skip_count} skipped")


def _find_thresholds_for_station(
    nc_path: str,
    output_dir: str,
    variables: list,
    percentile: float,
    use_mrl: bool,
    min_percentile: float,
    max_percentile: float,
    n_points: int
) -> str:
    """
    Find thresholds for a single station NetCDF file.
    
    Parameters
    ----------
    nc_path : str
        Path to NetCDF file
    output_dir : str
        Directory to save threshold results
    variables : list
        List of variables to analyze
    percentile : float
        Percentile for simple threshold selection
    use_mrl : bool
        Whether to use mean residual life analysis
    min_percentile : float
        Minimum percentile for MRL analysis
    max_percentile : float
        Maximum percentile for MRL analysis
    n_points : int
        Number of points for MRL analysis
        
    Returns
    -------
    str
        Status message
    """
    try:
        station_code = os.path.splitext(os.path.basename(nc_path))[0]
        logger.info(f"Finding thresholds for station {station_code}")
        
        # Open dataset
        ds = xr.open_dataset(nc_path)
        
        results = {
            'station_code': station_code,
            'file': nc_path,
            'variables': {}
        }
        
        # Process each variable
        for var in variables:
            if var not in ds:
                logger.warning(f"Variable {var} not found in {nc_path}")
                continue
                
            # Extract data array
            da = ds[var]
            
            # Skip if all values are NaN
            if np.isnan(da.values).all():
                logger.warning(f"Variable {var} contains only NaN values in {station_code}")
                continue
                
            var_results = {}
            
            # Find simple threshold at percentile
            try:
                simple_thr = select_threshold(da, percentile)
                var_results['simple_threshold'] = float(simple_thr)
                var_results['percentile'] = float(percentile)
            except Exception as e:
                logger.error(f"Error finding simple threshold for {var} in {station_code}: {e}")
                var_results['simple_threshold'] = None
            
            # Find optimal threshold using mean residual life if requested
            if use_mrl:
                try:
                    optimal_thr, mrl_data = find_optimal_threshold(
                        da,
                        min_percentile=min_percentile,
                        max_percentile=max_percentile,
                        n_points=n_points
                    )
                    var_results['optimal_threshold'] = float(optimal_thr)
                    var_results['mrl_analysis'] = {
                        'min_percentile': float(min_percentile),
                        'max_percentile': float(max_percentile),
                        'n_points': int(n_points)
                    }
                except Exception as e:
                    logger.error(f"Error finding optimal threshold for {var} in {station_code}: {e}")
                    var_results['optimal_threshold'] = None
            
            # Add variable results
            results['variables'][var] = var_results
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{station_code}_thresholds.json")
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        ds.close()
        return f"[OK]   {station_code}"
    except Exception as e:
        logger.error(f"Error processing {nc_path}: {type(e).__name__}: {e}")
        return f"[ERR]  {os.path.basename(nc_path)}: {type(e).__name__}: {e}"


def find_thresholds(
    netcdf_dir: str,
    output_dir: str,
    variables: list,
    percentile: float,
    use_mrl: bool,
    min_percentile: float,
    max_percentile: float,
    n_points: int,
    workers: int
) -> None:
    """
    Find thresholds for all stations in parallel.
    
    Parameters
    ----------
    netcdf_dir : str
        Directory containing NetCDF files
    output_dir : str
        Directory to save threshold results
    variables : list
        List of variables to analyze
    percentile : float
        Percentile for simple threshold selection
    use_mrl : bool
        Whether to use mean residual life analysis
    min_percentile : float
        Minimum percentile for MRL analysis
    max_percentile : float
        Maximum percentile for MRL analysis
    n_points : int
        Number of points for MRL analysis
    workers : int
        Number of parallel workers (0=all cores)
    """
    # Find all NetCDF files
    files = sorted(glob.glob(os.path.join(netcdf_dir, '*.nc')))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {netcdf_dir}")
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Decide on number of workers
    num_workers = workers if workers > 0 else multiprocessing.cpu_count()
    logger.info(f"Finding thresholds for {len(files)} stations with {num_workers} worker(s)...")
    
    # Prepare function with fixed parameters
    func = partial(
        _find_thresholds_for_station,
        output_dir=output_dir,
        variables=variables,
        percentile=percentile,
        use_mrl=use_mrl,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        n_points=n_points
    )
    
    # Initialize counters
    success_count = 0
    error_count = 0
    
    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, msg in enumerate(executor.map(func, files, chunksize=max(1, len(files) // (num_workers * 4)))):
            print(f"[{i+1}/{len(files)}] {msg}")
            if msg.startswith("[OK]"):
                success_count += 1
            elif msg.startswith("[ERR]"):
                error_count += 1
    
    logger.info(f"Threshold analysis complete: {success_count} succeeded, {error_count} failed")


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
    logger.info(f"Opening dataset {input_nc}...")
    ds = xr.open_dataset(input_nc)

    var1 = ds[u_var].values
    var2 = ds[v_var].values

    # drop NaNs and Infs
    valid = np.isfinite(var1) & np.isfinite(var2)
    n_drop = len(var1) - valid.sum()
    if n_drop:
        logger.warning(f"Dropping {n_drop} missing or invalid values before copula fitting.")
    u, v = var1[valid], var2[valid]
    
    # Check if we have enough valid data
    if len(u) < 30:
        raise ValueError(f"Insufficient valid data points ({len(u)}) for copula fitting. Minimum required: 30")

    # pseudo‐obs conversion
    df = pd.DataFrame({'u': u, 'v': v})
    u_p = df['u'].rank(method='average') / (len(df) + 1)
    v_p = df['v'].rank(method='average') / (len(df) + 1)
    logger.info("Converting to pseudo-observations...")

    # fit copula
    logger.info(f"Fitting copula (method={method})...")
    if method == 'auto':
        cop = select_best_copula(u_p.values, v_p.values)
    else:
        func_map = {
            'gumbel': fit_gumbel,
            'frank': fit_frank,
            'student': fit_student,
            'gaussian': fit_gaussian
        }
        if method not in func_map:
            raise ValueError(f"Unknown copula method: {method}. Valid methods: {', '.join(func_map.keys())}")
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

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to {output_json}")


def _run_one_copula(
    nc_path: str,
    u_var: str,
    v_var: str,
    method: str,
    output_json: str,
    out_is_dir: bool
) -> str:
    """
    Top-level helper for ProcessPoolExecutor. Fits a copula for one file.
    """
    try:
        station_code = os.path.splitext(os.path.basename(nc_path))[0]
        
        if out_is_dir:
            base = os.path.splitext(os.path.basename(nc_path))[0]
            outpath = os.path.join(output_json, f"{base}_copula.json")
        else:
            outpath = output_json
            
        # Skip if output already exists
        if os.path.exists(outpath):
            logger.info(f"Output file {outpath} already exists, skipping")
            return f"[SKIP] {station_code} - already processed"

        fit_copula(
            input_nc=nc_path,
            u_var=u_var,
            v_var=v_var,
            method=method,
            output_json=outpath
        )
        return f"[OK]   {station_code} → {outpath}"
    except Exception as e:
        logger.error(f"Error fitting copula for {nc_path}: {type(e).__name__}: {e}")
        return f"[ERR]  {os.path.basename(nc_path)}: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(
        prog='compound_flooding',
        description='Compound Flooding Toolkit'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ingest sub-command
    p_ingest = sub.add_parser('ingest', help='Ingest & preprocess all stations')
    p_ingest.add_argument('--metadata', default='compound_flooding/data/GESLA/usa_metadata.csv',
                         help='Path to GESLA metadata CSV')
    p_ingest.add_argument('--station-dir', default='compound_flooding/GESLA_ERA5_with_sea_level',
                         help='Directory containing station CSVs')
    p_ingest.add_argument('--output-dir', default='outputs/cleaned',
                         help='Directory to save cleaned NetCDF files')
    p_ingest.add_argument('--detrend', action='store_true',
                         help='Remove linear trend from sea_level')
    p_ingest.add_argument('--max-gap', type=int, default=2,
                         help='Maximum gap hours for interpolation')
    p_ingest.add_argument('--spike', type=float, default=None,
                         help='Threshold for clipping sea_level spikes')
    p_ingest.add_argument('--workers', type=int, default=0,
                         help='Number of parallel workers (0=all cores)')
    p_ingest.add_argument('--max-stations', type=int, default=None,
                         help='Maximum number of stations to process (for testing)')
    
    # thresholds sub-command
    p_thresh = sub.add_parser('thresholds', help='Find optimal thresholds for variables')
    p_thresh.add_argument('--netcdf-dir', default='outputs/cleaned',
                         help='Directory containing cleaned NetCDF files')
    p_thresh.add_argument('--output-dir', default='outputs/thresholds',
                         help='Directory to save threshold results')
    p_thresh.add_argument('--variables', nargs='+', default=['sea_level', 'total_precipitation'],
                         help='Variables to analyze')
    p_thresh.add_argument('--percentile', type=float, default=0.99,
                         help='Percentile for simple threshold selection')
    p_thresh.add_argument('--use-mrl', action='store_true',
                         help='Use mean residual life analysis for optimal thresholds')
    p_thresh.add_argument('--min-percentile', type=float, default=0.8,
                         help='Minimum percentile for MRL analysis')
    p_thresh.add_argument('--max-percentile', type=float, default=0.995,
                         help='Maximum percentile for MRL analysis')
    p_thresh.add_argument('--n-points', type=int, default=20,
                         help='Number of points for MRL analysis')
    p_thresh.add_argument('--workers', type=int, default=0,
                         help='Number of parallel workers (0=all cores)')

    # tier1 sub-command
    p_tier1 = sub.add_parser('tier1', help='Run Tier-1 analysis for all cleaned NetCDFs')
    p_tier1.add_argument('--netcdf-dir', default='outputs/cleaned',
                        help='Directory containing cleaned NetCDF files')
    p_tier1.add_argument('--threshold-dir', default='outputs/thresholds',
                        help='Directory containing threshold results (optional)')
    p_tier1.add_argument('--output-dir', default='outputs/tier1',
                        help='Directory to save Tier-1 analysis results')
    p_tier1.add_argument('--pct-sl', type=float, default=0.99,
                        help='Sea-level threshold percentile (used if not found in threshold-dir)')
    p_tier1.add_argument('--pct-pr', type=float, default=0.99,
                        help='Precipitation threshold percentile (used if not found in threshold-dir)')
    p_tier1.add_argument('--return-periods', nargs='+', type=float, default=[10,20,50,100],
                        help='Return periods to compute')
    p_tier1.add_argument('--lag-hours', type=int, default=0,
                        help='Lag hours for joint exceedance window')
    p_tier1.add_argument('--no-bootstrap', action='store_true',
                        help='Disable bootstrap uncertainty quantification')
    p_tier1.add_argument('--n-bootstrap', type=int, default=500,
                        help='Number of bootstrap samples')
    p_tier1.add_argument('--output-format', choices=['json', 'parquet'], default='json',
                        help='Output format (json or parquet)')
    p_tier1.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    p_tier1.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0=all cores)')

    # copula sub-command
    p_copula = sub.add_parser('copula', help='Fit copulas to one or many cleaned NetCDFs')
    group = p_copula.add_mutually_exclusive_group(required=True)
    group.add_argument('--input-nc', help='Single cleaned NetCDF file')
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
            workers=args.workers,
            max_stations=args.max_stations
        )
    
    elif args.command == 'thresholds':
        find_thresholds(
            netcdf_dir=args.netcdf_dir,
            output_dir=args.output_dir,
            variables=args.variables,
            percentile=args.percentile,
            use_mrl=args.use_mrl,
            min_percentile=args.min_percentile,
            max_percentile=args.max_percentile,
            n_points=args.n_points,
            workers=args.workers
        )

    elif args.command == 'tier1':
        run_tier1(
            netcdf_dir=args.netcdf_dir,
            output_dir=args.output_dir,
            threshold_dir=args.threshold_dir if hasattr(args, 'threshold_dir') else None,
            pct_sl=args.pct_sl,
            pct_pr=args.pct_pr,
            return_periods=args.return_periods,
            lag_hours=args.lag_hours,
            use_bootstrap=not args.no_bootstrap,
            n_bootstrap=args.n_bootstrap,
            workers=(args.workers if args.workers > 0 else None),
            max_files=args.max_files
        )

    elif args.command == 'copula':
        # gather input files
        if args.input_dir:
            files = sorted(glob.glob(os.path.join(args.input_dir, '*.nc')))
            if not files:
                raise FileNotFoundError(f"No NetCDF files found in {args.input_dir}")
            os.makedirs(args.output_json, exist_ok=True)
            out_is_dir = True
        else:
            files = [args.input_nc]
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            out_is_dir = False

        num_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
        logger.info(f"Running copula on {len(files)} file(s) with {num_workers} worker(s)...")

        # build a picklable function with all parameters baked in
        func = partial(
            _run_one_copula,
            u_var=args.u_var,
            v_var=args.v_var,
            method=args.method,
            output_json=args.output_json,
            out_is_dir=out_is_dir
        )

        # Initialize counters
        success_count = 0
        error_count = 0
        skip_count = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i, msg in enumerate(executor.map(func, files, chunksize=max(1, len(files) // (num_workers * 4)))):
                print(f"[{i+1}/{len(files)}] {msg}")
                if msg.startswith("[OK]"):
                    success_count += 1
                elif msg.startswith("[ERR]"):
                    error_count += 1
                elif msg.startswith("[SKIP]"):
                    skip_count += 1
        
        logger.info(f"Copula fitting complete: {success_count} succeeded, {error_count} failed, {skip_count} skipped")


if __name__ == '__main__':
    main()