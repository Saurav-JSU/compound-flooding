# src/compound_flooding/tier1_stats.py
"""
Module: tier1_stats.py
Responsibilities:
- Orchestrate Tier-1 analysis per station
  * Threshold selection
  * Univariate GPD fits
  * Return level estimation
  * Empirical joint exceedance stats
  * Uncertainty quantification
- Parallel processing over cleaned NetCDFs
- Save per-station Parquet or JSON outputs
"""
import os
import glob
import json
import logging
import numpy as np
import xarray as xr
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime

from src.compound_flooding.thresholds import select_threshold, find_optimal_threshold
from src.compound_flooding.univariate import fit_gpd, return_level, bootstrap_gpd
from src.compound_flooding.joint_empirical import compute_empirical_stats, bootstrap_joint_stats, compute_time_lag_dependency

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_json_serializable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, complex):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return [ensure_json_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): ensure_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return ensure_json_serializable(obj.__dict__)
    else:
        return str(obj)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is np.ma.masked:
            return None
        return super(NumpyEncoder, self).default(obj)

def load_threshold_from_json(threshold_dir: str, station_code: str, variable: str) -> Optional[float]:
    """
    Load threshold value from a previously computed threshold JSON file.
    
    Parameters
    ----------
    threshold_dir : str
        Directory containing threshold JSON files
    station_code : str
        Station code
    variable : str
        Variable name
        
    Returns
    -------
    Optional[float]
        Threshold value if found, None otherwise
    """
    try:
        filename = os.path.join(threshold_dir, f"{station_code}_thresholds.json")
        if not os.path.exists(filename):
            logger.warning(f"Threshold file not found: {filename}")
            return None
            
        with open(filename, 'r') as f:
            data = json.load(f)
            
        if variable in data['variables']:
            var_data = data['variables'][variable]
            
            # First try optimal threshold from MRL
            if 'optimal_threshold' in var_data and var_data['optimal_threshold'] is not None:
                logger.info(f"Using optimal threshold for {variable}: {var_data['optimal_threshold']}")
                return float(var_data['optimal_threshold'])
                
            # Fall back to simple percentile threshold
            if 'simple_threshold' in var_data and var_data['simple_threshold'] is not None:
                logger.info(f"Using simple threshold for {variable}: {var_data['simple_threshold']}")
                return float(var_data['simple_threshold'])
                
        logger.warning(f"No threshold found for {variable} in {filename}")
        return None
    except Exception as e:
        logger.warning(f"Error loading threshold from {threshold_dir}/{station_code}_thresholds.json: {e}")
        return None


def process_station(
    nc_file: str,
    output_dir: str,
    threshold_dir: Optional[str],
    pct_sl: float,
    pct_pr: float,
    return_periods: List[float],
    lag_hours: int,
    use_bootstrap: bool = True,
    n_bootstrap: int = 500,
    output_format: str = 'json'
) -> str:
    """
    Process one station NetCDF: compute thresholds, fits, joint stats, and save to output format.

    Parameters
    ----------
    nc_file : str
        Path to NetCDF file
    output_dir : str
        Directory to save outputs
    threshold_dir : str, optional
        Directory containing threshold results
    pct_sl : float
        Sea-level threshold percentile (if not found in threshold_dir)
    pct_pr : float
        Precipitation threshold percentile (if not found in threshold_dir)
    return_periods : List[float]
        Return periods to compute
    lag_hours : int
        Lag hours for joint exceedance window
    use_bootstrap : bool, default=True
        Whether to perform bootstrap for uncertainty quantification
    n_bootstrap : int, default=500
        Number of bootstrap samples
    output_format : str, default='json'
        Output format ('json' or 'parquet')
        
    Returns
    -------
    str
        Status message
    """
    # Extract station code from filename
    station_code = os.path.splitext(os.path.basename(nc_file))[0]
    
    try:
        logger.info(f"Processing station {station_code}")
        
        # Check if output already exists
        output_file = os.path.join(output_dir, f"{station_code}_tier1.{output_format}")
        if os.path.exists(output_file):
            logger.info(f"Output file already exists: {output_file}, skipping")
            return f"SKIP {station_code}: output already exists"
        
        # Open dataset
        ds = xr.open_dataset(nc_file)
        
        # Check for required variables
        if 'sea_level' not in ds or 'total_precipitation' not in ds:
            missing = []
            if 'sea_level' not in ds:
                missing.append('sea_level')
            if 'total_precipitation' not in ds:
                missing.append('total_precipitation')
                
            logger.warning(f"Missing required variables in {nc_file}: {', '.join(missing)}")
            ds.close()
            return f"SKIP {station_code}: missing variables {', '.join(missing)}"
        
        sea = ds['sea_level']
        pr = ds['total_precipitation']
        
        # Initialize results dictionary
        results = {
            'station': station_code,
            'file': nc_file,
            'sea_level': {},
            'total_precipitation': {},
            'joint': {}
        }
        
        # Select sea-level threshold
        try:
            # First try to load from threshold_dir if provided
            if threshold_dir:
                thr_sl = load_threshold_from_json(threshold_dir, station_code, 'sea_level')
            else:
                thr_sl = None
                
            # If not found or no threshold_dir, compute using percentile
            if thr_sl is None:
                thr_sl = select_threshold(sea, pct_sl)
                results['sea_level']['threshold_source'] = f"computed_percentile_{pct_sl}"
            else:
                results['sea_level']['threshold_source'] = "from_json"
                
            results['sea_level']['threshold'] = float(thr_sl)
                
        except Exception as e:
            logger.error(f"Sea level threshold error for {station_code}: {e}")
            ds.close()
            return f"ERROR {station_code}: sea_level threshold error ({type(e).__name__}: {e})"
        
        # Select precipitation threshold
        try:
            # First try to load from threshold_dir if provided
            if threshold_dir:
                thr_pr = load_threshold_from_json(threshold_dir, station_code, 'total_precipitation')
            else:
                thr_pr = None
                
            # If not found or no threshold_dir, compute using percentile
            if thr_pr is None:
                thr_pr = select_threshold(pr, pct_pr)
                results['total_precipitation']['threshold_source'] = f"computed_percentile_{pct_pr}"
            else:
                results['total_precipitation']['threshold_source'] = "from_json"
                
            results['total_precipitation']['threshold'] = float(thr_pr)
                
        except Exception as e:
            logger.error(f"Precipitation threshold error for {station_code}: {e}")
            ds.close()
            return f"ERROR {station_code}: precipitation threshold error ({type(e).__name__}: {e})"
        
        # Univariate GPD fits
        try:
            logger.info(f"Fitting GPD for sea_level with threshold {thr_sl:.4f}")
            res_sl = fit_gpd(sea, thr_sl)
            results['sea_level']['gpd'] = {
                'shape': float(res_sl['shape']),
                'scale': float(res_sl['scale']),
                'n_exceed': int(res_sl['n_exceed']),
                'rate': float(res_sl['rate']),
                'diagnostics': res_sl['diagnostics']
            }
            
            # Bootstrap uncertainty for sea level if requested
            if use_bootstrap and res_sl['n_exceed'] >= 30:
                logger.info(f"Bootstrapping sea_level GPD parameters")
                boot_sl = bootstrap_gpd(sea, thr_sl, n_bootstrap=n_bootstrap)
                results['sea_level']['bootstrap'] = boot_sl
        except Exception as e:
            logger.error(f"Sea level GPD fitting error for {station_code}: {e}")
            results['sea_level']['gpd'] = {
                'error': f"{type(e).__name__}: {str(e)}"
            }
        
        try:
            logger.info(f"Fitting GPD for total_precipitation with threshold {thr_pr:.4f}")
            res_pr = fit_gpd(pr, thr_pr)
            results['total_precipitation']['gpd'] = {
                'shape': float(res_pr['shape']),
                'scale': float(res_pr['scale']),
                'n_exceed': int(res_pr['n_exceed']),
                'rate': float(res_pr['rate']),
                'diagnostics': res_pr['diagnostics']
            }
            
            # Bootstrap uncertainty for precipitation if requested
            if use_bootstrap and res_pr['n_exceed'] >= 30:
                logger.info(f"Bootstrapping total_precipitation GPD parameters")
                boot_pr = bootstrap_gpd(pr, thr_pr, n_bootstrap=n_bootstrap)
                results['total_precipitation']['bootstrap'] = boot_pr
        except Exception as e:
            logger.error(f"Precipitation GPD fitting error for {station_code}: {e}")
            results['total_precipitation']['gpd'] = {
                'error': f"{type(e).__name__}: {str(e)}"
            }
        
        # Return levels
        try:
            if 'gpd' in results['sea_level'] and 'error' not in results['sea_level']['gpd']:
                logger.info(f"Computing return levels for sea_level")
                rl_sl = return_level(
                    results['sea_level']['gpd']['shape'], 
                    results['sea_level']['gpd']['scale'], 
                    results['sea_level']['threshold'], 
                    results['sea_level']['gpd']['rate'], 
                    return_periods,
                    ci_level=0.95
                )
                results['sea_level']['return_levels'] = rl_sl.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Sea level return level error for {station_code}: {e}")
            results['sea_level']['return_levels'] = {
                'error': f"{type(e).__name__}: {str(e)}"
            }
            
        try:
            if 'gpd' in results['total_precipitation'] and 'error' not in results['total_precipitation']['gpd']:
                logger.info(f"Computing return levels for total_precipitation")
                rl_pr = return_level(
                    results['total_precipitation']['gpd']['shape'], 
                    results['total_precipitation']['gpd']['scale'], 
                    results['total_precipitation']['threshold'], 
                    results['total_precipitation']['gpd']['rate'], 
                    return_periods,
                    ci_level=0.95
                )
                results['total_precipitation']['return_levels'] = rl_pr.to_dict(orient='records')
        except Exception as e:
            logger.error(f"Precipitation return level error for {station_code}: {e}")
            results['total_precipitation']['return_levels'] = {
                'error': f"{type(e).__name__}: {str(e)}"
            }
        
        # Empirical joint stats
        try:
            logger.info(f"Computing joint statistics with lag window ±{lag_hours}h")
            joint_stats = compute_empirical_stats(
                sea, pr, thr_sl, thr_pr, lag_hours=lag_hours
            )
            results['joint']['empirical'] = joint_stats
            
            # Compute lag dependency analysis
            logger.info(f"Computing time lag dependency")
            lag_max = max(24, lag_hours * 2)  # Look at least ±24h or 2x the specified lag
            lag_results = compute_time_lag_dependency(
                sea, pr, thr_sl, thr_pr, max_lag_hours=lag_max
            )
            results['joint']['lag_analysis'] = lag_results
            
            # Bootstrap joint statistics if requested
            if use_bootstrap and joint_stats['n_joint'] >= 10:
                logger.info(f"Bootstrapping joint statistics")
                boot_joint = bootstrap_joint_stats(
                    sea, pr, thr_sl, thr_pr, 
                    lag_hours=lag_hours, 
                    n_bootstrap=n_bootstrap
                )
                results['joint']['bootstrap'] = boot_joint
                
        except Exception as e:
            logger.error(f"Joint statistics error for {station_code}: {e}")
            results['joint']['error'] = f"{type(e).__name__}: {str(e)}"
        
        # Close dataset
        ds.close()
        
        # Save results to output file
        os.makedirs(output_dir, exist_ok=True)
        
        if output_format.lower() == 'json':
            try:
                # Make sure the entire results structure is serializable
                serializable_results = ensure_json_serializable(results)
                with open(output_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
                logger.info(f"Saved results to {output_file}")
            except Exception as e:
                logger.error(f"Error saving results to {output_file}: {e}")
                # Try a last resort approach
                with open(output_file, 'w') as f:
                    json.dump(str(results), f)  # Convert entire result to string if all else fails
                logger.warning(f"Saved stringified results to {output_file} due to serialization error")
        
        elif output_format.lower() == 'parquet':
            try:
                # For Parquet, we need a flat DataFrame
                flat_results = {
                    'station': station_code,
                    'file': nc_file,
                    'sl_threshold': results['sea_level'].get('threshold'),
                    'pr_threshold': results['total_precipitation'].get('threshold'),
                }
                
                # Add GPD parameters
                if 'gpd' in results['sea_level']:
                    gpd_sl = results['sea_level']['gpd']
                    if 'error' not in gpd_sl:
                        flat_results.update({
                            'sl_shape': gpd_sl['shape'],
                            'sl_scale': gpd_sl['scale'],
                            'sl_n_exceed': gpd_sl['n_exceed'],
                            'sl_rate': gpd_sl['rate']
                        })
                
                if 'gpd' in results['total_precipitation']:
                    gpd_pr = results['total_precipitation']['gpd']
                    if 'error' not in gpd_pr:
                        flat_results.update({
                            'pr_shape': gpd_pr['shape'],
                            'pr_scale': gpd_pr['scale'],
                            'pr_n_exceed': gpd_pr['n_exceed'],
                            'pr_rate': gpd_pr['rate']
                        })
                
                # Add joint statistics
                if 'empirical' in results['joint']:
                    joint = results['joint']['empirical']
                    flat_results.update({
                        'joint_n_exc1': joint['n_exc1'],
                        'joint_n_exc2': joint['n_exc2'],
                        'joint_n_joint': joint['n_joint'],
                        'joint_p_joint': joint['p_joint'],
                        'joint_p_independent': joint['p_independent'],
                        'joint_cpr': joint['cpr'],
                        'joint_p2_given_1': joint['p2_given_1'],
                        'joint_p1_given_2': joint['p1_given_2']
                    })
                
                # Add optimal lag if available
                if 'lag_analysis' in results['joint']:
                    lag = results['joint']['lag_analysis']
                    flat_results['optimal_lag'] = lag['optimal_lag']
                
                # Save as DataFrame
                df = pd.DataFrame([flat_results])
                df.to_parquet(output_file)
                logger.info(f"Saved results to {output_file}")
                
                # Also save the full results as JSON for reference
                json_file = os.path.join(output_dir, f"{station_code}_tier1_full.json")
                serializable_results = ensure_json_serializable(results)
                with open(json_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
                logger.info(f"Saved full results to {json_file}")
                
            except Exception as e:
                logger.error(f"Error saving parquet: {e}")
                # Fall back to JSON
                json_file = os.path.join(output_dir, f"{station_code}_tier1.json")
                try:
                    serializable_results = ensure_json_serializable(results)
                    with open(json_file, 'w') as f:
                        json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
                    logger.info(f"Saved results to {json_file} (fallback from parquet)")
                except Exception as e2:
                    logger.error(f"Error saving JSON fallback: {e2}")
                    return f"ERROR {station_code}: multiple save errors"
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return f"ERROR {station_code}: unsupported output format {output_format}"
        
        return f"OK {station_code}"
        
    except Exception as e:
        logger.error(f"Error processing {station_code}: {e}")
        return f"ERROR {station_code}: {type(e).__name__}: {e}"


def run_tier1(
    netcdf_dir: str,
    output_dir: str,
    threshold_dir: Optional[str] = None,
    pct_sl: float = 0.99,
    pct_pr: float = 0.99,
    return_periods: List[float] = [10, 20, 50, 100],
    lag_hours: int = 0,
    use_bootstrap: bool = True,
    n_bootstrap: int = 500,
    output_format: str = 'json',
    workers: Optional[int] = None,
    max_files: Optional[int] = None
) -> None:
    """
    Run Tier-1 analysis for all station NetCDFs in directory, in parallel.
    
    Parameters
    ----------
    netcdf_dir : str
        Directory containing NetCDF files
    output_dir : str
        Directory to save Tier-1 analysis results
    threshold_dir : str, optional
        Directory containing threshold results
    pct_sl : float, default=0.99
        Sea-level threshold percentile
    pct_pr : float, default=0.99
        Precipitation threshold percentile
    return_periods : List[float], default=[10, 20, 50, 100]
        Return periods to compute
    lag_hours : int, default=0
        Lag hours for joint exceedance window
    use_bootstrap : bool, default=True
        Whether to perform bootstrap for uncertainty quantification
    n_bootstrap : int, default=500
        Number of bootstrap samples
    output_format : str, default='json'
        Output format ('json' or 'parquet')
    workers : int, optional
        Number of parallel workers (None=all cores)
    max_files : int, optional
        Maximum number of files to process (for testing)
    
    Raises
    ------
    FileNotFoundError
        If no NetCDF files found in netcdf_dir
    ValueError
        If output_format is not 'json' or 'parquet'
    """
    # Validate output format
    if output_format.lower() not in ['json', 'parquet']:
        raise ValueError(f"Invalid output format: {output_format}. Use 'json' or 'parquet'.")
    
    # Find NetCDF files
    files = sorted(glob.glob(os.path.join(netcdf_dir, '*.nc')))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {netcdf_dir}")
        
    # Limit files if requested
    if max_files and max_files > 0 and max_files < len(files):
        logger.info(f"Limiting to {max_files} files (out of {len(files)})")
        files = files[:max_files]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    n_workers = workers or os.cpu_count() or 1
    logger.info(f"Running Tier-1 on {len(files)} stations with {n_workers} workers...")
    
    # Prepare function with fixed parameters
    func = partial(
        process_station,
        output_dir=output_dir,
        threshold_dir=threshold_dir,
        pct_sl=pct_sl,
        pct_pr=pct_pr,
        return_periods=return_periods,
        lag_hours=lag_hours,
        use_bootstrap=use_bootstrap,
        n_bootstrap=n_bootstrap,
        output_format=output_format
    )
    
    # Initialize counters
    success_count = 0
    error_count = 0
    skip_count = 0
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Use a reasonable chunk size for better load balancing
        chunk_size = max(1, min(100, len(files) // (n_workers * 4)))
        logger.info(f"Using chunk size {chunk_size}")
        
        for i, result in enumerate(executor.map(func, files, chunksize=chunk_size)):
            print(f"[{i+1}/{len(files)}] {result}")
            if result.startswith("OK"):
                success_count += 1
            elif result.startswith("ERROR"):
                error_count += 1
            elif result.startswith("SKIP"):
                skip_count += 1
    
    logger.info(f"Tier-1 processing complete: {success_count} succeeded, {error_count} failed, {skip_count} skipped")


if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Tier-1 stats CLI')
    parser.add_argument('--netcdf-dir', required=True, help='Directory of cleaned NetCDFs')
    parser.add_argument('--output-dir', required=True, help='Directory to save Tier-1 output files')
    parser.add_argument('--threshold-dir', help='Directory with threshold results (optional)')
    parser.add_argument('--pct-sl', type=float, default=0.99, help='Sea-level threshold percentile')
    parser.add_argument('--pct-pr', type=float, default=0.99, help='Precip threshold percentile')
    parser.add_argument('--return-periods', nargs='+', type=float, default=[10,20,50,100], help='Return periods')
    parser.add_argument('--lag-hours', type=int, default=0, help='Lag hours for joint exceedance window')
    parser.add_argument('--no-bootstrap', action='store_true', help='Disable bootstrap uncertainty quantification')
    parser.add_argument('--n-bootstrap', type=int, default=500, help='Number of bootstrap samples')
    parser.add_argument('--output-format', choices=['json', 'parquet'], default='json', 
                       help='Output format (json or parquet)')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (0=all cores)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    try:
        run_tier1(
            netcdf_dir=args.netcdf_dir,
            output_dir=args.output_dir,
            threshold_dir=args.threshold_dir,
            pct_sl=args.pct_sl,
            pct_pr=args.pct_pr,
            return_periods=args.return_periods,
            lag_hours=args.lag_hours,
            use_bootstrap=not args.no_bootstrap,
            n_bootstrap=args.n_bootstrap,
            output_format=args.output_format,
            workers=(args.workers if args.workers > 0 else None),
            max_files=args.max_files
        )
        print("Tier-1 analysis complete!")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)