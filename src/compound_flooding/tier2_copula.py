# src/compound_flooding/tier2_copula.py

"""
Module: tier2_copula.py
Responsibilities:
- Orchestrate Tier-2 copula analysis for compound flooding
- Load data from Tier-1 outputs or NetCDF files
- Fit copulas for bivariate relationships
- Compute joint exceedance probabilities and return periods
- Compute conditional exceedance probabilities
- Calculate tail dependence coefficients and CPR
- Save results to JSON or Parquet format
- Support parallel processing across stations
"""

import os
import glob
import json
import logging
import numpy as np
import xarray as xr
import pandas as pd
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Dict, List, Optional, Union, Any, Tuple
import warnings

# Import Tier-2 modules
from src.compound_flooding.copula_params import (
    estimate_theta_gumbel,
    estimate_theta_frank,
    compute_tail_dependence
)
from src.compound_flooding.copula_fit import (
    create_pseudo_observations,
    fit_copula_with_metrics,
    compute_joint_exceedance,
    compute_joint_return_period,
    compute_conditional_exceedance,
    compute_cpr
)

from src.compound_flooding.copula_fit import (
    compute_conditional_exceeding_given_exceeding)

from src.compound_flooding.copula_selection import (
    select_best_copula,
    compare_copula_fits
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tier1_output(file_path: str) -> Dict[str, Any]:
    """
    Load Tier-1 output from JSON or Parquet file.
    
    Parameters
    ----------
    file_path : str
        Path to Tier-1 output file (JSON or Parquet)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with Tier-1 results
        
    Raises
    ------
    ValueError
        If file format is not supported or file cannot be read
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        elif file_ext in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
            # Convert DataFrame to dictionary
            return df.to_dict(orient='records')[0]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        raise ValueError(f"Error loading Tier-1 output from {file_path}: {str(e)}")


def ensure_json_serializable(obj):
    """Convert a potentially non-serializable object to JSON-serializable form."""
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
    """Custom encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def process_station(
    netcdf_file: str,
    tier1_file: Optional[str] = None,
    output_dir: str = "./outputs/tier2",
    u_var: str = "sea_level",
    v_var: str = "total_precipitation",
    copula_method: str = "auto",
    compute_return_periods: List[float] = [10, 20, 50, 100],
    exceedance_levels: List[float] = [0.9, 0.95, 0.99],
    parallel: bool = False,
    output_format: str = "json"
) -> str:
    """
    Process a single station for Tier-2 copula analysis.
    
    Parameters
    ----------
    netcdf_file : str
        Path to NetCDF file with station data
    tier1_file : Optional[str]
        Path to Tier-1 output file (if available)
    output_dir : str
        Directory to save Tier-2 outputs
    u_var : str
        First variable for copula (default: sea_level)
    v_var : str
        Second variable for copula (default: total_precipitation)
    copula_method : str
        Copula type or 'auto' for automatic selection
    compute_return_periods : List[float]
        List of return periods to compute
    exceedance_levels : List[float]
        List of exceedance probabilities to analyze
    parallel : bool
        Whether to use parallel processing for model selection
    output_format : str
        Output format ('json' or 'parquet')
        
    Returns
    -------
    str
        Status message
    """
    # Extract station code from filename
    station_code = os.path.splitext(os.path.basename(netcdf_file))[0]
    
    try:
        logger.info(f"Processing station {station_code}")
        
        # Check if output already exists
        output_file = os.path.join(output_dir, f"{station_code}_tier2.{output_format}")
        if os.path.exists(output_file):
            logger.info(f"Output file already exists: {output_file}, skipping")
            return f"SKIP {station_code}: output already exists"
        
        # Initialize results dictionary
        results = {
            'station': station_code,
            'file': netcdf_file,
            'tier2_analysis': {},
            'tier1_reference': tier1_file
        }
        
        # Load NetCDF data
        try:
            ds = xr.open_dataset(netcdf_file)
            
            # Check for required variables
            if u_var not in ds or v_var not in ds:
                missing = []
                if u_var not in ds:
                    missing.append(u_var)
                if v_var not in ds:
                    missing.append(v_var)
                    
                logger.warning(f"Missing required variables in {netcdf_file}: {', '.join(missing)}")
                return f"SKIP {station_code}: missing variables {', '.join(missing)}"
            
            # Get the data arrays
            da_u = ds[u_var]
            da_v = ds[v_var]
            
            # Load Tier-1 thresholds if available
            tier1_data = None
            if tier1_file and os.path.exists(tier1_file):
                try:
                    tier1_data = load_tier1_output(tier1_file)
                    logger.info(f"Loaded Tier-1 data from {tier1_file}")
                    
                    # Extract thresholds
                    if u_var == 'sea_level' and v_var == 'total_precipitation':
                        u_threshold = tier1_data.get('sea_level', {}).get('threshold')
                        v_threshold = tier1_data.get('total_precipitation', {}).get('threshold')
                        
                        if u_threshold is not None and v_threshold is not None:
                            results['tier1_thresholds'] = {
                                u_var: float(u_threshold),
                                v_var: float(v_threshold)
                            }
                            logger.info(f"Using Tier-1 thresholds: {u_var}={u_threshold}, {v_var}={v_threshold}")
                    
                    # Extract CPR from Tier-1 joint analysis
                    if 'joint' in tier1_data and 'empirical' in tier1_data['joint']:
                        emp_cpr = tier1_data['joint']['empirical'].get('cpr')
                        if emp_cpr is not None:
                            results['tier1_empirical_cpr'] = float(emp_cpr)
                            logger.info(f"Tier-1 empirical CPR: {emp_cpr}")
                            
                except Exception as e:
                    logger.warning(f"Error loading Tier-1 data: {e}")
            
            # Convert raw data to arrays
            u_values = da_u.values.flatten()
            v_values = da_v.values.flatten()
            
            # Remove NaNs
            valid = ~np.isnan(u_values) & ~np.isnan(v_values)
            if np.sum(valid) < 100:
                logger.warning(f"Insufficient valid data points: {np.sum(valid)}")
                return f"SKIP {station_code}: insufficient valid data points"
                
            u_valid = u_values[valid]
            v_valid = v_values[valid]
            
            # Convert to pseudo-observations
            u_pobs, v_pobs = create_pseudo_observations(u_valid, v_valid)
            
            logger.info(f"Created pseudo-observations from {len(u_pobs)} data points")
            
            # Fit copula
            logger.info(f"Fitting copula (method={copula_method})")
            fit_results = fit_copula_with_metrics(
                u_pobs, v_pobs, 
                method=copula_method,
                compute_diagnostics=True
            )
            
            if 'error' in fit_results:
                logger.error(f"Error fitting copula: {fit_results['error']}")
                results['tier2_analysis']['copula_fit_error'] = fit_results['error']
                results['tier2_analysis']['copula_method'] = 'independence'
                
                # Save the partial results to output file
                os.makedirs(output_dir, exist_ok=True)
                
                if output_format.lower() == 'json':
                    with open(output_file, 'w') as f:
                        json.dump(ensure_json_serializable(results), f, indent=2, cls=NumpyEncoder)
                elif output_format.lower() == 'parquet':
                    # Convert to DataFrame
                    df = pd.DataFrame([results])
                    df.to_parquet(output_file)
                
                return f"ERROR {station_code}: copula fitting failed"
            
            # Extract fitted copula and method
            copula = fit_results['copula']
            best_method = fit_results['method']
            
            # Store basic copula info
            results['tier2_analysis']['copula'] = {
                'method': best_method,
                'log_likelihood': fit_results.get('log_likelihood'),
                'aic': fit_results.get('aic'),
                'bic': fit_results.get('bic'),
                'parameters': fit_results.get('params', {})
            }
            
            # Compute tail dependence
            tail_dep = fit_results.get('tail_dependence', {})
            if tail_dep:
                results['tier2_analysis']['tail_dependence'] = tail_dep
                logger.info(f"Tail dependence - Lower: {tail_dep.get('lower')}, Upper: {tail_dep.get('upper')}")
            
            # Compute joint exceedance probabilities for different levels
            joint_probs = {}
            for level in exceedance_levels:
                u_level = level
                v_level = level
                
                # Compute joint exceedance probability
                joint_prob = compute_joint_exceedance(copula, u_level, v_level)
                
                # Compute independent exceedance probability
                indep_prob = (1 - u_level) * (1 - v_level)
                
                # Compute CPR
                cpr_val = joint_prob / indep_prob if indep_prob > 0 else np.nan
                
                joint_probs[str(level)] = {
                    'joint_exceedance': float(joint_prob),
                    'independent_exceedance': float(indep_prob),
                    'cpr': float(cpr_val)
                }
            
            results['tier2_analysis']['joint_exceedance'] = joint_probs
            
            # Compute joint return periods
            return_periods = {}
            for rp in compute_return_periods:
                # Convert return period to exceedance probability
                p_exceed = 1.0 / rp
                
                # Find corresponding quantile levels (non-exceedance)
                u_level = 1 - p_exceed
                v_level = 1 - p_exceed
                
                # Compute joint return periods
                try:
                    and_rp = compute_joint_return_period(copula, u_level, v_level, type='and')
                    or_rp = compute_joint_return_period(copula, u_level, v_level, type='or')
                    
                    return_periods[str(rp)] = {
                        'and_return_period': float(and_rp),
                        'or_return_period': float(or_rp),
                        'u_level': float(u_level),
                        'v_level': float(v_level)
                    }
                except Exception as e:
                    logger.warning(f"Error computing return period for {rp}: {e}")
            
            results['tier2_analysis']['joint_return_periods'] = return_periods
            
            # Compute conditional exceedance probabilities
            conditional_probs = {}
            for level in exceedance_levels:
                try:
                    # Different conditioning scenarios
                    u_med = 0.5  # Median value
                    v_med = 0.5
                    
                    # P(V>v | U=median)
                    p_v_given_u_med = compute_conditional_exceedance(
                        copula, u_med, level, conditional_var='v|u'
                    )
                    
                    # P(U>u | V=median)
                    p_u_given_v_med = compute_conditional_exceedance(
                        copula, v_med, level, conditional_var='u|v'
                    )
                    
                    # P(V>v | U>u)
                    p_v_given_u_exceed = compute_conditional_exceeding_given_exceeding(
                        copula, level, level, conditional_var='v|u'
                    )
                    
                    # P(U>u | V>v)
                    p_u_given_v_exceed = compute_conditional_exceeding_given_exceeding(
                        copula, level, level, conditional_var='u|v'
                    )
                    
                    conditional_probs[str(level)] = {
                        'p_v_given_u_med': float(p_v_given_u_med),
                        'p_u_given_v_med': float(p_u_given_v_med),
                        'p_v_given_u_exceed': float(p_v_given_u_exceed),
                        'p_u_given_v_exceed': float(p_u_given_v_exceed)
                    }
                except Exception as e:
                    logger.warning(f"Error computing conditional probability for level {level}: {e}")
            
            results['tier2_analysis']['conditional_probabilities'] = conditional_probs
            
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
                        'file': netcdf_file,
                        'copula_method': best_method,
                        'aic': fit_results.get('aic'),
                        'log_likelihood': fit_results.get('log_likelihood')
                    }
                    
                    # Add parameters
                    params = fit_results.get('params', {})
                    for param_name, value in params.items():
                        flat_results[f'param_{param_name}'] = value
                    
                    # Add tail dependence
                    if tail_dep:
                        flat_results['tail_lower'] = tail_dep.get('lower')
                        flat_results['tail_upper'] = tail_dep.get('upper')
                    
                    # Add 99% exceedance stats
                    level_99 = joint_probs.get('0.99', {})
                    if level_99:
                        flat_results['joint_exc_99'] = level_99.get('joint_exceedance')
                        flat_results['cpr_99'] = level_99.get('cpr')
                    
                    # Add 100-year return period
                    rp_100 = return_periods.get('100', {})
                    if rp_100:
                        flat_results['and_rp_100'] = rp_100.get('and_return_period')
                        flat_results['or_rp_100'] = rp_100.get('or_return_period')
                    
                    # Save as DataFrame
                    df = pd.DataFrame([flat_results])
                    df.to_parquet(output_file)
                    logger.info(f"Saved results to {output_file}")
                    
                    # Also save the full results as JSON for reference
                    json_file = os.path.join(output_dir, f"{station_code}_tier2_full.json")
                    serializable_results = ensure_json_serializable(results)
                    with open(json_file, 'w') as f:
                        json.dump(serializable_results, f, indent=2, cls=NumpyEncoder)
                    logger.info(f"Saved full results to {json_file}")
                    
                except Exception as e:
                    logger.error(f"Error saving parquet: {e}")
                    # Fall back to JSON
                    json_file = os.path.join(output_dir, f"{station_code}_tier2.json")
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
            logger.error(f"Error processing NetCDF file: {e}")
            return f"ERROR {station_code}: NetCDF processing failed - {str(e)}"
    except Exception as e:
        logger.error(f"Error processing station {station_code}: {e}")
        return f"ERROR {station_code}: {type(e).__name__}: {e}"


def run_tier2(
    netcdf_dir: str,
    output_dir: str,
    tier1_dir: Optional[str] = None,
    u_var: str = "sea_level",
    v_var: str = "total_precipitation",
    copula_method: str = "auto",
    compute_return_periods: List[float] = [10, 20, 50, 100],
    exceedance_levels: List[float] = [0.9, 0.95, 0.99],
    parallel: bool = False,
    output_format: str = "json",
    workers: Optional[int] = None,
    max_files: Optional[int] = None
) -> None:
    """
    Run Tier-2 copula analysis for all station NetCDFs in directory.
    
    Parameters
    ----------
    netcdf_dir : str
        Directory containing NetCDF files
    output_dir : str
        Directory to save Tier-2 outputs
    tier1_dir : Optional[str]
        Directory containing Tier-1 outputs (optional)
    u_var : str
        First variable for copula (default: sea_level)
    v_var : str
        Second variable for copula (default: total_precipitation)
    copula_method : str
        Copula type or 'auto' for automatic selection
    compute_return_periods : List[float]
        List of return periods to compute
    exceedance_levels : List[float]
        List of exceedance probabilities to analyze
    parallel : bool
        Whether to use parallel processing for model selection
    output_format : str
        Output format ('json' or 'parquet')
    workers : int
        Number of parallel workers (None=all cores)
    max_files : int
        Maximum number of files to process
        
    Raises
    ------
    FileNotFoundError
        If no NetCDF files found in directory
    ValueError
        If output format is not supported
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
    
    # Determine number of workers
    n_workers = workers or os.cpu_count() or 1
    logger.info(f"Running Tier-2 on {len(files)} stations with {n_workers} workers...")
    
    # Find corresponding Tier-1 files if provided
    tier1_files = {}
    if tier1_dir and os.path.exists(tier1_dir):
        logger.info(f"Looking for Tier-1 outputs in {tier1_dir}")
        for ext in ['.json', '.parquet']:
            tier1_pattern = os.path.join(tier1_dir, f"*_tier1{ext}")
            found_files = glob.glob(tier1_pattern)
            
            for file in found_files:
                station_code = os.path.basename(file).split('_tier1')[0]
                tier1_files[station_code] = file
                
        logger.info(f"Found {len(tier1_files)} Tier-1 output files")
    
    # Prepare function with fixed parameters
    func = partial(
        process_station,
        output_dir=output_dir,
        u_var=u_var,
        v_var=v_var,
        copula_method=copula_method,
        compute_return_periods=compute_return_periods,
        exceedance_levels=exceedance_levels,
        parallel=parallel,
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
        
        # Create (netcdf_file, tier1_file) pairs
        tasks = []
        for nc_file in files:
            station_code = os.path.splitext(os.path.basename(nc_file))[0]
            tier1_file = tier1_files.get(station_code)
            tasks.append((nc_file, tier1_file))
        
        # Run tasks
        for i, (nc_file, tier1_file) in enumerate(tasks):
            future = executor.submit(func, nc_file, tier1_file)
            tasks[i] = future
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(tasks)):
            result = future.result()
            print(f"[{i+1}/{len(files)}] {result}")
            
            if result.startswith("OK"):
                success_count += 1
            elif result.startswith("ERROR"):
                error_count += 1
            elif result.startswith("SKIP"):
                skip_count += 1
    
    logger.info(f"Tier-2 processing complete: {success_count} succeeded, {error_count} failed, {skip_count} skipped")


def smoke_test(verbose: bool = True) -> bool:
    """
    Run a quick self-check with synthetic data.
    
    Parameters
    ----------
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    bool
        True if test passes, False otherwise
    """
    import tempfile
    from scipy.stats import norm
    
    print("Running Tier-2 smoke test with synthetic data...")
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate synthetic data
            n = 1000
            rho = 0.7
            
            # Generate correlated normal data
            np.random.seed(42)
            mean = [0, 0]
            cov = [[1, rho], [rho, 1]]
            x, y = np.random.multivariate_normal(mean, cov, n).T
            
            # Create time index
            times = pd.date_range('2000-01-01', periods=n, freq='h')
            
            # Create xarray Dataset
            ds = xr.Dataset(
                data_vars={
                    'sea_level': ('time', x),
                    'total_precipitation': ('time', y)
                },
                coords={
                    'time': times
                }
            )
            
            # Save to NetCDF
            nc_file = os.path.join(temp_dir, 'synthetic_station.nc')
            ds.to_netcdf(nc_file)
            
            if verbose:
                print(f"Created synthetic NetCDF: {nc_file}")
                print(f"Data shape: {ds.dims}")
                print(f"Variables: {list(ds.data_vars)}")
                print(f"Pearson correlation: {rho}")
                print(f"Kendall's tau: {np.sin(np.pi*rho/6):.4f}")
            
            # Create a mock Tier-1 output
            tier1_data = {
                'station': 'synthetic_station',
                'sea_level': {
                    'threshold': 1.0
                },
                'total_precipitation': {
                    'threshold': 1.0
                },
                'joint': {
                    'empirical': {
                        'cpr': 2.5
                    }
                }
            }
            
            tier1_file = os.path.join(temp_dir, 'synthetic_station_tier1.json')
            with open(tier1_file, 'w') as f:
                json.dump(tier1_data, f)
                
            if verbose:
                print(f"Created mock Tier-1 output: {tier1_file}")
            
            # Set up output directory
            output_dir = os.path.join(temp_dir, 'tier2_output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Test process_station function
            if verbose:
                print("\nTesting process_station function...")
                
            result = process_station(
                netcdf_file=nc_file,
                tier1_file=tier1_file,
                output_dir=output_dir,
                copula_method='auto'
            )
            
            if verbose:
                print(f"process_station result: {result}")
            
            # Check if output was created
            output_file = os.path.join(output_dir, 'synthetic_station_tier2.json')
            if not os.path.exists(output_file):
                print(f"ERROR: Output file not created: {output_file}")
                return False
                
            # Load and validate output
            with open(output_file, 'r') as f:
                output_data = json.load(f)
                
            if verbose:
                print("\nValidating output...")
                print(f"Copula method: {output_data['tier2_analysis']['copula']['method']}")
                print(f"AIC: {output_data['tier2_analysis']['copula'].get('aic')}")
                
                # Display joint exceedance probabilities
                if 'joint_exceedance' in output_data['tier2_analysis']:
                    joint_exc = output_data['tier2_analysis']['joint_exceedance']
                    for level, data in joint_exc.items():
                        print(f"Level {level} - Joint exceedance: {data['joint_exceedance']:.6f}, CPR: {data['cpr']:.4f}")
                
                # Display joint return periods
                if 'joint_return_periods' in output_data['tier2_analysis']:
                    joint_rps = output_data['tier2_analysis']['joint_return_periods']
                    for rp, data in joint_rps.items():
                        print(f"Return period {rp} - AND: {data['and_return_period']:.2f}, OR: {data['or_return_period']:.2f}")
            
            # Test run_tier2 function (with just one file)
            if verbose:
                print("\nTesting run_tier2 function...")
                
            run_tier2(
                netcdf_dir=temp_dir,
                output_dir=output_dir,
                tier1_dir=temp_dir,
                copula_method='auto',
                workers=1
            )
            
            if verbose:
                print("Smoke test completed successfully.")
                
            return True
            
    except Exception as e:
        print(f"Smoke test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    import sys
    import concurrent.futures
    
    parser = argparse.ArgumentParser(description='Tier-2 Copula Analysis')
    parser.add_argument('--netcdf-dir', required=True, help='Directory of cleaned NetCDFs')
    parser.add_argument('--output-dir', required=True, help='Directory to save Tier-2 output files')
    parser.add_argument('--tier1-dir', help='Directory with Tier-1 results (optional)')
    parser.add_argument('--u-var', default='sea_level', help='First variable name')
    parser.add_argument('--v-var', default='total_precipitation', help='Second variable name')
    parser.add_argument('--method', choices=['auto', 'gumbel', 'frank', 'student', 'gaussian', 'clayton'],
                       default='auto', help='Copula method (auto or specific family)')
    parser.add_argument('--return-periods', nargs='+', type=float, default=[10, 20, 50, 100],
                       help='Return periods to compute')
    parser.add_argument('--levels', nargs='+', type=float, default=[0.9, 0.95, 0.99],
                       help='Exceedance probability levels to analyze')
    parser.add_argument('--output-format', choices=['json', 'parquet'], default='json',
                       help='Output format (json or parquet)')
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of parallel workers (0=all cores)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process (for testing)')
    parser.add_argument('--smoke-test', action='store_true', help='Run smoke test and exit')
    
    args = parser.parse_args()
    
    if args.smoke_test:
        success = smoke_test(verbose=True)
        sys.exit(0 if success else 1)
    
    try:
        run_tier2(
            netcdf_dir=args.netcdf_dir,
            output_dir=args.output_dir,
            tier1_dir=args.tier1_dir,
            u_var=args.u_var,
            v_var=args.v_var,
            copula_method=args.method,
            compute_return_periods=args.return_periods,
            exceedance_levels=args.levels,
            output_format=args.output_format,
            workers=(args.workers if args.workers > 0 else None),
            max_files=args.max_files
        )
        print("Tier-2 analysis complete!")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)