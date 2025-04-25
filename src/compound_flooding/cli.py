#!/usr/bin/env python
"""
Command-line interface for compound flooding analysis.

This script provides a command-line interface for running
compound flooding analysis workflows, following the three-tier
approach outlined in the methodology.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
import logging
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('compound_flooding')

# Import our modules
try:
    from compound_flooding.data_io import DataLoader
    from compound_flooding.preprocess import DataPreprocessor
    from compound_flooding.tier1_stats import Tier1Analyzer
except ImportError:
    # If importing as a package fails, try relative import
    # This helps with development and testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from compound_flooding.data_io import DataLoader
    from compound_flooding.preprocess import DataPreprocessor
    from compound_flooding.tier1_stats import Tier1Analyzer

# Default directories
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_DATA_DIR = Path("compound_flooding/GESLA_ERA5_with_sea_level")
DEFAULT_METADATA_PATH = Path("data/GESLA/usa_metadata.csv")


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Compound flooding analysis tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add common arguments
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Use GPU acceleration if available"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing station CSV files"
    )
    parser.add_argument(
        "--metadata-path", 
        type=str, 
        default=str(DEFAULT_METADATA_PATH),
        help="Path to the metadata CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--use-dask", 
        action="store_true", 
        help="Use Dask for parallel processing"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", 
        help="Ingest and preprocess station data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ingest_parser.add_argument(
        "--station-codes", 
        type=str, 
        nargs="+", 
        help="List of station codes to process (default: all)"
    )
    ingest_parser.add_argument(
        "--start-date", 
        type=str, 
        help="Start date in ISO format (YYYY-MM-DD)"
    )
    ingest_parser.add_argument(
        "--end-date", 
        type=str, 
        help="End date in ISO format (YYYY-MM-DD)"
    )
    ingest_parser.add_argument(
        "--detrend-sea-level", 
        action="store_true", 
        help="Remove linear trend from sea level"
    )
    ingest_parser.add_argument(
        "--no-gap-handling", 
        action="store_true", 
        help="Skip gap handling"
    )
    ingest_parser.add_argument(
        "--no-spike-handling", 
        action="store_true", 
        help="Skip spike handling"
    )
    ingest_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    ingest_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )
    
    # Tier 1 command
    tier1_parser = subparsers.add_parser(
        "tier1", 
        help="Run Tier 1 statistical analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    tier1_parser.add_argument(
        "--station-codes", 
        type=str, 
        nargs="+", 
        help="List of station codes to process (default: all)"
    )
    tier1_parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile threshold for POT analysis (e.g., 99.0)"
    )
    tier1_parser.add_argument(
        "--lag-window",
        type=int,
        default=24,
        help="Window (in hours) for lag analysis"
    )
    tier1_parser.add_argument(
        "--min-cluster-separation",
        type=int,
        default=72,
        help="Minimum separation (in hours) between extreme events"
    )
    tier1_parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["sea_level", "total_precipitation"],
        help="Variables to analyze"
    )
    tier1_parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip saving diagnostic plots"
    )
    tier1_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )
    tier1_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    
    # Future command: Tier 2 (Copula modeling)
    # tier2_parser = subparsers.add_parser(...)
    
    # Future command: Tier 3 (Advanced modeling)
    # tier3_parser = subparsers.add_parser(...)
    
    # Future command: Plotting
    # plot_parser = subparsers.add_parser(...)
    
    return parser


# Define the process_station function at the module level (not nested)
def process_station(station_code: str, args, data_loader: DataLoader, preprocessor: DataPreprocessor) -> Dict[str, Any]:
    """
    Process a station for the ingest command.
    
    Parameters
    ----------
    station_code : str
        Station code to process.
    args : argparse.Namespace
        Command-line arguments.
    data_loader : DataLoader
        Initialized data loader.
    preprocessor : DataPreprocessor
        Initialized preprocessor.
    
    Returns
    -------
    Dict[str, Any]
        Processing result.
    """
    try:
        output_dir = Path(args.output_dir) / "netcdf"
        output_path = output_dir / f"{station_code}_preprocessed.nc"
        
        # Check if output file exists
        if output_path.exists() and not args.overwrite:
            logger.info(f"Skipping {station_code}: output file already exists (use --overwrite to force)")
            return {"status": "skipped", "station_code": station_code}
        
        logger.info(f"Processing station: {station_code}")
        
        # Load station data
        ds = data_loader.load_station_data_as_xarray(
            station_code=station_code,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Log basic info about the dataset
        logger.info(f"Loaded {station_code} data: {len(ds.time)} timesteps from "
                   f"{ds.time.values[0]} to {ds.time.values[-1]}")
        
        # Preprocess data
        ds = preprocessor.preprocess_station(
            ds=ds,
            detrend_sea_level=args.detrend_sea_level,
            handle_gaps=not args.no_gap_handling,
            handle_spikes=not args.no_spike_handling
        )
        
        # Save preprocessed data
        preprocessor.save_preprocessed_data(
            ds=ds,
            output_path=output_path,
            compress=True
        )
        
        logger.info(f"Successfully processed {station_code}")
        return {"status": "success", "station_code": station_code}
        
    except Exception as e:
        logger.error(f"Error processing station {station_code}: {str(e)}", exc_info=args.verbose > 0)
        return {"status": "error", "station_code": station_code, "error": str(e)}


# Define the analyze_station function at the module level (not nested)
def analyze_station(station_code: str, args, analyzer: Tier1Analyzer) -> Dict[str, Any]:
    """
    Analyze a station for Tier 1 command.
    
    Parameters
    ----------
    station_code : str
        Station code to analyze.
    args : argparse.Namespace
        Command-line arguments.
    analyzer : Tier1Analyzer
        Initialized analyzer.
    
    Returns
    -------
    Dict[str, Any]
        Analysis result.
    """
    try:
        output_dir = Path(args.output_dir) / "tier1"
        output_path = output_dir / f"{station_code}_tier1_results.parquet"
        
        # Check if output file exists
        if output_path.exists() and not args.overwrite:
            logger.info(f"Skipping {station_code}: Tier 1 results already exist (use --overwrite to force)")
            return {"status": "skipped", "station_code": station_code}
        
        logger.info(f"Analyzing station: {station_code}")
        
        # Run Tier 1 analysis
        results = analyzer.analyze_station(
            station_code=station_code,
            variables=args.variables
        )
        
        if results["status"] == "success":
            logger.info(f"Successfully analyzed {station_code}")
            return {"status": "success", "station_code": station_code}
        else:
            logger.warning(f"Analysis completed with warnings for {station_code}: {results.get('message', 'Unknown issue')}")
            return {"status": "warning", "station_code": station_code, "message": results.get("message")}
        
    except Exception as e:
        logger.error(f"Error analyzing station {station_code}: {str(e)}", exc_info=args.verbose > 0)
        return {"status": "error", "station_code": station_code, "error": str(e)}


# Helper functions for multiprocessing
def unpack_process_station(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Unpack arguments and call process_station.
    
    Parameters
    ----------
    args_tuple : Tuple
        Tuple of arguments for process_station.
    
    Returns
    -------
    Dict[str, Any]
        Result from process_station.
    """
    return process_station(*args_tuple)


def unpack_analyze_station(args_tuple: Tuple) -> Dict[str, Any]:
    """
    Unpack arguments and call analyze_station.
    
    Parameters
    ----------
    args_tuple : Tuple
        Tuple of arguments for analyze_station.
    
    Returns
    -------
    Dict[str, Any]
        Result from analyze_station.
    """
    return analyze_station(*args_tuple)


def ingest_command(args: argparse.Namespace) -> None:
    """
    Run the ingest command to preprocess station data.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    # Set up logging based on verbosity
    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    logger.info("Starting ingest command")
    
    # Initialize data loader
    try:
        data_loader = DataLoader(
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            use_dask=args.use_dask
        )
        logger.info(f"Initialized data loader with {len(data_loader.station_codes)} stations")
    except Exception as e:
        logger.error(f"Failed to initialize data loader: {e}")
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Get station codes to process
    if args.station_codes:
        station_codes = args.station_codes
        logger.info(f"Processing {len(station_codes)} specified stations")
    else:
        station_codes = data_loader.station_codes
        logger.info(f"Processing all {len(station_codes)} stations")
    
    # Create output directory
    output_dir = Path(args.output_dir) / "netcdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Setup parallel processing if requested
    if args.use_dask and args.max_workers > 1:
        import dask
        from dask.distributed import Client, LocalCluster
        
        # Configure Dask to use GPU if requested
        if args.gpu:
            try:
                import cupy  # Check if cupy is available
                dask.config.set({"array.backend": "cupy"})
                logger.info("Using CuPy as Dask array backend")
            except ImportError:
                logger.warning("CuPy not available. GPU acceleration disabled.")
        
        # Setup local cluster
        cluster = LocalCluster(
            n_workers=args.max_workers,
            threads_per_worker=1,
            processes=True
        )
        client = Client(cluster)
        logger.info(f"Initialized Dask cluster with {args.max_workers} workers")
    
    # Process each station
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    # Run processing in parallel or sequentially
    if args.max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Create a list of tasks with all arguments
            tasks = [(station_code, args, data_loader, preprocessor) for station_code in station_codes]
            results = list(executor.map(unpack_process_station, tasks))
    else:
        results = [process_station(code, args, data_loader, preprocessor) for code in station_codes]
    
    # Count successful and failed stations
    for result in results:
        if result["status"] == "success":
            processed_count += 1
        elif result["status"] == "error":
            error_count += 1
    
    # Clean up Dask client if used
    if args.use_dask and args.max_workers > 1:
        client.close()
        cluster.close()
    
    # Report summary
    elapsed_time = time.time() - start_time
    logger.info(f"Ingest completed in {elapsed_time:.1f} seconds.")
    logger.info(f"Processed: {processed_count}/{len(station_codes)} stations")
    if error_count > 0:
        logger.warning(f"Errors: {error_count}/{len(station_codes)} stations")


def tier1_command(args: argparse.Namespace) -> None:
    """
    Run Tier 1 statistical analysis.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    # Set up logging based on verbosity
    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    logger.info("Starting Tier 1 analysis")
    
    # Initialize the Tier1Analyzer
    preprocessed_dir = Path(args.output_dir) / "netcdf"
    output_dir = Path(args.output_dir) / "tier1"
    
    # Check if preprocessed directory exists
    if not preprocessed_dir.exists():
        logger.error(f"Preprocessed directory not found: {preprocessed_dir}")
        logger.error("Run the 'ingest' command first to preprocess data.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = Tier1Analyzer(
        preprocessed_dir=preprocessed_dir,
        output_dir=output_dir,
        threshold_percentile=args.threshold_percentile,
        lag_window=args.lag_window,
        min_cluster_separation=args.min_cluster_separation,
        save_diagnostics=not args.no_diagnostics
    )
    
    # Get station codes to process
    if args.station_codes:
        station_codes = args.station_codes
        logger.info(f"Analyzing {len(station_codes)} specified stations")
    else:
        # Get available preprocessed station files
        station_files = list(preprocessed_dir.glob("*_preprocessed.nc"))
        station_codes = [f.stem.split('_')[0] for f in station_files]
        logger.info(f"Analyzing all {len(station_codes)} preprocessed stations")
    
    if not station_codes:
        logger.error("No stations found for analysis.")
        sys.exit(1)
    
    # Run processing in parallel or sequentially
    start_time = time.time()
    
    if args.max_workers > 1:
        # For extreme value analysis, we need more memory per worker
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Create a list of tasks with all arguments
            tasks = [(station_code, args, analyzer) for station_code in station_codes]
            results = list(executor.map(unpack_analyze_station, tasks))
    else:
        results = [analyze_station(code, args, analyzer) for code in station_codes]
    
    # Count successful and failed stations
    success_count = 0
    warning_count = 0
    error_count = 0
    skipped_count = 0
    
    for result in results:
        if result["status"] == "success":
            success_count += 1
        elif result["status"] == "warning":
            warning_count += 1
        elif result["status"] == "error":
            error_count += 1
        elif result["status"] == "skipped":
            skipped_count += 1
    
    # Report summary
    elapsed_time = time.time() - start_time
    logger.info(f"Tier 1 analysis completed in {elapsed_time:.1f} seconds.")
    logger.info(f"Results summary:")
    logger.info(f"  Success: {success_count}/{len(station_codes)} stations")
    
    if warning_count > 0:
        logger.warning(f"  Warnings: {warning_count}/{len(station_codes)} stations")
    
    if error_count > 0:
        logger.error(f"  Errors: {error_count}/{len(station_codes)} stations")
    
    if skipped_count > 0:
        logger.info(f"  Skipped: {skipped_count}/{len(station_codes)} stations (already processed)")


def main() -> None:
    """
    Main entry point for the CLI.
    """
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging level based on verbosity
    if args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logger.setLevel(logging.INFO)
    
    # If no command specified, show help
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Run appropriate command
    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "tier1":
        tier1_command(args)
    # Additional commands would be handled here
    # elif args.command == "tier2":
    #     tier2_command(args)
    # elif args.command == "tier3":
    #     tier3_command(args)
    # elif args.command == "plot":
    #     plot_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()