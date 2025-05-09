"""
Module: tier3_bn.py
Responsibilities:
- Main orchestrator for Tier 3 Bayesian Network analysis
- Command-line interface
- Reads data from Tier 1 NetCDFs
- Coordinates processing pipeline
- Saves BN models and JSON summaries
"""
import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pgmpy.models import DiscreteBayesianNetwork

# Import core modules
from src.compound_flooding.bn_core import (
    NODE_STATES, SEA_LEVEL_STATE, PRECIPITATION_STATE, WIND_SPEED,
    WIND_DIRECTION, PRESSURE_ANOMALY, SEASON, COMPOUND_FLOOD_RISK, SEASON_STATES,COMPOUND_FLOOD_RISK_STATES,
    build_dag, save_bn_xmlbif, node_state_mapping_to_json
)
from src.compound_flooding.bn_learning import (
    discretise_df, compute_quantile_thresholds, fit_cpts
)
from src.compound_flooding.bn_diagnostics import (
    kfold_cv, brier_score, sensitivity
)

# Import from existing tiers
from src.compound_flooding.data_io import validate_paths

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(
    nc_file: str,
    datetime_col: str = 'datetime'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and preprocess data from a NetCDF file for Bayesian Network analysis."""
    logger.info(f"Loading data from {nc_file}")
    
    # Open the NetCDF file
    ds = xr.open_dataset(nc_file)
    
    # Create DataFrame from Dataset
    df = ds.to_dataframe().reset_index()
    
    # Make sure datetime column exists
    if datetime_col not in df.columns and 'time' in df.columns:
        df[datetime_col] = df['time']
        
    if datetime_col not in df.columns:
        logger.warning(f"Datetime column {datetime_col} not found in DataFrame")
    
    # Check if we have enough data
    if len(df) == 0:
        raise ValueError(f"No data found in {nc_file}")
    
    # Process variables
    metadata = {
        'station_code': os.path.splitext(os.path.basename(nc_file))[0],
        'n_rows': len(df),
        'date_range': [df[datetime_col].min().strftime('%Y-%m-%d'), 
                       df[datetime_col].max().strftime('%Y-%m-%d')],
        'variables': {}
    }
    
    # Calculate derived variables
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        # Calculate wind speed
        df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
        
        # Calculate wind direction (meteorological convention: 0=N, 90=E, etc.)
        df['wind_direction_deg'] = 270 - np.degrees(np.arctan2(df['v_component_of_wind_10m'], 
                                                               df['u_component_of_wind_10m']))
        # Ensure angles are in [0, 360)
        df['wind_direction_deg'] = df['wind_direction_deg'] % 360
        
        metadata['variables']['wind_speed'] = {
            'mean': float(df['wind_speed'].mean()),
            'std': float(df['wind_speed'].std()),
            'min': float(df['wind_speed'].min()),
            'max': float(df['wind_speed'].max())
        }
        
        metadata['variables']['wind_direction_deg'] = {
            'mean': float(df['wind_direction_deg'].mean()),
            'circular_mean': float(np.degrees(np.arctan2(
                np.sum(np.sin(np.radians(df['wind_direction_deg']))),
                np.sum(np.cos(np.radians(df['wind_direction_deg'])))
            )) % 360)
        }
    
    # Calculate pressure anomaly if surface_pressure is available
    if 'surface_pressure' in df.columns:
        df['pressure_anomaly'] = df['surface_pressure'] - df['surface_pressure'].mean()
        
        metadata['variables']['pressure_anomaly'] = {
            'mean': float(df['pressure_anomaly'].mean()),
            'std': float(df['pressure_anomaly'].std()),
            'min': float(df['pressure_anomaly'].min()),
            'max': float(df['pressure_anomaly'].max())
        }
    
    # Add season based on datetime
    if datetime_col in df.columns:
        # Extract month
        df['month'] = pd.DatetimeIndex(df[datetime_col]).month
        
        # Assign seasons (Northern Hemisphere convention)
        conditions = [
            (df['month'] >= 12) | (df['month'] <= 2),  # Winter: Dec-Feb
            (df['month'] >= 3) & (df['month'] <= 5),   # Spring: Mar-May
            (df['month'] >= 6) & (df['month'] <= 8),   # Summer: Jun-Aug
            (df['month'] >= 9) & (df['month'] <= 11)   # Fall: Sep-Nov
        ]
        choices = SEASON_STATES
        
        df['season'] = np.select(conditions, choices, default=None)
        
        # Count observations per season
        season_counts = df['season'].value_counts().to_dict()
        metadata['variables']['season'] = {
            'counts': season_counts
        }
    
    # Close the dataset
    ds.close()
    
    return df, metadata

def create_node_mappings(
    df: pd.DataFrame,
    coastline_angle: float = 0.0
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """
    Create mappings for node discretization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    coastline_angle : float, optional
        Coastline angle in degrees (0 means east-west coastline)
        
    Returns
    -------
    Dict[str, Dict[str, Union[float, List[float]]]]
        Dictionary mapping node names to their discretization parameters
    """
    # Calculate quantile thresholds
    quantiles = {
        'sea_level': [0.9, 0.99],
        'total_precipitation': [0.9, 0.99],
        'wind_speed': [0.75, 0.95]
    }
    
    thresholds = compute_quantile_thresholds(df, quantiles.keys(), quantiles)
    
    # Create node mappings
    node_mappings = {}
    
    # Sea level state
    if 'sea_level' in thresholds:
        node_mappings[SEA_LEVEL_STATE] = {
            'threshold_90th': thresholds['sea_level']['threshold_90th'],
            'threshold_99th': thresholds['sea_level']['threshold_99th']
        }
    
    # Precipitation state
    if 'total_precipitation' in thresholds:
        node_mappings[PRECIPITATION_STATE] = {
            'threshold_90th': thresholds['total_precipitation']['threshold_90th'],
            'threshold_99th': thresholds['total_precipitation']['threshold_99th']
        }
    
    # Wind speed
    if 'wind_speed' in thresholds:
        node_mappings[WIND_SPEED] = {
            'threshold_75th': thresholds['wind_speed']['threshold_75th'],
            'threshold_95th': thresholds['wind_speed']['threshold_95th']
        }
    
    # Wind direction
    # Define offshore, alongshore, and onshore sectors relative to coastline
    # Coastline angle: angle from north to coastline, measured clockwise
    coastline_rad = np.radians(coastline_angle)
    
    # Offshore angles: 90 degrees centered on direction away from coast
    offshore_center = (coastline_angle + 90) % 360
    offshore_start = (offshore_center - 45) % 360
    offshore_end = (offshore_center + 45) % 360
    
    # Alongshore angles: 90 degrees centered on directions parallel to coast
    alongshore_center1 = coastline_angle
    alongshore_center2 = (coastline_angle + 180) % 360
    
    # Define sectors
    if offshore_start < offshore_end:
        offshore = [(offshore_start, offshore_end)]
    else:
        offshore = [(offshore_start, 360), (0, offshore_end)]
    
    alongshore1_start = (alongshore_center1 - 45) % 360
    alongshore1_end = (alongshore_center1 + 45) % 360
    alongshore2_start = (alongshore_center2 - 45) % 360
    alongshore2_end = (alongshore_center2 + 45) % 360
    
    if alongshore1_start < alongshore1_end:
        alongshore1 = [(alongshore1_start, alongshore1_end)]
    else:
        alongshore1 = [(alongshore1_start, 360), (0, alongshore1_end)]
    
    if alongshore2_start < alongshore2_end:
        alongshore2 = [(alongshore2_start, alongshore2_end)]
    else:
        alongshore2 = [(alongshore2_start, 360), (0, alongshore2_end)]
    
    alongshore = alongshore1 + alongshore2
    
    node_mappings[WIND_DIRECTION] = {
        'offshore_angles': offshore,
        'alongshore_angles': alongshore
    }
    
    # Pressure anomaly
    if 'pressure_anomaly' in df.columns:
        std_dev = df['pressure_anomaly'].std()
        node_mappings[PRESSURE_ANOMALY] = {
            'high_threshold': std_dev,
            'low_threshold': -std_dev
        }
    
    # Season
    if 'datetime' in df.columns:
        node_mappings[SEASON] = {
            'datetime_col': 'datetime'
        }
    
    return node_mappings

def process_station(
    nc_file: str,
    output_dir: str,
    coastline_angle: float = 0.0,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process a single station for Tier 3 Bayesian Network analysis.
    
    Parameters
    ----------
    nc_file : str
        Path to NetCDF file
    output_dir : str
        Directory to save outputs
    coastline_angle : float, optional
        Coastline angle in degrees (0 means east-west coastline)
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with analysis results
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    station_code = os.path.splitext(os.path.basename(nc_file))[0]
    logger.info(f"Processing station {station_code}")
    
    # Create output directories
    model_dir = os.path.join(output_dir, 'models')
    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    
    # Define output paths
    model_path = os.path.join(model_dir, f"{station_code}.xml")
    summary_path = os.path.join(summary_dir, f"{station_code}_tier3_bn.json")
    
    # Check if outputs already exist
    if os.path.exists(model_path) and os.path.exists(summary_path):
        logger.info(f"Outputs already exist for station {station_code}, skipping")
        # Load existing summary
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    # Load and preprocess data
    try:
        df, metadata = load_and_preprocess_data(nc_file)
    except Exception as e:
        logger.error(f"Error loading data for station {station_code}: {e}")
        return {'error': str(e), 'station': station_code}
    
    # Create node mappings for discretization
    node_mappings = create_node_mappings(df, coastline_angle)
    
    bn_df = pd.DataFrame()
    
    # Map input variables to BN node names, excluding datetime
    if 'sea_level' in df.columns:
        bn_df[SEA_LEVEL_STATE] = df['sea_level']
        
    if 'total_precipitation' in df.columns:
        bn_df[PRECIPITATION_STATE] = df['total_precipitation']
        
    if 'wind_speed' in df.columns:
        bn_df[WIND_SPEED] = df['wind_speed']
        
    if 'wind_direction_deg' in df.columns:
        bn_df[WIND_DIRECTION] = df['wind_direction_deg']
        
    if 'pressure_anomaly' in df.columns:
        bn_df[PRESSURE_ANOMALY] = df['pressure_anomaly']
        
    if 'season' in df.columns:
        bn_df[SEASON] = df['season']
    
    # Add compound flood risk (initially None)
    bn_df[COMPOUND_FLOOD_RISK] = None
    
    # Discretize data
    try:
        discrete_df = discretise_df(bn_df, node_mappings, random_state=random_state)
    except Exception as e:
        logger.error(f"Error discretizing data for station {station_code}: {e}")
        return {'error': str(e), 'station': station_code}
    
    # Define compound flood risk based on sea level and precipitation
    # High risk: both sea level and precipitation are above 99th percentile
    # Moderate risk: one of sea level or precipitation is above 99th percentile, or both above 90th
    # Low risk: all other cases
    conditions = [
        (discrete_df[SEA_LEVEL_STATE] == "Above_99th") & (discrete_df[PRECIPITATION_STATE] == "Above_99th"),
        (discrete_df[SEA_LEVEL_STATE] == "Above_99th") | (discrete_df[PRECIPITATION_STATE] == "Above_99th"),
        (discrete_df[SEA_LEVEL_STATE] == "90th_99th") & (discrete_df[PRECIPITATION_STATE] == "90th_99th")
    ]
    choices = [COMPOUND_FLOOD_RISK_STATES[2], COMPOUND_FLOOD_RISK_STATES[1], COMPOUND_FLOOD_RISK_STATES[1]]
    discrete_df[COMPOUND_FLOOD_RISK] = np.select(conditions, choices, default=COMPOUND_FLOOD_RISK_STATES[0])
    
    # Create DAG and fit CPTs
    model = build_dag()
    
    try:
        fitted_model = fit_cpts(
            model, discrete_df, method='bayes', 
            pseudocount=1.0, random_state=random_state
        )
    except Exception as e:
        logger.error(f"Error fitting CPTs for station {station_code}: {e}")
        return {'error': str(e), 'station': station_code}
    
    # Perform cross-validation
    try:
        cv_results = kfold_cv(
            model, discrete_df, k=5, method='bayes',
            pseudocount=1.0, random_state=random_state
        )
    except Exception as e:
        logger.warning(f"Error in cross-validation for station {station_code}: {e}")
        cv_results = {'error': str(e)}
    
    # Perform sensitivity analysis
    try:
        sensitivity_results = sensitivity(
            fitted_model, target_node=COMPOUND_FLOOD_RISK,
            n_samples=1000, random_state=random_state
        )
    except Exception as e:
        logger.warning(f"Error in sensitivity analysis for station {station_code}: {e}")
        sensitivity_results = {'error': str(e)}
    
    # Save model to XMLBIF
    save_bn_xmlbif(fitted_model, model_path)
    
    # Prepare summary data
    thresholds = {}
    for node, mapping in node_mappings.items():
        thresholds[node] = {}
        for key, value in mapping.items():
            if key.startswith('threshold_'):
                thresholds[node][key] = value
    
    summary = {
        'station': station_code,
        'tier3_analysis': {
            'bayesian_network': {
                'structure': {
                    'nodes': list(model.nodes()),
                    'edges': list(model.edges()),
                    'node_states': node_state_mapping_to_json(NODE_STATES, thresholds)
                },
                'validation': {
                    'log_likelihood': cv_results.get('log_likelihood'),
                    'accuracy': cv_results.get('accuracy'),
                    'brier_score': cv_results.get('brier_score')
                },
                'sensitivity': {
                    node: {
                        'average_impact': metrics['average_impact'],
                        'max_impact': metrics['max_impact']
                    } for node, metrics in sensitivity_results.items()
                } if isinstance(sensitivity_results, dict) else {'error': str(sensitivity_results)}
            }
        },
        'metadata': metadata
    }
    
    # Calculate conditional probabilities of interest
    # Save summary to JSON
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Completed processing for station {station_code}")
    return summary

def main():
    """Command-line interface for Tier 3 Bayesian Network analysis."""
    parser = argparse.ArgumentParser(description='Tier 3 Bayesian Network Analysis')
    
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')
    
    # 'run' subcommand for processing a single station
    run_parser = subparsers.add_parser('run', help='Process a single station')
    run_parser.add_argument('--station-code', required=True, help='Station code')
    run_parser.add_argument('--netcdf-dir', default='outputs/cleaned', 
                           help='Directory containing cleaned NetCDF files')
    run_parser.add_argument('--output-dir', default='outputs/tier3', 
                           help='Directory to save Tier 3 outputs')
    run_parser.add_argument('--coastline-angle', type=float, default=0.0, 
                           help='Coastline angle in degrees (0=east-west)')
    run_parser.add_argument('--random-state', type=int, help='Random state for reproducibility')
    
    # 'batch' subcommand for processing multiple stations
    batch_parser = subparsers.add_parser('batch', help='Process multiple stations')
    batch_parser.add_argument('--station-list', help='File with list of station codes')
    batch_parser.add_argument('--netcdf-dir', default='outputs/cleaned', 
                            help='Directory containing cleaned NetCDF files')
    batch_parser.add_argument('--output-dir', default='outputs/tier3', 
                            help='Directory to save Tier 3 outputs')
    batch_parser.add_argument('--coastline-angle', type=float, default=0.0, 
                            help='Coastline angle in degrees (0=east-west)')
    batch_parser.add_argument('--workers', type=int, default=1, 
                            help='Number of parallel workers (1=sequential)')
    batch_parser.add_argument('--random-state', type=int, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        # Process a single station
        station_code = args.station_code
        nc_file = os.path.join(args.netcdf_dir, f"{station_code}.nc")
        
        if not os.path.exists(nc_file):
            logger.error(f"NetCDF file not found: {nc_file}")
            return 1
        
        result = process_station(
            nc_file=nc_file,
            output_dir=args.output_dir,
            coastline_angle=args.coastline_angle,
            random_state=args.random_state
        )
        
        if 'error' in result:
            logger.error(f"Error processing station {station_code}: {result['error']}")
            return 1
        
        logger.info(f"Successfully processed station {station_code}")
        return 0
        
    elif args.command == 'batch':
        # Process multiple stations
        raise NotImplementedError("Batch processing not implemented yet")
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    main()