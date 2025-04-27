"""
Base visualization module for compound flooding analysis.

This module provides common utilities, settings, and helper functions 
for creating consistent, publication-ready visualizations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json
import glob
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard figure sizes (in inches)
FIG_SIZES = {
    'small': (6, 4),      # For simple plots
    'medium': (8, 6),     # For standard plots
    'large': (10, 8),     # For detailed plots
    'wide': (12, 6),      # For side-by-side comparisons
    'tall': (8, 10),      # For vertical stacked plots
    'full': (12, 10),     # For complex multipanel figures
    'poster': (16, 12),   # For poster presentations
    'square': (8, 8),     # For square plots
    'wide_small': (8, 4), # For wide but small plots
    'map': (12, 8)        # For map plotting
}

# Custom colormaps for common visualizations
# Bivariate colormap for joint density
RED_BLUE_CMAP = LinearSegmentedColormap.from_list(
    'red_blue', ['#FFFFFF', '#6BAED6', '#08519C', '#843C39', '#E6550D', '#FFEDA0']
)

# A colormap for CPR values (centered at 1.0)
CPR_CMAP = LinearSegmentedColormap.from_list(
    'cpr_cmap', ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
)

# USGS-style colormap for flood risk
RISK_CMAP = LinearSegmentedColormap.from_list(
    'risk_cmap', ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
)

# Oceanic colormap for sea level
SEA_CMAP = LinearSegmentedColormap.from_list(
    'sea_cmap', ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c']
)

# Precipitation colormap
PRECIP_CMAP = LinearSegmentedColormap.from_list(
    'precip_cmap', ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c']
)

def set_publication_style(use_tex: bool = False):
    """
    Set matplotlib parameters for publication-quality figures.
    
    Parameters
    ----------
    use_tex : bool, optional
        If True, use LaTeX for text rendering (requires LaTeX installation).
        Default is False.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Common settings
    plt.rcParams['figure.figsize'] = FIG_SIZES['medium']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Grid settings
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.3
    
    # Line settings
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    
    # LaTeX settings (if requested)
    if use_tex:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts,mathrsfs}'


def load_tier1_results(output_dir: str, station_code: str = None) -> Dict:
    """
    Load Tier 1 analysis results from JSON or Parquet files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing Tier 1 output files
    station_code : str, optional
        Specific station code to load. If None, load all stations.
        
    Returns
    -------
    Dict
        Dictionary of Tier 1 results, keyed by station code
    """
    results = {}
    
    if station_code:
        # Look for a specific station
        json_file = os.path.join(output_dir, f"{station_code}_tier1.json")
        parquet_file = os.path.join(output_dir, f"{station_code}_tier1.parquet")
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results[station_code] = json.load(f)
        elif os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            results[station_code] = df.to_dict(orient='records')[0]
        else:
            logger.warning(f"No Tier 1 results found for station {station_code}")
    else:
        # Load all stations
        json_files = glob.glob(os.path.join(output_dir, "*_tier1.json"))
        parquet_files = glob.glob(os.path.join(output_dir, "*_tier1.parquet"))
        
        # Process JSON files
        for json_file in json_files:
            station_code = os.path.basename(json_file).split('_tier1.json')[0]
            try:
                with open(json_file, 'r') as f:
                    results[station_code] = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        # Process Parquet files for stations not already loaded
        for parquet_file in parquet_files:
            station_code = os.path.basename(parquet_file).split('_tier1.parquet')[0]
            if station_code not in results:
                try:
                    df = pd.read_parquet(parquet_file)
                    results[station_code] = df.to_dict(orient='records')[0]
                except Exception as e:
                    logger.warning(f"Error loading {parquet_file}: {e}")
    
    return results


def load_tier2_results(output_dir: str, station_code: str = None) -> Dict:
    """
    Load Tier 2 analysis results from JSON or Parquet files.
    
    Parameters
    ----------
    output_dir : str
        Directory containing Tier 2 output files
    station_code : str, optional
        Specific station code to load. If None, load all stations.
        
    Returns
    -------
    Dict
        Dictionary of Tier 2 results, keyed by station code
    """
    results = {}
    
    if station_code:
        # Look for a specific station
        json_file = os.path.join(output_dir, f"{station_code}_tier2.json")
        full_json_file = os.path.join(output_dir, f"{station_code}_tier2_full.json")
        parquet_file = os.path.join(output_dir, f"{station_code}_tier2.parquet")
        
        if os.path.exists(full_json_file):
            with open(full_json_file, 'r') as f:
                results[station_code] = json.load(f)
        elif os.path.exists(json_file):
            with open(json_file, 'r') as f:
                results[station_code] = json.load(f)
        elif os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            results[station_code] = df.to_dict(orient='records')[0]
        else:
            logger.warning(f"No Tier 2 results found for station {station_code}")
    else:
        # Load all stations
        json_files = glob.glob(os.path.join(output_dir, "*_tier2.json")) + \
                    glob.glob(os.path.join(output_dir, "*_tier2_full.json"))
        parquet_files = glob.glob(os.path.join(output_dir, "*_tier2.parquet"))
        
        # Process JSON files
        for json_file in json_files:
            if "_full" in json_file:
                station_code = os.path.basename(json_file).split('_tier2_full.json')[0]
            else:
                station_code = os.path.basename(json_file).split('_tier2.json')[0]
                
            # Skip if we already loaded the full JSON for this station
            if station_code in results:
                continue
                
            try:
                with open(json_file, 'r') as f:
                    results[station_code] = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        # Process Parquet files for stations not already loaded
        for parquet_file in parquet_files:
            station_code = os.path.basename(parquet_file).split('_tier2.parquet')[0]
            if station_code not in results:
                try:
                    df = pd.read_parquet(parquet_file)
                    results[station_code] = df.to_dict(orient='records')[0]
                except Exception as e:
                    logger.warning(f"Error loading {parquet_file}: {e}")
    
    return results


def load_station_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load station metadata CSV containing geographic information.
    
    Parameters
    ----------
    metadata_path : str
        Path to metadata CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with station metadata
    """
    try:
        df = pd.read_csv(metadata_path)
        # Normalize station code column name
        if 'SITE CODE' in df.columns:
            df['station_code'] = df['SITE CODE'].astype(str)
        elif 'site_code' in df.columns:
            df['station_code'] = df['site_code'].astype(str)
        
        # Normalize lat/lon column names
        if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns:
            df['latitude'] = df['LATITUDE']
            df['longitude'] = df['LONGITUDE']
        
        return df
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return pd.DataFrame()


def create_output_dirs(base_dir: str = 'outputs/plots') -> Dict[str, str]:
    """
    Create output directories for saving plots.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for plot outputs
        
    Returns
    -------
    Dict[str, str]
        Dictionary of output directory paths
    """
    dirs = {
        'tier1': os.path.join(base_dir, 'tier1'),
        'tier1_extremes': os.path.join(base_dir, 'tier1', 'extremes'),
        'tier1_joint': os.path.join(base_dir, 'tier1', 'joint'),
        'tier1_stations': os.path.join(base_dir, 'tier1', 'stations'),
        'tier2': os.path.join(base_dir, 'tier2'),
        'tier2_copulas': os.path.join(base_dir, 'tier2', 'copulas'),
        'tier2_dependence': os.path.join(base_dir, 'tier2', 'dependence'),
        'tier2_joint_returns': os.path.join(base_dir, 'tier2', 'joint_returns'),
        'maps': os.path.join(base_dir, 'maps'),
        'events': os.path.join(base_dir, 'events'),
        'publication': os.path.join(base_dir, 'publication')
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs


def save_figure(fig, filename, dpi=300, bbox_inches='tight', **kwargs):
    """
    Save a figure with proper formatting and DPI.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename (if no extension, .png is added)
    dpi : int, optional
        Resolution (dots per inch)
    bbox_inches : str, optional
        Bounding box setting
    **kwargs : dict
        Additional parameters to pass to savefig
    """
    # Add .png extension if no extension provided
    if not os.path.splitext(filename)[1]:
        filename = f"{filename}.png"
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    logger.info(f"Figure saved to {filename}")


def generate_summary_stats(tier1_results: Dict, tier2_results: Dict) -> pd.DataFrame:
    """
    Generate summary statistics from Tier 1 and Tier 2 results for all stations.
    
    Parameters
    ----------
    tier1_results : Dict
        Dictionary of Tier 1 results
    tier2_results : Dict
        Dictionary of Tier 2 results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with summary statistics
    """
    summary_data = []
    
    for station_code in tier1_results.keys():
        station_data = {'station_code': station_code}
        
        # Extract Tier 1 data
        t1_data = tier1_results.get(station_code, {})
        if t1_data:
            # Sea level data
            if 'sea_level' in t1_data and 'gpd' in t1_data['sea_level']:
                sl_gpd = t1_data['sea_level']['gpd']
                station_data['sl_shape'] = sl_gpd.get('shape')
                station_data['sl_scale'] = sl_gpd.get('scale')
                station_data['sl_threshold'] = t1_data['sea_level'].get('threshold')
                station_data['sl_n_exceed'] = sl_gpd.get('n_exceed')
            
            # Precipitation data
            if 'total_precipitation' in t1_data and 'gpd' in t1_data['total_precipitation']:
                pr_gpd = t1_data['total_precipitation']['gpd']
                station_data['pr_shape'] = pr_gpd.get('shape')
                station_data['pr_scale'] = pr_gpd.get('scale')
                station_data['pr_threshold'] = t1_data['total_precipitation'].get('threshold')
                station_data['pr_n_exceed'] = pr_gpd.get('n_exceed')
            
            # Joint statistics
            if 'joint' in t1_data and 'empirical' in t1_data['joint']:
                joint = t1_data['joint']['empirical']
                station_data['joint_cpr'] = joint.get('cpr')
                station_data['joint_n_joint'] = joint.get('n_joint')
                station_data['p_joint'] = joint.get('p_joint')
                station_data['p_independent'] = joint.get('p_independent')
        
        # Extract Tier 2 data
        t2_data = tier2_results.get(station_code, {})
        if t2_data and 'tier2_analysis' in t2_data:
            analysis = t2_data['tier2_analysis']
            
            # Copula info
            if 'copula' in analysis:
                copula = analysis['copula']
                station_data['copula_method'] = copula.get('method')
                station_data['copula_aic'] = copula.get('aic')
                
                # Extract parameters
                params = copula.get('parameters', {})
                if 'theta' in params:
                    station_data['copula_theta'] = params.get('theta')
                if 'rho' in params:
                    station_data['copula_rho'] = params.get('rho')
                if 'df' in params:
                    station_data['copula_df'] = params.get('df')
            
            # Tail dependence
            if 'tail_dependence' in analysis:
                tail = analysis['tail_dependence']
                station_data['tail_lower'] = tail.get('lower')
                station_data['tail_upper'] = tail.get('upper')
            
            # Joint exceedance at 0.99 level
            if 'joint_exceedance' in analysis and '0.99' in analysis['joint_exceedance']:
                joint_99 = analysis['joint_exceedance']['0.99']
                station_data['joint_exc_99'] = joint_99.get('joint_exceedance')
                station_data['cpr_99'] = joint_99.get('cpr')
                
            # 100-year joint return period
            if 'joint_return_periods' in analysis and '100' in analysis['joint_return_periods']:
                rp_100 = analysis['joint_return_periods']['100']
                station_data['rp_100_and'] = rp_100.get('and_return_period')
                station_data['rp_100_or'] = rp_100.get('or_return_period')
        
        summary_data.append(station_data)
    
    return pd.DataFrame(summary_data)


def combine_with_metadata(summary_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine summary statistics with station metadata.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with summary statistics
    metadata_df : pd.DataFrame
        DataFrame with station metadata
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame
    """
    # Make sure station_code columns are strings for reliable merging
    summary_df['station_code'] = summary_df['station_code'].astype(str)
    metadata_df['station_code'] = metadata_df['station_code'].astype(str)
    
    # Merge the dataframes
    merged_df = pd.merge(summary_df, metadata_df, on='station_code', how='left')
    
    # Log merge statistics
    n_summary = len(summary_df)
    n_merged = sum(~merged_df['latitude'].isna())
    logger.info(f"Merged {n_merged} of {n_summary} stations with metadata")
    
    return merged_df


if __name__ == "__main__":
    # Basic test of the module
    print("Testing visualization base module...")
    
    # Set the style
    set_publication_style()
    
    # Create a simple plot
    plt.figure(figsize=FIG_SIZES['medium'])
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, label='Sine Wave')
    plt.title('Test Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # Create output directories
    dirs = create_output_dirs('outputs/plots_test')
    
    # Save the plot
    save_figure(plt.gcf(), os.path.join(dirs['tier1'], 'test_plot'))
    
    plt.close()
    
    print("Test complete. Check outputs/plots_test/tier1/test_plot.png")