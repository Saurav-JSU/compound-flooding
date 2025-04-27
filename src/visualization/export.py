"""
Export module for compound flooding visualizations.

This module provides utilities for creating publication-ready figures with:
- Consistent formatting
- High-resolution export
- Multiple format support (PNG, PDF, SVG, etc.)
- Multi-panel figure creation
- LaTeX support for publication-quality text
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import base visualization utilities
from compound_flooding.visualization.base import (
    FIG_SIZES, set_publication_style, save_figure,
    load_tier1_results, load_tier2_results, load_station_metadata
)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_multi_panel_figure(
    panels: List[Dict],
    suptitle: str = None,
    figsize: Tuple[float, float] = (12, 10),
    grid_spec: Dict = None,
    output_path: str = None,
    dpi: int = 300,
    formats: List[str] = ['png', 'pdf'],
    use_tex: bool = False
) -> plt.Figure:
    """
    Create a multi-panel figure with custom layout.
    
    Parameters
    ----------
    panels : List[Dict]
        List of panel configurations:
        [{'function': plot_func, 'args': (arg1, arg2), 'kwargs': {'key': value}, 
          'grid_pos': (row, col, row_span, col_span)}, ...]
    suptitle : str, optional
        Super title for the entire figure
    figsize : Tuple[float, float], optional
        Figure size in inches
    grid_spec : Dict, optional
        GridSpec parameters: {'nrows': n, 'ncols': m, 'height_ratios': [...], ...}
    output_path : str, optional
        Output path (without extension)
    dpi : int, optional
        Resolution for raster formats
    formats : List[str], optional
        Output formats to save
    use_tex : bool, optional
        Whether to use LaTeX for text rendering
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Set publication style with LaTeX support if requested
    set_publication_style(use_tex=use_tex)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Setup GridSpec for custom layout
    if grid_spec is None:
        grid_spec = {'nrows': 2, 'ncols': 2}
    
    gs = gridspec.GridSpec(figure=fig, **grid_spec)
    
    # Create each panel
    for panel in panels:
        # Extract panel configuration
        plot_func = panel['function']
        args = panel.get('args', ())
        kwargs = panel.get('kwargs', {})
        grid_pos = panel.get('grid_pos', (0, 0, 1, 1))
        
        # Create subplot
        if isinstance(grid_pos, tuple) and len(grid_pos) == 4:
            row, col, row_span, col_span = grid_pos
            ax = fig.add_subplot(gs[row:row+row_span, col:col+col_span])
        else:
            ax = fig.add_subplot(gs[grid_pos])
        
        # Call plotting function with the axis
        kwargs['ax'] = ax
        plot_func(*args, **kwargs)
    
    # Add super title if provided
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
        fig.subplots_adjust(top=0.9)  # Make room for suptitle
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure in multiple formats if output path is provided
    if output_path:
        for fmt in formats:
            filename = f"{output_path}.{fmt}"
            save_figure(fig, filename, dpi=dpi)
            logger.info(f"Saved figure to {filename}")
    
    return fig


def create_panel_function(
    func: Callable,
    *args,
    **kwargs
) -> Callable:
    """
    Create a panel function that can be used with create_multi_panel_figure.
    
    This wraps an existing plotting function to make it compatible with the
    multi-panel figure creation process.
    
    Parameters
    ----------
    func : Callable
        Original plotting function
    *args, **kwargs
        Arguments and keyword arguments to pass to the function
        
    Returns
    -------
    Callable
        Panel function with ax parameter
    """
    def panel_func(ax=None, **panel_kwargs):
        # Override original kwargs with panel-specific kwargs
        merged_kwargs = {**kwargs, **panel_kwargs}
        
        # If ax is provided, add it to kwargs
        if ax is not None:
            merged_kwargs['ax'] = ax
            
        # Call the original function
        return func(*args, **merged_kwargs)
    
    return panel_func


def create_regional_comparison_figure(
    tier1_data: Dict,
    tier2_data: Dict,
    metadata_df: pd.DataFrame,
    region_stations: Dict[str, List[str]],
    output_path: str = None,
    dpi: int = 300,
    formats: List[str] = ['png', 'pdf']
) -> plt.Figure:
    """
    Create a figure comparing different regions.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    metadata_df : pd.DataFrame
        DataFrame with station metadata
    region_stations : Dict[str, List[str]]
        Dictionary mapping region names to lists of station codes
    output_path : str, optional
        Output path (without extension)
    dpi : int, optional
        Resolution for raster formats
    formats : List[str], optional
        Output formats to save
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Set publication style
    set_publication_style()
    
    # Extract the main metrics for comparison
    comparison_df = {
        'station_code': [],
        'region': [],
        'shape_sl': [],
        'shape_pr': [],
        'cpr': [],
        'tau': [],
        'tail_lower': [],
        'tail_upper': [],
        'sl_rl100': [],
        'pr_rl100': []
    }
    
    # Process each region and station
    for region, stations in region_stations.items():
        for station in stations:
            # Skip if station not in data
            if station not in tier1_data or station not in tier2_data:
                continue
                
            # Add station base info
            comparison_df['station_code'].append(station)
            comparison_df['region'].append(region)
            
            # Extract Tier 1 metrics
            t1_data = tier1_data[station]
            
            # Shape parameters
            shape_sl = np.nan
            shape_pr = np.nan
            if 'sea_level' in t1_data and 'gpd' in t1_data['sea_level']:
                shape_sl = t1_data['sea_level']['gpd'].get('shape', np.nan)
            if 'total_precipitation' in t1_data and 'gpd' in t1_data['total_precipitation']:
                shape_pr = t1_data['total_precipitation']['gpd'].get('shape', np.nan)
                
            comparison_df['shape_sl'].append(shape_sl)
            comparison_df['shape_pr'].append(shape_pr)
            
            # Return levels
            sl_rl100 = np.nan
            pr_rl100 = np.nan
            
            if 'sea_level' in t1_data and 'return_levels' in t1_data['sea_level']:
                rl_data = t1_data['sea_level']['return_levels']
                if isinstance(rl_data, list):
                    for rl in rl_data:
                        if isinstance(rl, dict) and rl.get('return_period') == 100:
                            sl_rl100 = rl.get('return_level', np.nan)
                            break
                            
            if 'total_precipitation' in t1_data and 'return_levels' in t1_data['total_precipitation']:
                rl_data = t1_data['total_precipitation']['return_levels']
                if isinstance(rl_data, list):
                    for rl in rl_data:
                        if isinstance(rl, dict) and rl.get('return_period') == 100:
                            pr_rl100 = rl.get('return_level', np.nan)
                            break
            
            comparison_df['sl_rl100'].append(sl_rl100)
            comparison_df['pr_rl100'].append(pr_rl100)
            
            # CPR
            cpr = np.nan
            if 'joint' in t1_data and 'empirical' in t1_data['joint']:
                cpr = t1_data['joint']['empirical'].get('cpr', np.nan)
            comparison_df['cpr'].append(cpr)
            
            # Extract Tier 2 metrics
            t2_data = tier2_data[station]
            
            # Tau
            tau = np.nan
            if ('tier2_analysis' in t2_data and 
                'copula' in t2_data['tier2_analysis'] and
                'parameters' in t2_data['tier2_analysis']['copula']):
                
                params = t2_data['tier2_analysis']['copula']['parameters']
                method = t2_data['tier2_analysis']['copula'].get('method', '')
                
                if method in ['Gaussian', 'StudentT'] and 'rho' in params:
                    rho = params['rho']
                    # Convert to tau using sin formula
                    tau = 2 * np.arcsin(rho) / np.pi
                elif method == 'Gumbel' and 'theta' in params:
                    theta = params['theta']
                    # Tau = (theta-1)/theta for Gumbel
                    tau = max(0, (theta - 1) / theta)
                elif method == 'Frank' and 'theta' in params:
                    # For Frank, use Spearman's rho as proxy (approximate)
                    theta = params['theta']
                    if abs(theta) < 1e-6:
                        tau = 0
                    else:
                        # Very rough approximation
                        tau = np.sign(theta) * min(0.8, abs(theta) / 10)
            
            comparison_df['tau'].append(tau)
            
            # Tail dependence
            tail_lower = np.nan
            tail_upper = np.nan
            
            if ('tier2_analysis' in t2_data and 
                'tail_dependence' in t2_data['tier2_analysis']):
                tail_dep = t2_data['tier2_analysis']['tail_dependence']
                tail_lower = tail_dep.get('lower', np.nan)
                tail_upper = tail_dep.get('upper', np.nan)
                
            comparison_df['tail_lower'].append(tail_lower)
            comparison_df['tail_upper'].append(tail_upper)
    
    # Convert to DataFrame
    df = pd.DataFrame(comparison_df)
    
    # Create figure with 2x3 panels
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Define the metrics to plot
    metrics = [
        {'name': 'CPR', 'key': 'cpr', 'grid_pos': (0, 0), 
         'ylabel': 'Conditional Probability Ratio', 'ylim': [0, 5]},
        {'name': 'Kendall\'s Tau', 'key': 'tau', 'grid_pos': (0, 1), 
         'ylabel': 'Kendall\'s τ', 'ylim': [-0.2, 1]},
        {'name': 'Sea Level Shape Parameter', 'key': 'shape_sl', 'grid_pos': (0, 2), 
         'ylabel': 'Shape Parameter (ξ)', 'ylim': [-0.5, 0.5]},
        {'name': 'Upper Tail Dependence', 'key': 'tail_upper', 'grid_pos': (1, 0), 
         'ylabel': 'Upper Tail Dependence', 'ylim': [0, 1]},
        {'name': 'Lower Tail Dependence', 'key': 'tail_lower', 'grid_pos': (1, 1), 
         'ylabel': 'Lower Tail Dependence', 'ylim': [0, 1]},
        {'name': '100-year Sea Level', 'key': 'sl_rl100', 'grid_pos': (1, 2), 
         'ylabel': '100-year Return Level (m)', 'ylim': None}
    ]
    
    # Create region-colored box plots for each metric
    for metric in metrics:
        # Create subplot
        ax = fig.add_subplot(gs[metric['grid_pos']])
        
        # Create box plot
        sns.boxplot(x='region', y=metric['key'], data=df, ax=ax)
        
        # Add individual points
        sns.stripplot(x='region', y=metric['key'], data=df, 
                     color='black', alpha=0.5, jitter=True, ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Region')
        ax.set_ylabel(metric['ylabel'])
        ax.set_title(metric['name'])
        
        # Set y-limits if specified
        if metric['ylim']:
            ax.set_ylim(metric['ylim'])
        
        # Set x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add reference lines for special cases
        if metric['key'] == 'cpr':
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, 
                      label='Independence')
            ax.legend()
        elif metric['key'] == 'tau':
            ax.axhline(0.0, color='red', linestyle='--', alpha=0.7,
                      label='Independence')
            ax.legend()
        elif metric['key'] == 'shape_sl':
            ax.axhline(0.0, color='red', linestyle='--', alpha=0.7,
                      label='Gumbel')
            ax.legend()
    
    # Add super title
    fig.suptitle('Regional Comparison of Compound Flooding Characteristics', fontsize=16)
    fig.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    
    # Save figure in multiple formats if output path is provided
    if output_path:
        for fmt in formats:
            filename = f"{output_path}.{fmt}"
            save_figure(fig, filename, dpi=dpi)
            logger.info(f"Saved figure to {filename}")
    
    return fig


def create_publication_figure(
    tier1_data: Dict,
    tier2_data: Dict,
    metadata_df: pd.DataFrame,
    figure_type: str,
    station_codes: List[str] = None,
    output_path: str = None,
    dpi: int = 300,
    formats: List[str] = ['png', 'pdf'],
    use_tex: bool = False
) -> plt.Figure:
    """
    Create a publication-ready figure of the specified type.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    metadata_df : pd.DataFrame
        DataFrame with station metadata
    figure_type : str
        Type of figure to create: 'regional_dependence', 'compound_risk_map',
        'copula_comparison', 'return_period_map'
    station_codes : List[str], optional
        List of station codes to include. If None, use all stations.
    output_path : str, optional
        Output path (without extension)
    dpi : int, optional
        Resolution for raster formats
    formats : List[str], optional
        Output formats to save
    use_tex : bool, optional
        Whether to use LaTeX for text rendering
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Set publication style with LaTeX if requested
    set_publication_style(use_tex=use_tex)
    
    # Import required visualization modules on demand
    if figure_type in ['compound_risk_map', 'return_period_map']:
        from compound_flooding.visualization.maps import (
            plot_compound_flood_risk_map,
            plot_tier2_parameter_map,
            plot_tier1_parameter_map
        )
    
    if figure_type in ['regional_dependence', 'copula_comparison']:
        from compound_flooding.visualization.tier2_plots import (
            plot_copula_density,
            plot_tail_dependence,
            plot_joint_return_periods,
            plot_conditional_exceedance
        )
        
        from compound_flooding.visualization.tier1_plots import (
            plot_joint_exceedance,
            plot_gpd_diagnostics
        )
    
    # Filter stations if provided
    if station_codes:
        tier1_filtered = {k: v for k, v in tier1_data.items() if k in station_codes}
        tier2_filtered = {k: v for k, v in tier2_data.items() if k in station_codes}
    else:
        tier1_filtered = tier1_data
        tier2_filtered = tier2_data
        
    # Create figure based on type
    if figure_type == 'regional_dependence':
        # Define regions (example - you would customize these)
        regions = {
            'East Coast': [s for s, m in metadata_df.iterrows() 
                          if -85 < m.get('longitude', -999) < -65 and 25 < m.get('latitude', -999) < 45],
            'Gulf Coast': [s for s, m in metadata_df.iterrows() 
                          if -98 < m.get('longitude', -999) < -80 and 24 < m.get('latitude', -999) < 31],
            'West Coast': [s for s, m in metadata_df.iterrows() 
                          if -125 < m.get('longitude', -999) < -115 and 32 < m.get('latitude', -999) < 49]
        }
        
        # Create regional comparison figure
        fig = create_regional_comparison_figure(
            tier1_data=tier1_filtered,
            tier2_data=tier2_filtered,
            metadata_df=metadata_df,
            region_stations=regions,
            output_path=output_path,
            dpi=dpi,
            formats=formats
        )
        
    elif figure_type == 'compound_risk_map':
        # Create compound risk map
        fig = plot_compound_flood_risk_map(
            tier1_data=tier1_filtered,
            tier2_data=tier2_filtered,
            metadata_df=metadata_df,
            region='usa',
            title="Compound Flood Risk in the United States",
            output_dir=os.path.dirname(output_path) if output_path else None,
            show=True
        )
        
        # Save in requested formats
        if output_path:
            for fmt in formats:
                filename = f"{output_path}.{fmt}"
                save_figure(fig, filename, dpi=dpi)
                logger.info(f"Saved figure to {filename}")
        
    elif figure_type == 'copula_comparison':
        # Select representative stations
        # This is an example approach - in practice you might select specific stations
        if not station_codes or len(station_codes) < 3:
            # Find stations with diverse copula types
            copula_types = {}
            for station, data in tier2_filtered.items():
                if ('tier2_analysis' in data and 
                    'copula' in data['tier2_analysis']):
                    method = data['tier2_analysis']['copula'].get('method')
                    if method:
                        if method not in copula_types:
                            copula_types[method] = []
                        copula_types[method].append(station)
            
            # Select one station per copula type (up to 3)
            selected_stations = []
            for method, stations in copula_types.items():
                if stations and len(selected_stations) < 3:
                    selected_stations.append(stations[0])
            
            # If we still don't have 3 stations, use whatever we have
            if len(selected_stations) < 3 and tier2_filtered:
                additional = [s for s in tier2_filtered.keys() 
                             if s not in selected_stations]
                selected_stations.extend(additional[:3-len(selected_stations)])
        else:
            # Use provided station codes (up to 3)
            selected_stations = station_codes[:3]
        
        # Create multi-panel figure with copula visualizations
        panels = []
        for i, station in enumerate(selected_stations):
            if i >= 3:  # Limit to 3 stations
                break
                
            if station in tier2_filtered:
                # Add copula density panel
                panels.append({
                    'function': plot_copula_density,
                    'args': (tier2_filtered[station],),
                    'kwargs': {'show': False},
                    'grid_pos': (0, i, 1, 1)
                })
                
                # Add joint return period panel
                panels.append({
                    'function': plot_joint_return_periods,
                    'args': (tier2_filtered[station],),
                    'kwargs': {'show': False},
                    'grid_pos': (1, i, 1, 1)
                })
        
        # Create figure
        grid_spec = {'nrows': 2, 'ncols': 3, 'height_ratios': [1, 1], 'width_ratios': [1, 1, 1]}
        
        fig = create_multi_panel_figure(
            panels=panels,
            suptitle="Copula Comparison Across Stations",
            figsize=(15, 10),
            grid_spec=grid_spec,
            output_path=output_path,
            dpi=dpi,
            formats=formats,
            use_tex=use_tex
        )
        
    elif figure_type == 'return_period_map':
        # Create return period map for 100-year sea level
        fig = plot_tier1_parameter_map(
            tier1_data=tier1_filtered,
            metadata_df=metadata_df,
            parameter='return_level_100',
            variable='sea_level',
            region='usa',
            title="100-year Sea Level Return Level",
            output_dir=os.path.dirname(output_path) if output_path else None,
            show=True
        )
        
        # Save in requested formats
        if output_path:
            for fmt in formats:
                filename = f"{output_path}.{fmt}"
                save_figure(fig, filename, dpi=dpi)
                logger.info(f"Saved figure to {filename}")
    
    else:
        # Invalid figure type
        logger.error(f"Invalid figure type: {figure_type}")
        fig = plt.figure()
        plt.text(0.5, 0.5, f"Invalid figure type: {figure_type}",
                ha='center', va='center')
    
    return fig


def create_all_publication_figures(
    tier1_data: Dict,
    tier2_data: Dict,
    metadata_df: pd.DataFrame,
    output_dir: str,
    use_tex: bool = False
) -> None:
    """
    Create all standard publication figures and save them.
    
    Parameters
    ----------
    tier1_data : Dict
        Dictionary mapping station codes to Tier 1 results
    tier2_data : Dict
        Dictionary mapping station codes to Tier 2 results
    metadata_df : pd.DataFrame
        DataFrame with station metadata
    output_dir : str
        Output directory
    use_tex : bool, optional
        Whether to use LaTeX for text rendering
    """
    # Set publication style with LaTeX if requested
    set_publication_style(use_tex=use_tex)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the figures to create
    figures = [
        'regional_dependence',
        'compound_risk_map',
        'copula_comparison',
        'return_period_map'
    ]
    
    # Create each figure
    for figure_type in figures:
        logger.info(f"Creating {figure_type} figure...")
        
        output_path = os.path.join(output_dir, f"publication_{figure_type}")
        
        create_publication_figure(
            tier1_data=tier1_data,
            tier2_data=tier2_data,
            metadata_df=metadata_df,
            figure_type=figure_type,
            output_path=output_path,
            use_tex=use_tex
        )
    
    logger.info(f"All publication figures created and saved to {output_dir}")


if __name__ == "__main__":
    # Basic test of the module
    import sys
    from compound_flooding.visualization.base import (
        load_tier1_results, 
        load_tier2_results, 
        load_station_metadata,
        create_output_dirs
    )
    
    print("Testing export module...")
    
    # Create output directories
    dirs = create_output_dirs('outputs/plots_test')
    
    # Check if we have required arguments
    if len(sys.argv) > 3:
        tier1_dir = sys.argv[1]
        tier2_dir = sys.argv[2]
        metadata_file = sys.argv[3]
        
        try:
            # Load data
            tier1_data = load_tier1_results(tier1_dir)
            tier2_data = load_tier2_results(tier2_dir)
            metadata_df = load_station_metadata(metadata_file)
            
            print(f"Loaded Tier 1 data for {len(tier1_data)} stations")
            print(f"Loaded Tier 2 data for {len(tier2_data)} stations")
            print(f"Loaded metadata for {len(metadata_df)} stations")
            
            # Test creating a publication figure
            if tier1_data and tier2_data and len(metadata_df) > 0:
                # Test with a simple figure type
                figure_type = 'compound_risk_map'
                
                output_path = os.path.join(dirs['publication'], f"test_{figure_type}")
                
                create_publication_figure(
                    tier1_data=tier1_data,
                    tier2_data=tier2_data,
                    metadata_df=metadata_df,
                    figure_type=figure_type,
                    output_path=output_path
                )
                
                print(f"Test complete. Check output in {dirs['publication']}")
            else:
                print("Insufficient data for testing")
                
        except Exception as e:
            print(f"Error testing with provided files: {e}")
    else:
        print("No test data provided. Please provide Tier 1 directory, Tier 2 directory, and metadata file as arguments.")