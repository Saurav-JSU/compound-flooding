"""
Visualization package for compound flooding analysis.

This package provides a comprehensive set of visualization tools for analyzing
and presenting results from compound flooding analysis. It includes:

Modules:
- base: Base visualization utilities and common settings
- tier1_plots: Tier 1 specific visualizations (extreme values, joint exceedances)
- tier2_plots: Tier 2 specific visualizations (copulas, tail dependence)
- maps: Spatial visualization across stations
- events: Event-specific visualization (hurricanes, storms)
- export: Publication-ready figure export utilities
"""

# Import base utilities
from compound_flooding.visualization.base import (
    set_publication_style,
    save_figure,
    FIG_SIZES,
    load_tier1_results,
    load_tier2_results,
    load_station_metadata,
    combine_with_metadata,
    create_output_dirs,
    generate_summary_stats
)

# Import Tier 1 visualization functions
from compound_flooding.visualization.tier1_plots import (
    plot_gpd_diagnostics,
    plot_joint_exceedance,
    plot_cpr_heatmap,
    plot_lag_dependency,
    plot_station_tier1_summary,
    plot_bootstrap_comparison,
    create_tier1_summary_report
)

# Import Tier 2 visualization functions
from compound_flooding.visualization.tier2_plots import (
    plot_copula_density,
    plot_joint_return_periods,
    plot_conditional_exceedance,
    plot_tail_dependence,
    plot_station_tier2_summary,
    plot_tau_vs_cpr,
    create_tier2_summary_report
)

# Import map visualization functions
from compound_flooding.visualization.maps import (
    create_base_map,
    plot_station_map,
    plot_tier1_parameter_map,
    plot_tier2_parameter_map,
    plot_compound_flood_risk_map,
    create_spatial_visualizations
)

# Import event visualization functions
from compound_flooding.visualization.events import (
    load_event_data,
    plot_event_timeseries,
    plot_multi_station_event,
    plot_multi_event_comparison,
    analyze_hurricane_event,
    compare_multiple_events
)

# Import export utilities
from compound_flooding.visualization.export import (
    create_multi_panel_figure,
    create_panel_function,
    create_regional_comparison_figure,
    create_publication_figure,
    create_all_publication_figures
)

# Define __all__ for explicit import control
__all__ = [
    # Base utilities
    'set_publication_style', 'save_figure', 'FIG_SIZES',
    'load_tier1_results', 'load_tier2_results', 'load_station_metadata',
    'combine_with_metadata', 'create_output_dirs', 'generate_summary_stats',
    
    # Tier 1 visualizations
    'plot_gpd_diagnostics', 'plot_joint_exceedance', 'plot_cpr_heatmap',
    'plot_lag_dependency', 'plot_station_tier1_summary', 'plot_bootstrap_comparison',
    'create_tier1_summary_report',
    
    # Tier 2 visualizations
    'plot_copula_density', 'plot_joint_return_periods', 'plot_conditional_exceedance',
    'plot_tail_dependence', 'plot_station_tier2_summary', 'plot_tau_vs_cpr',
    'create_tier2_summary_report',
    
    # Map visualizations
    'create_base_map', 'plot_station_map', 'plot_tier1_parameter_map',
    'plot_tier2_parameter_map', 'plot_compound_flood_risk_map',
    'create_spatial_visualizations',
    
    # Event visualizations
    'load_event_data', 'plot_event_timeseries', 'plot_multi_station_event',
    'plot_multi_event_comparison', 'analyze_hurricane_event', 'compare_multiple_events',
    
    # Export utilities
    'create_multi_panel_figure', 'create_panel_function',
    'create_regional_comparison_figure', 'create_publication_figure',
    'create_all_publication_figures'
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'Compound Flooding Research Team'