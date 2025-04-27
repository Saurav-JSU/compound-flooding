#!/usr/bin/env python
"""
Demonstration script for compound flooding visualization.

This script shows how to use the visualization package to create a variety
of visualizations from Tier 1 and Tier 2 analysis results.

Usage:
    python visualize_compound_flooding.py --tier1-dir TIER1_DIR --tier2-dir TIER2_DIR --metadata METADATA_FILE --netcdf-dir NETCDF_DIR --output-dir OUTPUT_DIR

Optional Arguments:
    --region REGION           Region for maps ('usa', 'east_coast', 'gulf_coast', 'west_coast', 'world')
    --stations STATION [...]  Specific station codes to analyze
    --event-name EVENT_NAME   Name of event to analyze (requires --event-dates)
    --event-dates START END   Start and end dates of event in ISO format (YYYY-MM-DD)
    --publication             Generate publication-ready figures
    --use-tex                 Use LaTeX for text rendering in figures
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Import visualization package
from src.compound_flooding.visualization import (
    # Base utilities
    set_publication_style, create_output_dirs, 
    load_tier1_results, load_tier2_results, load_station_metadata,
    
    # Tier 1 and Tier 2 reports
    create_tier1_summary_report, create_tier2_summary_report,
    
    # Maps and spatial visualizations
    create_spatial_visualizations,
    
    # Event analysis
    analyze_hurricane_event, compare_multiple_events,
    
    # Publication figures
    create_all_publication_figures
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Create visualizations for compound flooding analysis')
    
    # Required arguments
    parser.add_argument('--tier1-dir', required=True, help='Directory with Tier 1 results')
    parser.add_argument('--tier2-dir', required=True, help='Directory with Tier 2 results')
    parser.add_argument('--metadata', required=True, help='Station metadata CSV file')
    parser.add_argument('--netcdf-dir', required=True, help='Directory with cleaned NetCDF files')
    parser.add_argument('--output-dir', required=True, help='Directory for output plots')
    
    # Optional arguments
    parser.add_argument('--region', default='usa', 
                       choices=['usa', 'east_coast', 'gulf_coast', 'west_coast', 'world'],
                       help='Region for map visualizations')
    parser.add_argument('--stations', nargs='+', help='Specific station codes to analyze')
    parser.add_argument('--event-name', help='Name of event to analyze')
    parser.add_argument('--event-dates', nargs=2, metavar=('START', 'END'),
                       help='Start and end dates of event in ISO format (YYYY-MM-DD)')
    parser.add_argument('--publication', action='store_true',
                       help='Generate publication-ready figures')
    parser.add_argument('--use-tex', action='store_true',
                       help='Use LaTeX for text rendering (requires LaTeX installation)')
    
    args = parser.parse_args()
    
    # Set publication style
    set_publication_style(use_tex=args.use_tex)
    
    # Create output directories
    print(f"Creating output directories in {args.output_dir}")
    dirs = create_output_dirs(args.output_dir)
    
    # Load data
    print(f"Loading Tier 1 data from {args.tier1_dir}")
    tier1_data = load_tier1_results(args.tier1_dir)
    
    print(f"Loading Tier 2 data from {args.tier2_dir}")
    tier2_data = load_tier2_results(args.tier2_dir)
    
    print(f"Loading station metadata from {args.metadata}")
    metadata_df = load_station_metadata(args.metadata)
    
    print(f"Loaded Tier 1 data for {len(tier1_data)} stations")
    print(f"Loaded Tier 2 data for {len(tier2_data)} stations")
    print(f"Loaded metadata for {len(metadata_df)} stations")
    
    # Create Tier 1 visualizations
    print("Generating Tier 1 visualizations...")
    create_tier1_summary_report(
        tier1_data=tier1_data,
        output_dir=dirs['tier1'],
        station_codes=args.stations
    )
    
    # Create Tier 2 visualizations
    print("Generating Tier 2 visualizations...")
    create_tier2_summary_report(
        tier1_data=tier1_data,
        tier2_data=tier2_data,
        output_dir=dirs['tier2'],
        station_codes=args.stations
    )
    
    # Create map visualizations
    print(f"Generating spatial visualizations for region: {args.region}")
    create_spatial_visualizations(
        tier1_data=tier1_data,
        tier2_data=tier2_data,
        metadata_df=metadata_df,
        output_dir=dirs['maps'],
        regions=[args.region]
    )
    
    # Analyze event if specified
    if args.event_name and args.event_dates:
        print(f"Analyzing event: {args.event_name} ({args.event_dates[0]} to {args.event_dates[1]})")
        analyze_hurricane_event(
            netcdf_dir=args.netcdf_dir,
            event_dates=tuple(args.event_dates),
            event_name=args.event_name,
            tier1_data=tier1_data,
            output_dir=dirs['events'],
            station_codes=args.stations
        )
    
    # Create publication-ready figures if requested
    if args.publication:
        print("Generating publication-ready figures...")
        create_all_publication_figures(
            tier1_data=tier1_data,
            tier2_data=tier2_data,
            metadata_df=metadata_df,
            output_dir=dirs['publication'],
            use_tex=args.use_tex
        )
    
    print(f"All visualizations completed and saved to {args.output_dir}")

if __name__ == '__main__':
    main()