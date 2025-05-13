#!/usr/bin/env python
"""
Script to generate Tier 3 Bayesian Network visualizations.

This script loads Tier 3 results and creates visualizations for the Bayesian Network analysis.
"""
import os
import argparse
import logging
from src.compound_flooding.visualization import (
    load_tier3_results,
    plot_bayesian_network,
    plot_cpt_heatmap,
    plot_sensitivity_analysis,
    plot_risk_distribution,
    plot_validation_metrics,
    plot_station_tier3_summary,
    create_tier3_summary_report,
    set_publication_style
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate Tier 3 visualizations')
    parser.add_argument('--tier3-dir', default='outputs/tier3',
                        help='Directory containing Tier 3 outputs')
    parser.add_argument('--output-dir', default='outputs/plots/tier3',
                        help='Directory to save visualizations')
    parser.add_argument('--station-code', default=None,
                        help='Specific station code to process (default: all)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots (default: save only)')
    
    args = parser.parse_args()
    
    # Set publication style
    set_publication_style()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Tier 3 results
    if args.station_code:
        # Load specific station
        tier3_data = load_tier3_results(args.tier3_dir, args.station_code)
        if not tier3_data:
            logger.error(f"No Tier 3 data found for station {args.station_code}")
            return 1
        
        # Extract station data
        station_data = tier3_data[args.station_code]
        
        # Create station-specific directory
        station_dir = os.path.join(args.output_dir, args.station_code)
        os.makedirs(station_dir, exist_ok=True)
        
        # Create all visualizations for the station
        logger.info(f"Creating visualizations for station {args.station_code}")
        
        # Network structure
        logger.info("Creating Bayesian Network structure plot")
        plot_bayesian_network(station_data, output_dir=station_dir, show=args.show)
        
        # Risk distribution
        logger.info("Creating risk distribution plot")
        plot_risk_distribution(station_data, output_dir=station_dir, show=args.show)
        
        # Sensitivity analysis
        logger.info("Creating sensitivity analysis plot")
        plot_sensitivity_analysis(station_data, output_dir=station_dir, show=args.show)
        
        # Validation metrics
        logger.info("Creating validation metrics plot")
        plot_validation_metrics(station_data, output_dir=station_dir, show=args.show)
        
        # CPT heatmaps
        logger.info("Creating CPT heatmaps")
        # Get the list of nodes
        bn_structure = station_data.get('tier3_analysis', {}).get(
            'bayesian_network', {}).get('structure', {})
        nodes = bn_structure.get('nodes', [])
        
        for node in nodes:
            logger.info(f"Creating CPT heatmap for node {node}")
            plot_cpt_heatmap(station_data, node=node, output_dir=station_dir, show=args.show)
        
        # Summary plot
        logger.info("Creating summary plot")
        plot_station_tier3_summary(station_data, output_dir=station_dir, show=args.show)
        
        logger.info(f"All visualizations created and saved to {station_dir}")
        
    else:
        # Load all stations
        logger.info("Loading all stations")
        tier3_data = load_tier3_results(args.tier3_dir)
        
        if not tier3_data:
            logger.error(f"No Tier 3 data found in {args.tier3_dir}")
            return 1
        
        # Create summary report for all stations
        logger.info(f"Creating summary report for {len(tier3_data)} stations")
        create_tier3_summary_report(tier3_data, args.output_dir, show=args.show)
        
        logger.info(f"All visualizations created and saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    main() 