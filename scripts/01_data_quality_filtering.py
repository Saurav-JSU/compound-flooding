# scripts/01_data_quality_filtering.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
from datetime import datetime

# Add the src directory to the Python path - this is critical
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix the import paths
from src.preprocessing.data_loader import load_metadata, get_station_files, extract_site_code, load_station_data, get_station_metadata
from src.preprocessing.quality_control import analyze_station_quality
from src.utils.parallel import process_in_parallel
from src.visualization.plotting import plot_quality_distribution, plot_quality_issues
from src.utils.file_utils import create_directory_structure

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'quality_filtering_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('quality_filtering')

def process_station(file_path):
    """Process a single station file and return quality assessment."""
    site_code = extract_site_code(file_path)
    logger.debug(f"Processing station {site_code}")
    
    # Get station metadata
    station_meta = get_station_metadata(metadata, site_code)
    
    if station_meta is None:
        logger.warning(f"No metadata found for station {site_code}")
        return None
    
    # Load station data
    df = load_station_data(file_path)
    
    if df is None:
        logger.warning(f"Failed to load data for station {site_code}")
        return None
        
    # Check if sea level column exists
    if 'sea_level' not in df.columns:
        logger.warning(f"Sea level column missing in station {site_code}")
        return None
    
    # Get null value from metadata
    null_value = station_meta.get('null_value', -99.9999)
    
    # Analyze station quality
    try:
        quality_assessment = analyze_station_quality(df, station_meta, null_value=null_value)
        return quality_assessment
    except Exception as e:
        logger.error(f"Error analyzing station {site_code}: {e}")
        return None

def main():
    """Main function to run the quality control process."""
    logger.info("Starting data quality filtering process")
    
    # Create directory structure if it doesn't exist
    create_directory_structure()
    
    # Load metadata
    global metadata
    metadata_path = os.path.join('compound_flooding', 'data', 'GESLA', 'usa_metadata.csv')
    metadata = load_metadata(metadata_path)
    
    # Get station files
    data_dir = os.path.join('compound_flooding', 'GESLA_ERA5_with_sea_level')
    station_files = get_station_files(data_dir)
    
    logger.info(f"Analyzing quality for {len(station_files)} stations")
    
    # Process stations in parallel
    start_time = time.time()
    results = process_in_parallel(
        station_files, 
        process_station, 
        max_workers=96,  # Use all available cores on your system
        desc="Analyzing station quality"
    )
    
    # Filter out None results and convert to DataFrame
    quality_results = [r for r in results if r is not None]
    quality_df = pd.DataFrame(quality_results)
    
    end_time = time.time()
    logger.info(f"Quality analysis completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully analyzed {len(quality_df)} out of {len(station_files)} stations")
    
    # Log quality tier distribution
    tier_counts = quality_df['quality_tier'].value_counts()
    logger.info("Quality tier distribution:")
    for tier, count in tier_counts.sort_index().items():
        logger.info(f"  Tier {tier}: {count} stations ({count/len(quality_df)*100:.1f}%)")
    
    # Save results
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full quality assessment
    quality_df.to_csv(os.path.join(output_dir, 'station_quality_assessment.csv'), index=False)
    logger.info(f"Full quality assessment saved to {os.path.join(output_dir, 'station_quality_assessment.csv')}")
    
    # Create filtered datasets for each quality tier
    for tier in ['A', 'B', 'C']:
        tier_df = quality_df[quality_df['quality_tier'] == tier]
        tier_df.to_csv(os.path.join(output_dir, f'tier_{tier}_stations.csv'), index=False)
        logger.info(f"Tier {tier} stations ({len(tier_df)}) saved to {os.path.join(output_dir, f'tier_{tier}_stations.csv')}")
    
    # Create visualizations
    logger.info("Creating quality visualization plots")
    plot_output_dir = os.path.join('plots', 'quality')
    
    try:
        plot_quality_distribution(quality_df, output_dir=plot_output_dir)
        plot_quality_issues(quality_df, output_dir=plot_output_dir)
        logger.info(f"Quality plots saved to {plot_output_dir}")
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
    
    logger.info("Data quality filtering process completed")

if __name__ == "__main__":
    main()