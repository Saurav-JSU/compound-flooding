"""
GESLA-ERA5 Data Integrator

This script adds sea level data from GESLA to ERA5 meteorological data files.
It extracts station codes from ERA5 filenames, finds corresponding GESLA data,
and merges them based on timestamps.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from gesla import GeslaDataset
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gesla_era5_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GESLA-ERA5-Integrator")

class GeslaEra5Integrator:
    """
    A class to integrate GESLA sea level data with ERA5 meteorological data.
    """
    
    def __init__(self, base_path):
        """
        Initialize the integrator with paths.
        
        Args:
            base_path (str): Base path to the compound_flooding directory
        """
        # Convert to absolute path
        self.base_path = os.path.abspath(base_path)
        logger.info(f"Base path: {self.base_path}")
        
        # Find compound_flooding directory if needed
        if os.path.basename(self.base_path) != 'compound_flooding':
            compound_path = os.path.join(self.base_path, 'compound_flooding')
            if os.path.exists(compound_path):
                self.base_path = compound_path
                logger.info(f"Found compound_flooding directory: {self.base_path}")
        
        # Define paths
        self.gesla_path = os.path.join(self.base_path, 'data', 'GESLA')
        self.gesla_data_path = os.path.join(self.gesla_path, 'GESLA3.0_ALL')
        self.gesla_meta_path = os.path.join(self.gesla_path, 'metadata.csv')
        self.era5_path = os.path.join(self.base_path, 'GESLA_ERA5')
        
        # Define output path
        self.output_path = os.path.join(self.base_path, 'GESLA_ERA5_with_sea_level')
        
        # Verify paths exist
        self._verify_paths()
        
        # Initialize GESLA dataset
        self.gesla_ds = GeslaDataset(
            meta_file=str(self.gesla_meta_path),
            data_path=str(self.gesla_data_path) + '/'
        )
    
    def _verify_paths(self):
        """Verify that all required paths exist and log their status."""
        paths = {
            "GESLA path": self.gesla_path,
            "GESLA data path": self.gesla_data_path,
            "GESLA metadata path": self.gesla_meta_path,
            "ERA5 path": self.era5_path
        }
        
        for name, path in paths.items():
            exists = os.path.exists(path)
            logger.info(f"{name} exists: {exists} - {path}")
            if not exists:
                logger.warning(f"Required path not found: {path}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            logger.info(f"Created output directory: {self.output_path}")
    
    def extract_station_code(self, filename):
        """
        Extract station code from ERA5 filename.
        
        Args:
            filename (str): ERA5 filename (e.g., '240A_ERA5.csv')
        
        Returns:
            str: Station code
        """
        # Extract the part before '_ERA5.csv'
        match = re.match(r'(.+)_ERA5\.csv$', filename)
        if match:
            return match.group(1)
        return None
    
    def process_file(self, era5_file):
        """
        Process a single ERA5 file to add sea level data.
        
        Args:
            era5_file (str): Path to ERA5 CSV file
        
        Returns:
            bool: True if processed successfully, False otherwise
        """
        try:
            # Extract filename and station code
            filename = os.path.basename(era5_file)
            station_code = self.extract_station_code(filename)
            
            if not station_code:
                logger.warning(f"Could not extract station code from filename: {filename}")
                return False
            
            logger.info(f"Processing file: {filename} (Station Code: {station_code})")
            
            # Find matching GESLA metadata
            station_meta = self.gesla_ds.meta[self.gesla_ds.meta['site_code'] == station_code]
            
            if station_meta.empty:
                logger.warning(f"No GESLA metadata found for station code: {station_code}")
                return False
            
            # Get GESLA filename
            gesla_filename = station_meta['filename'].iloc[0]
            logger.info(f"Found GESLA file: {gesla_filename}")
            
            # Load GESLA sea level data
            sea_level_data, _ = self.gesla_ds.file_to_pandas(gesla_filename)
            
            # Load ERA5 data
            era5_data = pd.read_csv(era5_file)
            
            # Process timestamp from system:index column
            era5_data['datetime'] = pd.to_datetime(era5_data['system:index'], format='%Y%m%dT%H')
            era5_data.set_index('datetime', inplace=True)
            
            # Keep only the sea_level column from GESLA data
            sea_level_df = sea_level_data[['sea_level']]
            
            # Ensure hourly frequency for GESLA data if needed
            frequency = pd.infer_freq(sea_level_df.index)
            if frequency and frequency != 'H':
                logger.info(f"Resampling GESLA data from frequency {frequency} to hourly")
                sea_level_hourly = sea_level_df.resample('H').mean()
            else:
                sea_level_hourly = sea_level_df
            
            # Merge data based on timestamp
            merged_data = era5_data.join(sea_level_hourly, how='left')
            
            # Calculate coverage statistics
            total_rows = len(merged_data)
            rows_with_sea_level = merged_data['sea_level'].notna().sum()
            coverage_percent = (rows_with_sea_level / total_rows) * 100 if total_rows > 0 else 0
            
            logger.info(f"Sea level data coverage: {rows_with_sea_level}/{total_rows} rows ({coverage_percent:.2f}%)")
            
            # Save to output file
            output_file = os.path.join(self.output_path, f"{station_code}_ERA5_with_sea_level.csv")
            
            # Reset index to include datetime as column
            merged_data.reset_index(inplace=True)
            
            # Remove system:index column as we now have datetime
            merged_data = merged_data.drop(columns=['system:index'])
            
            # Save to CSV
            merged_data.to_csv(output_file, index=False)
            logger.info(f"Saved merged data to: {output_file}")
            
            # Remove the original file to save space
            os.remove(era5_file)
            logger.info(f"Removed original file to save space: {era5_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {era5_file}: {str(e)}", exc_info=True)
            return False
    
    def process_files(self, limit=None):
        """
        Process multiple ERA5 files to add sea level data.
        
        Args:
            limit (int, optional): Number of files to process. Set to None to process all.
            
        Returns:
            tuple: (number of successful files, total number of files processed)
        """
        # Get list of ERA5 files
        era5_files = [os.path.join(self.era5_path, f) for f in os.listdir(self.era5_path) 
                       if f.endswith('_ERA5.csv')]
        
        # Limit the number of files if specified
        if limit is not None:
            era5_files = era5_files[:limit]
        
        logger.info(f"Found {len(era5_files)} ERA5 files to process")
        
        # Process each file
        successful = 0
        for era5_file in era5_files:
            if self.process_file(era5_file):
                successful += 1
        
        logger.info(f"Processed {successful} out of {len(era5_files)} files successfully")
        return successful, len(era5_files)


def main():
    """
    Main function to run the GESLA-ERA5 integrator.
    """
    # Get current directory
    base_path = os.path.abspath(os.getcwd())
    
    # Create integrator
    integrator = GeslaEra5Integrator(base_path)
    
    # Process all files (no limit)
    successful, total = integrator.process_files()
    
    logger.info(f"Process complete. {successful}/{total} files processed successfully.")
    logger.info(f"Results saved to {integrator.output_path}")


if __name__ == "__main__":
    main()