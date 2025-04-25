"""
Ground Precipitation Data Integration (Bounding Box Method with Date Range Coverage)

This script enhances GESLA-ERA5 combined files by adding ground-based precipitation data
from nearby weather stations using the meteostat library.

Optimized for multiprocessing on systems with many CPU cores.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from datetime import datetime, timedelta
from meteostat import Stations, Hourly
import multiprocessing as mp
from functools import partial
import tqdm
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ground_precipitation_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Ground-Precipitation-Integrator")

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points."""
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

# Global stations cache - shared between processes
_stations_cache = None

def get_all_stations():
    """Get all stations from Meteostat (globally cached)."""
    global _stations_cache
    if _stations_cache is None:
        logger.info("Fetching all Meteostat stations (this may take a moment)...")
        _stations_cache = Stations().fetch()
        logger.info(f"Loaded {len(_stations_cache)} Meteostat stations")
    return _stations_cache


class GroundPrecipitationIntegrator:
    """
    A class to integrate ground-based precipitation data with GESLA-ERA5 files.
    """
    
    def __init__(self, base_path, n_processes=None):
        """Initialize the integrator with paths."""
        # Convert to absolute path
        self.base_path = os.path.abspath(base_path)
        logger.info(f"Base path: {self.base_path}")
        
        # Set number of processes
        self.n_processes = n_processes or os.cpu_count()
        logger.info(f"Using {self.n_processes} CPU cores for processing")
        
        # Find compound_flooding directory if needed
        if os.path.basename(self.base_path) != 'compound_flooding':
            compound_path = os.path.join(self.base_path, 'compound_flooding')
            if os.path.exists(compound_path):
                self.base_path = compound_path
                logger.info(f"Found compound_flooding directory: {self.base_path}")
        
        # Define paths
        self.gesla_path = os.path.join(self.base_path, 'data', 'GESLA')
        self.gesla_meta_path = os.path.join(self.gesla_path, 'metadata.csv')
        self.integrated_path = os.path.join(self.base_path, 'GESLA_ERA5_with_sea_level')
        
        # Define output path for metadata
        self.output_meta_path = os.path.join(self.base_path, 'ground_precipitation_metadata.csv')
        
        # Initial search box size (degrees)
        self.INITIAL_SEARCH_DEG = 0.5  # Roughly 50km at mid-latitudes
        
        # Maximum search box size (degrees)
        self.MAX_SEARCH_DEG = 2.0  # Roughly 200km at mid-latitudes
        
        # Verify paths exist
        self._verify_paths()
        
        # Load GESLA metadata
        self.gesla_meta = pd.read_csv(self.gesla_meta_path)
        logger.info(f"Loaded GESLA metadata with {len(self.gesla_meta)} stations")
        
        # Using "SITE CODE" directly as mentioned by the user
        self.station_code_column = "SITE CODE"
        logger.info(f"Using column '{self.station_code_column}' for station codes")
        
        # Preload stations to make them available to all processes
        _ = get_all_stations()
    
    def _verify_paths(self):
        """Verify that all required paths exist and log their status."""
        paths = {
            "GESLA metadata path": self.gesla_meta_path,
            "Integrated data path": self.integrated_path
        }
        
        for name, path in paths.items():
            exists = os.path.exists(path)
            logger.info(f"{name} exists: {exists} - {path}")
            if not exists:
                logger.warning(f"Required path not found: {path}")
    
    def extract_station_code(self, filename):
        """Extract station code from integrated filename."""
        # Extract the part before '_ERA5_with_sea_level.csv'
        match = re.match(r'(.+)_ERA5_with_sea_level\.csv$', filename)
        if match:
            return match.group(1)
        return None
    
    def get_nearby_stations(self, lat, lon, search_deg):
        """Get nearby stations within a bounding box."""
        all_stations = get_all_stations()
        
        # Define bounding box
        min_lat = lat - search_deg
        max_lat = lat + search_deg
        min_lon = lon - search_deg
        max_lon = lon + search_deg
        
        # Filter stations by bounding box
        nearby_stations = all_stations[
            (all_stations['latitude'] >= min_lat) &
            (all_stations['latitude'] <= max_lat) &
            (all_stations['longitude'] >= min_lon) &
            (all_stations['longitude'] <= max_lon)
        ].copy()
        
        # Reset index to make it easier to work with
        nearby_stations = nearby_stations.reset_index()
        
        if nearby_stations.empty:
            return nearby_stations
        
        # Calculate distances and sort by distance
        distances = []
        for _, station in nearby_stations.iterrows():
            dist = haversine_distance(lat, lon, station['latitude'], station['longitude'])
            distances.append(dist)
        
        nearby_stations['distance_km'] = distances
        nearby_stations = nearby_stations.sort_values('distance_km')
        
        return nearby_stations
    
    def get_precipitation_data(self, station_id, start_date, end_date):
        """Get precipitation data for a station."""
        try:
            data = Hourly(station_id, start=start_date, end=end_date)
            hourly_data = data.fetch()
            return hourly_data
        except Exception as e:
            logger.warning(f"Error fetching data for station {station_id}: {str(e)}")
            return None
    
    def find_best_precipitation_station(self, lat, lon, start_date, end_date):
        """
        Find the best nearby precipitation station with highest data coverage
        using a bounding box approach.
        """
        try:
            # Add diagnostics for coordinates and date range
            logger.info(f"Searching for precipitation stations near ({lat}, {lon})")
            logger.info(f"Need data for date range: {start_date} to {end_date}")
            
            # First try with initial search box
            best_station, best_data, used_buffer = self._search_precipitation_stations_bbox(
                lat, lon, start_date, end_date, self.INITIAL_SEARCH_DEG
            )
            
            # If no stations found, try with expanded search box
            if best_station is None:
                logger.info(f"No suitable stations found within {self.INITIAL_SEARCH_DEG} degrees, expanding search to {self.MAX_SEARCH_DEG} degrees")
                best_station, best_data, used_buffer = self._search_precipitation_stations_bbox(
                    lat, lon, start_date, end_date, self.MAX_SEARCH_DEG
                )
                used_buffer = True
            
            return best_station, best_data, used_buffer
            
        except Exception as e:
            logger.error(f"Error finding precipitation station: {str(e)}", exc_info=True)
            return None, None, False
    
    def _search_precipitation_stations_bbox(self, lat, lon, start_date, end_date, search_deg):
        """
        Search for precipitation stations within a bounding box and evaluate based on
        coverage for the specific date range needed.
        """
        try:
            # Get nearby stations
            nearby_stations = self.get_nearby_stations(lat, lon, search_deg)
            
            if nearby_stations.empty:
                logger.warning(f"No weather stations found within {search_deg} degrees of ({lat}, {lon})")
                return None, None, False
            
            logger.info(f"Found {len(nearby_stations)} nearby weather stations within {search_deg} degrees")
            
            # Evaluate data coverage for up to 5 closest stations
            best_station = None
            best_coverage = 0
            best_data = None
            
            # Only check closest 5 stations to save time
            for _, station in nearby_stations.head(5).iterrows():
                station_id = station['id']
                
                # Get hourly precipitation data
                logger.info(f"Checking station {station_id} ({station['name']}) for date range: {start_date} to {end_date}")
                hourly_data = self.get_precipitation_data(station_id, start_date, end_date)
                
                if hourly_data is None or hourly_data.empty or 'prcp' not in hourly_data.columns:
                    logger.warning(f"No precipitation data for station {station_id}")
                    continue
                
                # Calculate coverage for the SPECIFIC date range
                total_hours = (end_date - start_date).total_seconds() / 3600
                hours_with_data = hourly_data['prcp'].notna().sum()
                coverage = (hours_with_data / total_hours) * 100
                
                logger.info(f"Station {station_id} ({station['name']}) has {coverage:.2f}% precipitation data coverage for the requested date range")
                logger.info(f"Date range has {total_hours} hours, data available for {hours_with_data} hours")
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_station = station
                    best_data = hourly_data
            
            if best_station is None:
                logger.warning(f"No station with precipitation data found near ({lat}, {lon})")
                return None, None, False
            
            # Check if station is outside initial buffer
            used_buffer = search_deg > self.INITIAL_SEARCH_DEG
            
            logger.info(f"Selected station {best_station['id']} ({best_station['name']}) with {best_coverage:.2f}% coverage for requested date range")
            return best_station, best_data, used_buffer
            
        except Exception as e:
            logger.error(f"Error searching for precipitation stations: {str(e)}", exc_info=True)
            return None, None, False

# Function for multiprocessing - must be at module level for Windows compatibility
def process_file_worker(args):
    """Process a single file and return metadata."""
    file_path, gesla_meta, station_code_column, integrated_path, initial_search_deg, max_search_deg = args
    
    try:
        filename = os.path.basename(file_path)
        # Extract station code from filename
        match = re.match(r'(.+)_ERA5_with_sea_level\.csv$', filename)
        if not match:
            logger.warning(f"Could not extract station code from filename: {filename}")
            return None
            
        station_code = match.group(1)
        logger.info(f"Processing file: {filename} (Station Code: {station_code})")
        
        # Find station in GESLA metadata
        station_meta = gesla_meta[gesla_meta[station_code_column] == station_code]
        
        if station_meta.empty:
            # Try case insensitive matching
            station_meta = gesla_meta[gesla_meta[station_code_column].str.lower() == station_code.lower()]
            
            if station_meta.empty:
                logger.warning(f"No GESLA metadata found for station code: {station_code} (case insensitive)")
                
                # Create metadata record for missing station
                metadata = {
                    'sea_level_station_code': station_code,
                    'sea_level_station_name': station_code,
                    'sea_level_latitude': None,
                    'sea_level_longitude': None,
                    'precip_station_id': None,
                    'precip_station_name': None,
                    'precip_station_latitude': None,
                    'precip_station_longitude': None,
                    'distance_km': None,
                    'used_buffer_distance': None,
                    'precipitation_coverage': 0,
                    'date_range_days': None,
                    'notes': 'No GESLA metadata found'
                }
                return metadata
        
        # Get lat/lon
        try:
            lat_column = [col for col in station_meta.columns if 'lat' in col.lower()][0]
            lon_column = [col for col in station_meta.columns if 'lon' in col.lower()][0]
            
            lat = station_meta[lat_column].iloc[0]
            lon = station_meta[lon_column].iloc[0]
        except (IndexError, KeyError):
            logger.warning(f"Could not find latitude/longitude columns for station {station_code}")
            return None
        
        # Load integrated data
        integrated_data = pd.read_csv(file_path, parse_dates=['datetime'])
        
        # Get date range from file
        start_date = integrated_data['datetime'].min()
        end_date = integrated_data['datetime'].max()
        
        date_range_days = (end_date - start_date).days
        logger.info(f"File date range: {start_date} to {end_date} ({date_range_days} days)")
        
        # Find name column
        name_columns = [col for col in station_meta.columns if 'name' in col.lower()]
        name_column = name_columns[0] if name_columns else station_code_column
        station_name = station_meta[name_column].iloc[0] if name_column else station_code
        
        # Create precipitation integrator for this file
        integrator = GroundPrecipitationIntegrator(os.path.dirname(os.path.dirname(file_path)))
        
        # Find best precipitation station
        best_station, precip_data, used_buffer = integrator.find_best_precipitation_station(
            lat, lon, start_date, end_date
        )
        
        if best_station is None or precip_data is None:
            logger.warning(f"No suitable precipitation station found for {station_code}")
            
            # Create metadata record for station with no precipitation data
            metadata = {
                'sea_level_station_code': station_code,
                'sea_level_station_name': station_name,
                'sea_level_latitude': lat,
                'sea_level_longitude': lon,
                'precip_station_id': None,
                'precip_station_name': None,
                'precip_station_latitude': None,
                'precip_station_longitude': None,
                'distance_km': None,
                'used_buffer_distance': None,
                'precipitation_coverage': 0,
                'date_range_days': date_range_days,
                'notes': 'No precipitation station found'
            }
            return metadata
        
        # Format precipitation data
        precip_data = precip_data[['prcp']].copy()
        precip_data.columns = ['ground_precipitation']
        
        # Merge with integrated data
        integrated_data.set_index('datetime', inplace=True)
        precip_data.index.name = 'datetime'
        
        # Ensure hourly frequency for both datasets
        if not integrated_data.index.is_monotonic_increasing:
            integrated_data = integrated_data.sort_index()
        
        if not precip_data.index.is_monotonic_increasing:
            precip_data = precip_data.sort_index()
        
        # Merge datasets
        merged_data = integrated_data.join(precip_data, how='left')
        
        # Reset index for saving
        merged_data.reset_index(inplace=True)
        
        # Calculate coverage based on our specific date range
        total_hours = len(integrated_data)
        hours_with_data = merged_data['ground_precipitation'].notna().sum()
        coverage = (hours_with_data / total_hours) * 100 if total_hours > 0 else 0
        
        logger.info(f"Ground precipitation coverage: {coverage:.2f}% ({hours_with_data}/{total_hours} hours)")
        
        # Save results
        merged_data.to_csv(file_path, index=False)
        logger.info(f"Updated file with ground precipitation data: {file_path}")
        
        # Create metadata record
        station_id = best_station['id']
        station_name = best_station['name']
        station_lat = best_station['latitude']
        station_lon = best_station['longitude']
        distance = haversine_distance(lat, lon, station_lat, station_lon)
        
        metadata = {
            'sea_level_station_code': station_code,
            'sea_level_station_name': station_name,
            'sea_level_latitude': lat,
            'sea_level_longitude': lon,
            'precip_station_id': station_id,
            'precip_station_name': station_name,
            'precip_station_latitude': station_lat,
            'precip_station_longitude': station_lon,
            'distance_km': distance,
            'used_buffer_distance': used_buffer,
            'precipitation_coverage': coverage,
            'date_range_days': date_range_days,
            'notes': 'Successful'
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
        return None

class ProcessFilesManager:
    """Main class to manage file processing with multiprocessing."""
    
    def __init__(self, base_path, n_processes=None):
        """Initialize the manager."""
        self.base_path = os.path.abspath(base_path)
        self.n_processes = n_processes or os.cpu_count()
        
        # Find compound_flooding directory if needed
        if os.path.basename(self.base_path) != 'compound_flooding':
            compound_path = os.path.join(self.base_path, 'compound_flooding')
            if os.path.exists(compound_path):
                self.base_path = compound_path
        
        # Define paths
        self.gesla_path = os.path.join(self.base_path, 'data', 'GESLA')
        self.gesla_meta_path = os.path.join(self.gesla_path, 'metadata.csv')
        self.integrated_path = os.path.join(self.base_path, 'GESLA_ERA5_with_sea_level')
        self.output_meta_path = os.path.join(self.base_path, 'ground_precipitation_metadata.csv')
        
        # Initial search box size (degrees)
        self.INITIAL_SEARCH_DEG = 0.5
        self.MAX_SEARCH_DEG = 2.0
        
        # Load GESLA metadata
        self.gesla_meta = pd.read_csv(self.gesla_meta_path)
        self.station_code_column = "SITE CODE"
        
        # Get list of files
        self.integrated_files = [
            os.path.join(self.integrated_path, f) 
            for f in os.listdir(self.integrated_path) 
            if f.endswith('_ERA5_with_sea_level.csv')
        ]
    
    def process_files(self):
        """Process all files using multiprocessing."""
        total_files = len(self.integrated_files)
        logger.info(f"Found {total_files} integrated files to process")
        
        if total_files == 0:
            logger.warning("No files found to process")
            return 0, 0
        
        # Prepare arguments for worker function
        worker_args = []
        for file_path in self.integrated_files:
            worker_args.append((
                file_path, 
                self.gesla_meta, 
                self.station_code_column,
                self.integrated_path,
                self.INITIAL_SEARCH_DEG,
                self.MAX_SEARCH_DEG
            ))
        
        # Process files in parallel
        start_time = time.time()
        with mp.Pool(processes=self.n_processes) as pool:
            results = list(tqdm.tqdm(
                pool.imap(process_file_worker, worker_args),
                total=total_files,
                desc="Processing files"
            ))
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        # Filter valid results and count successful
        metadata_records = [r for r in results if r is not None]
        successful = sum(1 for r in metadata_records if r.get('notes') == 'Successful')
        
        # Save all metadata
        if metadata_records:
            metadata_df = pd.DataFrame(metadata_records)
            metadata_df.to_csv(self.output_meta_path, index=False)
            logger.info(f"Saved metadata for {len(metadata_records)} stations to {self.output_meta_path}")
        
        logger.info(f"Processed {successful} out of {total_files} files successfully with precipitation data")
        return successful, total_files


def main():
    """Main function to run the ground precipitation integrator."""
    # Get current directory
    base_path = os.path.abspath(os.getcwd())
    
    # Use all available CPU cores
    n_processes = 32
    
    # Create manager and process files
    manager = ProcessFilesManager(base_path, n_processes=n_processes)
    successful, total = manager.process_files()
    
    logger.info(f"Process complete. {successful}/{total} files processed successfully with precipitation data.")
    logger.info(f"Metadata saved to {manager.output_meta_path}")


if __name__ == "__main__":
    main()