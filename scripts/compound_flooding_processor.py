"""
Memory-Efficient Compound Flooding Processor

This script processes sea level data from GESLA and precipitation data from Meteostat
for compound flooding analysis, focusing on CONUS coastal stations.

Key features:
1. Memory efficient - processes and saves one station at a time
2. Separates metadata from time series data
3. Aggregates to daily maximum values for sea level
4. Uses maximum daily precipitation across nearby stations

Usage:
    python efficient_compound_flooding_processor.py
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from scipy.spatial import cKDTree
from scripts.gesla import GeslaDataset
import pytz
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class EfficientCompoundFloodingProcessor:
    def __init__(self, base_path):
        """
        Initialize processor with paths to data.
        
        Args:
            base_path (str): Base path to the compound_flooding directory
        """
        # Convert to absolute path
        self.base_path = os.path.abspath(base_path)
        print(f"Base path: {self.base_path}")
        
        # Find compound_flooding directory if needed
        if os.path.basename(self.base_path) != 'compound_flooding':
            if os.path.exists(os.path.join(self.base_path, 'compound_flooding')):
                self.base_path = os.path.join(self.base_path, 'compound_flooding')
                print(f"Found compound_flooding directory: {self.base_path}")
        
        # Define paths
        self.gesla_path = os.path.join(self.base_path, 'data', 'GESLA')
        self.gesla_data_path = os.path.join(self.gesla_path, 'GESLA3.0_ALL')
        self.gesla_meta_path = os.path.join(self.gesla_path, 'metadata.csv')
        self.precip_path = os.path.join(self.base_path, 'data', 'Precipitation')
        self.precip_meta_path = os.path.join(self.precip_path, 'precipitation_metadata.csv')
        
        # Verify paths exist
        print(f"GESLA path exists: {os.path.exists(self.gesla_path)}")
        print(f"GESLA data path exists: {os.path.exists(self.gesla_data_path)}")
        print(f"GESLA metadata path exists: {os.path.exists(self.gesla_meta_path)}")
        print(f"Precipitation path exists: {os.path.exists(self.precip_path)}")
        print(f"Precipitation metadata path exists: {os.path.exists(self.precip_meta_path)}")
        
        # Define CONUS bounding box
        self.conus_bounds = {
            'south_lat': 24.0,  # Southern tip of Florida
            'north_lat': 49.5,  # Northern border with Canada
            'west_lon': -125.0,  # Western coast
            'east_lon': -66.0    # Eastern coast
        }
        
        # Initialize GESLA dataset
        self.gesla_ds = GeslaDataset(
            meta_file=str(self.gesla_meta_path),
            data_path=str(self.gesla_data_path) + '/'
        )
        
        # Earth radius in kilometers for Haversine distance
        self.EARTH_RADIUS_KM = 6371.0
    
    def prepare_output_dirs(self, output_dir):
        """
        Create output directories.
        
        Args:
            output_dir (str): Path to output directory
        
        Returns:
            tuple: (output_path, time_series_path)
        """
        # Create main output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create time_series subdirectory
        time_series_path = output_path / "time_series"
        time_series_path.mkdir(exist_ok=True)
        
        return output_path, time_series_path
    
    def create_readme(self, output_path):
        """
        Create README file in output directory.
        
        Args:
            output_path (Path): Path to output directory
        """
        readme_content = """# Compound Flooding Dataset

        ## Data Structure
        This dataset contains merged sea level and precipitation data optimized for storage efficiency.

        ### Files:
        - `station_metadata.csv`: Contains metadata for all stations (name, location, timezone, etc.)
        - `time_series/`: Directory containing time series data
        - `{station_code}.csv`: Individual time series files with daily data:
            - `sea_level`: Maximum daily water level
            - `precipitation`: Maximum daily precipitation from nearby stations
            - `precip_3day`: 3-day cumulative precipitation 
            - `precip_prev_day`: Previous day's precipitation

        ### Usage:
        To reconstruct the full dataset, join the time series with the metadata based on station_code.

        ```python
        # Example code to load a station's data with metadata
        import pandas as pd

        # Load metadata
        metadata = pd.read_csv('station_metadata.csv')

        # Load time series for a specific station
        station_code = '240A'  # Example station code
        time_series = pd.read_csv(f'time_series/{station_code}.csv', parse_dates=['date'], index_col='date')

        # Get metadata for this station
        station_meta = metadata[metadata['station_code'] == station_code].iloc[0]

        # Now you can use both the time series and metadata
        print(f"Station: {station_meta['station_name']} ({station_code})")
        print(f"Location: {station_meta['latitude']}, {station_meta['longitude']}")
        print(f"Time series data points: {len(time_series)}")

        # Analysis example - find days with both high sea level and high precipitation
        high_water_events = time_series[(time_series['sea_level'] > time_series['sea_level'].quantile(0.95)) & 
                                    (time_series['precipitation'] > 0)]
        print(f"Found {len(high_water_events)} high water events with precipitation")
        ```
        """
        readme_file = output_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        print(f"Created README file: {readme_file}")
    
    def get_conus_gesla_stations(self):
        """
        Get all CONUS coastal GESLA stations.
        
        Returns:
            pandas.DataFrame: Filtered metadata for CONUS stations
        """
        # Access metadata from GeslaDataset
        meta = self.gesla_ds.meta
        
        # Filter for CONUS stations (USA and within CONUS bounding box)
        conus_meta = meta[
            (meta['country'].str.upper() == 'USA') &
            (meta['latitude'] >= self.conus_bounds['south_lat']) &
            (meta['latitude'] <= self.conus_bounds['north_lat']) &
            (meta['longitude'] >= self.conus_bounds['west_lon']) &
            (meta['longitude'] <= self.conus_bounds['east_lon'])
        ]
        
        print(f"Found {len(conus_meta)} CONUS coastal stations in GESLA data")
        return conus_meta
    
    def get_precipitation_stations(self):
        """
        Get all precipitation stations in CONUS.
        
        Returns:
            pandas.DataFrame: Precipitation station metadata
        """
        precip_meta = pd.read_csv(self.precip_meta_path)
        
        # Filter for stations within CONUS
        conus_precip_meta = precip_meta[
            (precip_meta['country'] == 'US') &
            (precip_meta['latitude'] >= self.conus_bounds['south_lat']) &
            (precip_meta['latitude'] <= self.conus_bounds['north_lat']) &
            (precip_meta['longitude'] >= self.conus_bounds['west_lon']) &
            (precip_meta['longitude'] <= self.conus_bounds['east_lon'])
        ]
        
        print(f"Loaded {len(conus_precip_meta)} precipitation stations in CONUS")
        return conus_precip_meta
    
    def build_spatial_index(self, precip_meta):
        """
        Build spatial index for precipitation stations.
        
        Args:
            precip_meta (pandas.DataFrame): Precipitation metadata
        
        Returns:
            tuple: (cKDTree, index_mapping)
        """
        # Convert lat/lon to radians for KDTree
        rad_coords = np.radians(
            np.vstack([precip_meta['latitude'].values, precip_meta['longitude'].values]).T
        )
        tree = cKDTree(rad_coords)
        
        # Map tree indices to dataframe indices
        idx_mapping = {i: idx for i, idx in enumerate(precip_meta.index)}
        
        return tree, idx_mapping
    
    def find_nearby_stations(self, lat, lon, tree, idx_mapping, precip_meta, radius_km=50):
        """
        Find precipitation stations within radius of a GESLA station.
        
        Args:
            lat, lon (float): GESLA station coordinates
            tree (cKDTree): Spatial index
            idx_mapping (dict): Index mapping
            precip_meta (pandas.DataFrame): Precipitation metadata
            radius_km (float): Search radius in kilometers
        
        Returns:
            pandas.DataFrame: Nearby precipitation stations
        """
        # Convert search radius from km to radians
        radius_rad = radius_km / self.EARTH_RADIUS_KM
        
        # Convert lat/lon to radians for query
        query_point = np.radians([lat, lon])
        
        # Find stations within radius
        indices = tree.query_ball_point(query_point, radius_rad)
        
        # Convert to metadata indices
        meta_indices = [idx_mapping[i] for i in indices]
        
        # Get nearby stations
        nearby_stations = precip_meta.loc[meta_indices].copy() if meta_indices else pd.DataFrame()
        
        # Calculate distances
        if not nearby_stations.empty:
            distances = []
            for _, station in nearby_stations.iterrows():
                dist = self.haversine_distance(lat, lon, station['latitude'], station['longitude'])
                distances.append(dist)
            
            nearby_stations['distance_km'] = distances
            nearby_stations = nearby_stations.sort_values('distance_km')
        
        return nearby_stations
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate Haversine distance between two points.
        
        Args:
            lat1, lon1, lat2, lon2 (float): Coordinates in degrees
        
        Returns:
            float: Distance in kilometers
        """
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
        
        return self.EARTH_RADIUS_KM * c
    
    def load_gesla_data(self, filename):
        """
        Load GESLA data for a station.
        
        Args:
            filename (str): GESLA filename
        
        Returns:
            tuple: (DataFrame with data, Series with metadata)
        """
        # Use GeslaDataset to load data
        data, meta = self.gesla_ds.file_to_pandas(filename)
        
        # Filter for valid data points
        data = data[data['use_flag'] == 1]
        
        # Keep only sea_level column
        processed_data = data[['sea_level']].copy()
        
        return processed_data, meta
    
    def load_precipitation_data(self, station_ids):
        """
        Load precipitation data for multiple stations.
        
        Args:
            station_ids (list): List of precipitation station IDs
        
        Returns:
            pandas.DataFrame: Daily maximum precipitation
        """
        all_data = []
        
        for station_id in station_ids:
            file_path = os.path.join(self.precip_path, f"{station_id}.csv")
            
            if not os.path.exists(file_path):
                print(f"Warning: Precipitation file for station {station_id} not found")
                continue
            
            try:
                # Load data
                precip_data = pd.read_csv(file_path)
                
                # Convert time to datetime
                precip_data['datetime'] = pd.to_datetime(precip_data['time'])
                precip_data.set_index('datetime', inplace=True)
                
                # Process precipitation data
                if 'prcp' in precip_data.columns:
                    # Convert to numeric values
                    precip_data['prcp'] = pd.to_numeric(precip_data['prcp'], errors='coerce')
                    
                    # Check if we have valid data
                    if precip_data['prcp'].notna().sum() > 0:
                        station_data = precip_data[['prcp']].copy()
                        station_data.columns = [f'prcp_{station_id}']
                        all_data.append(station_data)
                    else:
                        print(f"Warning: No valid numeric data for station {station_id}")
                else:
                    print(f"Warning: No 'prcp' column for station {station_id}")
            
            except Exception as e:
                print(f"Error processing precipitation station {station_id}: {e}")
        
        if not all_data:
            print("No valid precipitation data found")
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, axis=1)
        
        # Resample to daily resolution with maximum values
        daily_data = combined_data.resample('D').max()
        
        # Calculate maximum across all stations
        daily_data['precipitation'] = daily_data.max(axis=1, skipna=True)
        
        # Keep only precipitation column
        result = daily_data[['precipitation']]
        
        print(f"Precipitation data: Found {len(result)} days with valid data")
        
        return result
    
    def convert_tz_offset_to_name(self, offset_hours):
        """
        Convert timezone offset to standard timezone name.
        
        Args:
            offset_hours (float): Timezone offset in hours
        
        Returns:
            str: Timezone name
        """
        if offset_hours == 0:
            return 'UTC'
        
        # For other offsets, use Etc/GMT+X or Etc/GMT-X (sign is inverted)
        sign = '-' if offset_hours > 0 else '+'
        abs_offset = abs(int(offset_hours))
        
        return f"Etc/GMT{sign}{abs_offset}"
    
    def adjust_timezone(self, df, from_tz, to_tz):
        """
        Adjust timezone of DataFrame index.
        
        Args:
            df (pandas.DataFrame): Data with datetime index
            from_tz (str): Source timezone
            to_tz (str): Target timezone
        
        Returns:
            pandas.DataFrame: Adjusted data
        """
        if from_tz == to_tz:
            return df
        
        try:
            # Create a copy
            adjusted_df = df.copy()
            
            # Get timezone objects
            source_tz = pytz.timezone(from_tz)
            target_tz = pytz.timezone(to_tz)
            
            # Convert timezone
            adjusted_df.index = adjusted_df.index.tz_localize(source_tz).tz_convert(target_tz)
            
            # Remove timezone info to match original format
            adjusted_df.index = adjusted_df.index.tz_localize(None)
            
            return adjusted_df
        
        except Exception as e:
            print(f"Warning: Timezone adjustment failed: {e}")
            return df
    
    def process_station(self, station, precip_meta, precip_tree, idx_mapping,
                      time_series_path, metadata_file):
        """
        Process a single GESLA station and save results immediately.
        
        Args:
            station (pandas.Series): GESLA station metadata
            precip_meta (pandas.DataFrame): Precipitation metadata
            precip_tree (cKDTree): Spatial index for precipitation stations
            idx_mapping (dict): Index mapping for precipitation stations
            time_series_path (Path): Path to save time series data
            metadata_file (Path): Path to save metadata
        
        Returns:
            dict: Station metadata (if successful) or None
        """
        station_name = station['site_name']
        station_code = station['site_code']
        filename = station['filename']
        
        print(f"\nProcessing GESLA station: {station_name} (Code: {station_code})")
        
        try:
            # Find nearby precipitation stations
            nearby_stations = self.find_nearby_stations(
                station['latitude'], 
                station['longitude'],
                precip_tree,
                idx_mapping,
                precip_meta
            )
            
            print(f"Found {len(nearby_stations)} precipitation stations within 50km")
            
            if len(nearby_stations) == 0:
                print("No nearby precipitation stations. Skipping.")
                return None
            
            # Load GESLA data
            sea_level_data, _ = self.load_gesla_data(filename)
            
            if sea_level_data.empty:
                print("No sea level data. Skipping.")
                return None
            
            # Load precipitation data
            precipitation_data = self.load_precipitation_data(nearby_stations['id'].tolist())
            
            if precipitation_data is None or precipitation_data.empty:
                print("No precipitation data. Skipping.")
                return None
            
            # Get timezone information
            gesla_tz_hours = station['time_zone_hours']
            gesla_tz = self.convert_tz_offset_to_name(gesla_tz_hours)
            
            # Use timezone of closest precipitation station
            precip_tz = nearby_stations.iloc[0]['timezone']
            
            # Adjust precipitation data to GESLA timezone
            adjusted_precip = self.adjust_timezone(precipitation_data, precip_tz, gesla_tz)
            
            # Create lookup dictionaries for precipitation values
            precip_dict = {}
            precip_3day_dict = {}
            
            # Sort precipitation data by date
            sorted_precip = adjusted_precip.sort_index()
            
            # Get dates and values
            dates = sorted_precip.index.date
            values = sorted_precip['precipitation'].values
            
            # Create mappings
            for i, date in enumerate(dates):
                precip_dict[date] = values[i]
                
                # Calculate 3-day totals
                if i >= 2:
                    precip_3day_dict[date] = sum(values[i-2:i+1])
                elif i == 1:
                    precip_3day_dict[date] = sum(values[i-1:i+1])
                else:
                    precip_3day_dict[date] = values[i]
            
            # Process sea level data
            # Add date column for aggregation
            sea_level_data['date'] = sea_level_data.index.date
            
            # Group by date and calculate daily maximum
            daily_sea_level = sea_level_data.groupby('date')['sea_level'].max().reset_index()
            daily_sea_level['date'] = pd.to_datetime(daily_sea_level['date'])
            daily_sea_level.set_index('date', inplace=True)
            
            # Add precipitation data
            daily_sea_level['precipitation'] = [
                precip_dict.get(date, np.nan) for date in daily_sea_level.index.date
            ]
            
            # Add 3-day precipitation totals
            daily_sea_level['precip_3day'] = [
                precip_3day_dict.get(date, np.nan) for date in daily_sea_level.index.date
            ]
            
            # Add previous day's precipitation
            daily_sea_level['precip_prev_day'] = daily_sea_level['precipitation'].shift(1)
            
            # Calculate coverage statistics
            sea_level_coverage = (100 - daily_sea_level['sea_level'].isna().mean() * 100).round(2)
            precip_coverage = (100 - daily_sea_level['precipitation'].isna().mean() * 100).round(2)
            
            # Create metadata
            metadata = {
                'station_code': station_code,
                'station_name': station_name,
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'timezone': gesla_tz,
                'start_date': daily_sea_level.index.min(),
                'end_date': daily_sea_level.index.max(),
                'num_days': len(daily_sea_level),
                'sea_level_coverage': sea_level_coverage,
                'precipitation_coverage': precip_coverage
            }
            
            # Save time series to file
            output_file = time_series_path / f"{station_code}.csv"
            daily_sea_level.to_csv(output_file)
            print(f"Saved daily time series for station {station_code} to {output_file}")
            
            # Append metadata to file if it exists, otherwise create new file
            metadata_df = pd.DataFrame([metadata])
            if metadata_file.exists():
                metadata_df.to_csv(metadata_file, mode='a', header=False, index=False)
            else:
                metadata_df.to_csv(metadata_file, index=False)
            
            # Clean up memory
            del sea_level_data
            del precipitation_data
            del daily_sea_level
            gc.collect()
            
            return metadata
            
        except Exception as e:
            print(f"Error processing station {station_name}: {e}")
            return None
    
    def process_all_stations(self, output_dir):
        """
        Process all CONUS coastal GESLA stations.
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            int: Number of successfully processed stations
        """
        # Prepare output directories
        output_path, time_series_path = self.prepare_output_dirs(output_dir)
        
        # Create metadata file path
        metadata_file = output_path / "station_metadata.csv"
        
        # Get GESLA stations
        gesla_stations = self.get_conus_gesla_stations()
        
        # Get precipitation stations
        precip_stations = self.get_precipitation_stations()
        
        # Build spatial index
        precip_tree, idx_mapping = self.build_spatial_index(precip_stations)
        
        # Count successful stations
        successful_stations = 0
        
        # Process each station
        for _, station in gesla_stations.iterrows():
            metadata = self.process_station(
                station, 
                precip_stations, 
                precip_tree, 
                idx_mapping,
                time_series_path,
                metadata_file
            )
            
            if metadata is not None:
                successful_stations += 1
        
        # Create README file
        self.create_readme(output_path)
        
        print(f"\nSuccessfully processed {successful_stations} out of {len(gesla_stations)} stations")
        
        return successful_stations

def main():
    """
    Main function to run the processor.
    """
    # Get current directory
    base_path = os.path.abspath(os.getcwd())
    
    # Create processor
    processor = EfficientCompoundFloodingProcessor(base_path)
    
    # Set output directory
    output_dir = os.path.join(base_path, "processed_data")
    print(f"Output directory: {output_dir}")
    
    # Process all stations
    num_processed = processor.process_all_stations(output_dir)
    
    print(f"Processing complete. {num_processed} stations successfully processed.")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()