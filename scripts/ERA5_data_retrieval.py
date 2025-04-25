import ee
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz
import gc
from compound_flooding_processor import EfficientCompoundFloodingProcessor

class GESLA_ERA5_Extractor:
    """
    Extracts ERA5 data at GESLA station coordinates and merges with GESLA sea level data.
    Focuses on efficient memory management and proper time zone handling.
    """
    
    def __init__(self, gesla_processor, min_years=5, output_dir='ERA5_data'):
        """Initialize the extractor with GESLA processor and parameters."""
        self.gesla_processor = gesla_processor
        self.min_years = min_years
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Earth Engine
        try:
            ee.Initialize(project = 'ee-074bce151saurav')
            print("Earth Engine initialized successfully")
        except Exception as e:
            print(f"Error initializing Earth Engine: {e}")
            print("Please make sure you've authenticated with Earth Engine")
            raise
        
        # Define ERA5 data availability period
        self.era5_start_date = datetime(1981, 1, 1)  # ERA5 Land hourly starts from 1981-01-01
        self.era5_end_date = datetime.now()  # Current date as the upper limit
        
        print(f"GESLA-ERA5 Extractor initialized. Minimum years threshold: {min_years}")
        print(f"ERA5 Land hourly data available from: {self.era5_start_date.strftime('%Y-%m-%d')} to present")
    
    def get_station_date_range(self, filename):
        """Get actual start/end dates of valid data for a GESLA station."""
        try:
            # Load the GESLA data using the processor's methods
            data, _ = self.gesla_processor.gesla_ds.file_to_pandas(filename)
            
            # Filter for valid data points (use_flag = 1)
            valid_data = data[data['use_flag'] == 1]
            
            if valid_data.empty:
                return None, None
                
            # Get earliest and latest dates with valid data
            start_date = valid_data.index.min()
            end_date = valid_data.index.max()
            
            return start_date, end_date
            
        except Exception as e:
            print(f"Error getting date range for {filename}: {e}")
            return None, None
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate Haversine distance between two points.
        
        Args:
            lat1, lon1, lat2, lon2 (float): Coordinates in degrees
        
        Returns:
            float: Distance in kilometers
        """
        # Earth radius in kilometers
        EARTH_RADIUS_KM = 6371.0
        
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
        
        return EARTH_RADIUS_KM * c
    
    def process_all_stations(self, export_to_drive=False, resume=True):
        """Process all GESLA stations in CONUS meeting the minimum years threshold with resume capability.
        
        Args:
            export_to_drive: Whether to export to Google Drive
            resume: Whether to resume from previous run or start fresh
        """
        # Get CONUS GESLA stations using the processor's method
        conus_stations = self.gesla_processor.get_conus_gesla_stations()
        
        results = {}
        metadata_rows = []
        processed_stations = set()
        
        # If resuming, check for already processed station files
        if resume:
            # Get list of files in output directory
            output_files = os.listdir(self.output_dir)
            
            # Look for station output files (pattern: STATION_CODE_ERA5_sea_level.csv)
            for filename in output_files:
                if filename.endswith('_ERA5_sea_level.csv'):
                    # Extract station code from filename
                    station_code = filename.split('_ERA5_sea_level.csv')[0]
                    processed_stations.add(station_code)
            
            if processed_stations:
                print(f"Resuming extraction. Found {len(processed_stations)} already processed stations.")
        
        for _, station in conus_stations.iterrows():
            station_code = station['site_code']
            station_name = station['site_name']
            lat = station['latitude']
            lon = station['longitude']
            filename = station['filename']
            
            # Skip already processed stations
            if station_code in processed_stations:
                print(f"\nSkipping already processed station: {station_code} ({station_name})")
                results[station_code] = "Already processed"
                continue
                
            # Rest of the existing processing code remains the same
            print(f"\nEvaluating station: {station_code} ({station_name})")
            
            # Get actual data date range (not just metadata dates)
            start_date, end_date = self.get_station_date_range(filename)
            
            if start_date is None or end_date is None:
                print(f"  Skipping - no valid data found")
                results[station_code] = "Skipped - no valid data"
                continue
            
            # Adjust date range to ERA5 availability
            original_start = start_date
            original_end = end_date
            
            # Ensure dates are within ERA5 data range
            start_date = max(start_date, self.era5_start_date)
            end_date = min(end_date, self.era5_end_date)
            
            if start_date >= end_date:
                print(f"  Skipping - no overlap with ERA5 data availability (station data: {original_start.strftime('%Y-%m-%d')} to {original_end.strftime('%Y-%m-%d')})")
                results[station_code] = "Skipped - no overlap with ERA5 data period"
                continue
            
            # Calculate duration in years of the adjusted date range
            duration_years = relativedelta(end_date, start_date).years + \
                            (relativedelta(end_date, start_date).months / 12)
            
            if duration_years < self.min_years:
                print(f"  Skipping - only {duration_years:.2f} years of overlapping data (minimum: {self.min_years})")
                results[station_code] = f"Skipped - insufficient overlapping data ({duration_years:.2f} years)"
                continue
                
            print(f"  Processing station {station_code} ({station_name})")
            print(f"  Coordinates: {lat}, {lon}")
            print(f"  Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({duration_years:.2f} years)")
            
            try:
                # Extract ERA5 data and merge with GESLA sea level data
                if export_to_drive:
                    status = self.extract_era5_to_drive(station_code, station, start_date, end_date)
                else:
                    output_file = os.path.join(self.output_dir, f"{station_code}_ERA5_sea_level.csv")
                    status = self.extract_era5_direct(station_code, station, start_date, end_date, output_file)
                    
                results[station_code] = status
                
                # Add to metadata
                metadata_rows.append({
                    'station_code': station_code,
                    'station_name': station_name,
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': start_date,
                    'end_date': end_date,
                    'original_start_date': original_start,
                    'original_end_date': original_end,
                    'duration_years': duration_years,
                    'extraction_status': 'Success' if 'Error' not in status else 'Failed'
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                results[station_code] = f"Error: {str(e)}"
                
                # Add to metadata with error status
                metadata_rows.append({
                    'station_code': station_code,
                    'station_name': station_name,
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': start_date,
                    'end_date': end_date,
                    'original_start_date': original_start,
                    'original_end_date': original_end,
                    'duration_years': duration_years,
                    'extraction_status': f'Failed: {str(e)}'
                })
            
            # Force garbage collection between stations
            gc.collect()
        
        # Save metadata to CSV
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_file = os.path.join(self.output_dir, 'ERA5_extraction_metadata.csv')
        metadata_df.to_csv(metadata_file, index=False)
        print(f"\nMetadata saved to {metadata_file}")
        print(f"Successfully processed: {metadata_df['extraction_status'].eq('Success').sum()} stations")
        
        return results
    
    def convert_tz_offset_to_name(self, offset_hours):
        """Convert timezone offset to standard timezone name."""
        if offset_hours == 0:
            return 'UTC'
        
        # For other offsets, use Etc/GMT+X or Etc/GMT-X (sign is inverted for Etc/GMT)
        sign = '-' if offset_hours > 0 else '+'
        abs_offset = abs(int(offset_hours))
        
        return f"Etc/GMT{sign}{abs_offset}"
    
    def extract_era5_direct(self, station_code, station, start_date, end_date, output_file):
        """Extract ERA5 data and merge with GESLA sea level data at hourly resolution."""
        lat = station['latitude']
        lon = station['longitude']
        filename = station['filename']
        
        # Get timezone information
        tz_hours = station['time_zone_hours']
        local_tz_name = self.convert_tz_offset_to_name(tz_hours)
        print(f"  Station timezone: {local_tz_name} (UTC{'+' if tz_hours >= 0 else ''}{tz_hours})")
        
        # First, get the GESLA sea level data
        try:
            # Load sea level data
            sea_level_data, _ = self.gesla_processor.gesla_ds.file_to_pandas(filename)
            
            # Filter for valid data points (use_flag = 1)
            sea_level_data = sea_level_data[sea_level_data['use_flag'] == 1]
            
            # Filter to the requested date range
            sea_level_data = sea_level_data[(sea_level_data.index >= start_date) & 
                                        (sea_level_data.index <= end_date)]
            
            if sea_level_data.empty:
                print("  No valid sea level data in the requested time range.")
                sea_level_hourly = None
            else:
                # Important: GESLA data is in local time, but ERA5 is in UTC
                # First, localize GESLA timestamps to the station's timezone
                local_tz = pytz.timezone(local_tz_name)
                
                # Create a copy with timezone-aware timestamps
                sea_level_data_tz = sea_level_data.copy()
                sea_level_data_tz.index = sea_level_data_tz.index.tz_localize(
                    local_tz, ambiguous='raise', nonexistent='raise'
                )
                
                # Convert to UTC to match ERA5
                sea_level_data_utc = sea_level_data_tz.copy()
                sea_level_data_utc.index = sea_level_data_utc.index.tz_convert('UTC')
                
                # Resample to hourly (using maximum for sea levels)
                sea_level_hourly = sea_level_data_utc['sea_level'].resample('H').max()
                
                print(f"  Loaded {len(sea_level_hourly)} hourly sea level records (UTC)")
        except Exception as e:
            print(f"  Error loading or processing GESLA data: {e}")
            sea_level_hourly = None
        
        # Now extract ERA5 data (which is in UTC)
        all_data = []
        
        # Start at the beginning of the date range, rounded to hour
        current_date = start_date.replace(minute=0, second=0, microsecond=0)
        
        # Original point
        original_point = ee.Geometry.Point(lon, lat)
        
        # Check if station is on land or needs nearest land point
        sample_point = original_point  # Start with original point
        is_on_land = True
        
        # Try to get sample data for the first date to check if we're on land
        try:
            test_date = ee.Date(current_date.strftime('%Y-%m-%dT%H:%M:%S'))
            test_end = ee.Date(test_date.advance(1, 'day'))
            
            # Get a sample image
            test_img = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                .filterDate(test_date, test_end) \
                .first()
            
            # Try to sample at the exact point
            sample = test_img.sample(
                region=original_point,
                scale=11132,
                projection='EPSG:4326'
            )
            
            # If sample size is 0, station is in ocean, try nearby points
            if sample.size().getInfo() == 0:
                is_on_land = False
                print(f"  Station appears to be in ocean. Trying nearby land points...")
                
                # Define offsets to try (in degrees)
                # This creates a grid of points around the original point
                offsets = [
                    (0.05, 0),    # East
                    (-0.05, 0),   # West
                    (0, 0.05),    # North
                    (0, -0.05),   # South
                    (0.05, 0.05), # Northeast
                    (-0.05, 0.05), # Northwest
                    (0.05, -0.05), # Southeast
                    (-0.05, -0.05) # Southwest
                ]
                
                for offset_lon, offset_lat in offsets:
                    # Create test point with offset
                    test_point = ee.Geometry.Point(lon + offset_lon, lat + offset_lat)
                    
                    # Sample at this point
                    test_sample = test_img.sample(
                        region=test_point,
                        scale=11132,
                        projection='EPSG:4326'
                    )
                    
                    # If we get data, use this point
                    if test_sample.size().getInfo() > 0:
                        sample_point = test_point
                        distance = self.haversine_distance(lat, lon, lat + offset_lat, lon + offset_lon)
                        print(f"  Found land at offset point: {lat + offset_lat}, {lon + offset_lon} (Distance: {distance:.2f} km)")
                        break
                
                if not is_on_land and sample_point == original_point:
                    print(f"  WARNING: No land found with standard offsets. Trying larger offsets...")
                    
                    # Try larger offsets (in degrees)
                    larger_offsets = [
                        (0.1, 0),     # East
                        (-0.1, 0),    # West
                        (0, 0.1),     # North
                        (0, -0.1),    # South
                        (0.1, 0.1),   # Northeast
                        (-0.1, 0.1),  # Northwest
                        (0.1, -0.1),  # Southeast
                        (-0.1, -0.1)  # Southwest
                    ]
                    
                    for offset_lon, offset_lat in larger_offsets:
                        # Create test point with offset
                        test_point = ee.Geometry.Point(lon + offset_lon, lat + offset_lat)
                        
                        # Sample at this point
                        test_sample = test_img.sample(
                            region=test_point,
                            scale=11132,
                            projection='EPSG:4326'
                        )
                        
                        # If we get data, use this point
                        if test_sample.size().getInfo() > 0:
                            sample_point = test_point
                            distance = self.haversine_distance(lat, lon, lat + offset_lat, lon + offset_lon)
                            print(f"  Found land at offset point: {lat + offset_lat}, {lon + offset_lon} (Distance: {distance:.2f} km)")
                            break
            
        except Exception as e:
            print(f"  Error checking land/ocean status: {e}")
        
        # Process data in monthly chunks to avoid timeouts and memory issues
        while current_date <= end_date:
            # Set chunk end date (end of current month or overall end date)
            next_month = current_date + relativedelta(months=1)
            chunk_end = min(next_month, end_date)
            
            print(f"  Processing chunk: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            # Convert dates to EE format
            start_date_ee = ee.Date(current_date.strftime('%Y-%m-%dT%H:%M:%S'))
            end_date_ee = ee.Date(chunk_end.strftime('%Y-%m-%dT%H:%M:%S'))
            
            # Get ERA5 collection for this chunk
            era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                .filterDate(start_date_ee, end_date_ee) \
                .select(['total_precipitation', 'u_component_of_wind_10m', 
                    'v_component_of_wind_10m', 'surface_pressure', 'temperature_2m'])
            
            def extract_at_point(image):
                try:
                    # Get timestamp first (won't fail)
                    timestamp = image.date().millis()
                    
                    # Try to sample at point
                    sample_result = image.sample(
                        region=sample_point,
                        scale=11132,
                        projection='EPSG:4326'
                    )
                    
                    # Check if we got any results
                    if sample_result.size().getInfo() > 0:
                        # Get the first result and add timestamp
                        return sample_result.first().set('timestamp', timestamp)
                    else:
                        # No data found, return a placeholder feature
                        return ee.Feature(sample_point, {'timestamp': timestamp})
                except Exception as e:
                    # Always return something, even on error
                    print(f"      Error sampling image: {e}")
                    return ee.Feature(sample_point, {'timestamp': image.date().millis()})
            
            # Filter out features that don't have the required properties
            point_collection = era5.map(extract_at_point).filter(
                ee.Filter.notNull(['total_precipitation', 'u_component_of_wind_10m', 
                                'v_component_of_wind_10m', 'surface_pressure', 'temperature_2m'])
            )
            
            try:
                # Check if collection is empty
                collection_size = point_collection.size().getInfo()
                if collection_size == 0:
                    print(f"    No ERA5 data found for this period. Skipping.")
                    current_date = next_month
                    continue
                
                # Convert to a list for batch processing
                features = point_collection.toList(10000)
                
                # Get size of the collection
                size = features.size().getInfo()
                print(f"    Found {size} hourly ERA5 records in this chunk")
                
                # Process in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, size, batch_size):
                    end_idx = min(i + batch_size, size)
                    batch = features.slice(i, end_idx)
                    
                    # Convert batch to a feature collection
                    batch_fc = ee.FeatureCollection(batch)
                    
                    # Get the data
                    batch_data = batch_fc.getInfo()
                    
                    # Process features
                    for feature in batch_data['features']:
                        props = feature['properties']
                        # ERA5 timestamps are in milliseconds since epoch (UTC)
                        timestamp = datetime.fromtimestamp(props['timestamp'] / 1000, tz=pytz.UTC)
                        
                        # Calculate wind speed from components
                        u_wind = props.get('u_component_of_wind_10m', 0)
                        v_wind = props.get('v_component_of_wind_10m', 0)
                        wind_speed = np.sqrt(u_wind**2 + v_wind**2) if u_wind is not None and v_wind is not None else None
                        
                        # Convert temperature to Celsius
                        temp_k = props.get('temperature_2m')
                        temp_c = temp_k - 273.15 if temp_k is not None else None
                        
                        # Convert precipitation from m to mm
                        precip_m = props.get('total_precipitation')
                        precip_mm = precip_m * 1000 if precip_m is not None else None
                        
                        data_row = {
                            'datetime': timestamp,
                            'temperature_k': temp_k,
                            'temperature_c': temp_c,
                            'precipitation_m': precip_m,
                            'precipitation_mm': precip_mm,
                            'u_wind_10m': u_wind,
                            'v_wind_10m': v_wind,
                            'wind_speed_10m': wind_speed,
                            'surface_pressure': props.get('surface_pressure')
                        }
                        
                        all_data.append(data_row)
                
                # Move to next month
                current_date = next_month
                
                # Brief pause to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"    Error in chunk {current_date} to {chunk_end}: {str(e)}")
                # Continue to next chunk rather than failing the entire station
                current_date = next_month
                continue
        
        # Create dataframe from ERA5 data
        if all_data:
            # Create DataFrame
            era5_df = pd.DataFrame(all_data)
            
            # Set datetime as index
            era5_df.set_index('datetime', inplace=True)
            
            # Sort by time
            era5_df = era5_df.sort_index()
            
            # Merge with sea level data if available
            if sea_level_hourly is not None:
                # Both datasets are now in UTC, so we can merge directly
                merged_df = pd.merge(
                    era5_df,
                    sea_level_hourly.to_frame('sea_level'),
                    left_index=True,
                    right_index=True,
                    how='outer'
                )
                
                # Calculate additional variables
                # 3-hour cumulative precipitation
                if 'precipitation_mm' in merged_df.columns:
                    merged_df['precip_3hr_mm'] = merged_df['precipitation_mm'].rolling(window=3).sum()
                
                # Save merged data
                merged_df.index = merged_df.index.tz_localize(None)  # Remove timezone info for CSV
                merged_df.to_csv(output_file)
                
                sea_level_coverage = merged_df['sea_level'].notna().sum() / len(merged_df) * 100
                print(f"  Saved {len(merged_df)} hourly records to {output_file}")
                print(f"  Sea level data coverage: {sea_level_coverage:.2f}%")
                return f"Saved {len(merged_df)} records with {sea_level_coverage:.1f}% sea level coverage"
            else:
                # Just save ERA5 data without sea level
                era5_df.index = era5_df.index.tz_localize(None)  # Remove timezone info for CSV
                era5_df.to_csv(output_file)
                print(f"  Saved {len(era5_df)} hourly ERA5 records to {output_file} (no sea level data)")
                return f"Saved {len(era5_df)} ERA5 records (no sea level data)"
        else:
            return "No data extracted - time range likely outside ERA5 availability"

    def extract_era5_to_drive(self, station_code, station, start_date, end_date):
        """Extract ERA5 data and export to Google Drive (without sea level merging)."""
        lat = station['latitude']
        lon = station['longitude']

        using_original_point = True
        
        # Convert dates to EE format
        start_date_ee = ee.Date(start_date.strftime('%Y-%m-%dT%H:%M:%S'))
        end_date_ee = ee.Date(end_date.strftime('%Y-%m-%dT%H:%M:%S'))
        
        # Create point geometry
        original_point = ee.Geometry.Point(lon, lat)
        
        # Check if station is on land or needs nearest land point
        sample_point = original_point  # Start with original point
        found_land = False
        
        # Try to get sample data for the first date to check if we're on land
        try:
            test_date = ee.Date(start_date.strftime('%Y-%m-%dT%H:%M:%S'))
            test_end = ee.Date(test_date.advance(1, 'day'))
            
            # Get a sample image
            test_img = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
                .filterDate(test_date, test_end) \
                .first()
            
            # Try to sample at the exact point
            test_values = test_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=original_point,
                scale=11132,
                bestEffort=True
            ).getInfo()
            
            if not any(test_values.values()):
                print(f"  Station appears to be in ocean. Trying nearby land points...")
                
                # Define offsets to try (in degrees)
                offsets = [
                    (0.05, 0),    # East
                    (-0.05, 0),   # West
                    (0, 0.05),    # North
                    (0, -0.05),   # South
                    (0.05, 0.05), # Northeast
                    (-0.05, 0.05), # Northwest
                    (0.05, -0.05), # Southeast
                    (-0.05, -0.05) # Southwest
                ]
                
                for offset_lon, offset_lat in offsets:
                    # Create test point with offset
                    test_lon = lon + offset_lon
                    test_lat = lat + offset_lat
                    test_point = ee.Geometry.Point(test_lon, test_lat)
                    
                    # Test this point with reduceRegion
                    test_values = test_img.reduceRegion(
                        reducer=ee.Reducer.first(),
                        geometry=test_point,
                        scale=11132,
                        bestEffort=True
                    ).getInfo()
                    
                    # Check if we got any data values
                    if any(test_values.values()):
                        sample_point = test_point
                        using_original_point = False  # We're using an offset point
                        distance = self.haversine_distance(lat, lon, test_lat, test_lon)
                        print(f"  Found land at offset point: {test_lat}, {test_lon} (Distance: {distance:.2f} km)")
                        found_land = True
                        break
                
                # If still no data found, try larger offsets
                if not found_land:
                    print(f"  WARNING: No land found with standard offsets. Trying larger offsets...")
                    
                    # Try larger offsets (in degrees)
                    larger_offsets = [
                        (0.1, 0),     # East
                        (-0.1, 0),    # West
                        (0, 0.1),     # North
                        (0, -0.1),    # South
                        (0.1, 0.1),   # Northeast
                        (-0.1, 0.1),  # Northwest
                        (0.1, -0.1),  # Southeast
                        (-0.1, -0.1)  # Southwest
                    ]
                    
                    for offset_lon, offset_lat in larger_offsets:
                        # Create test point with offset
                        test_lon = lon + offset_lon
                        test_lat = lat + offset_lat
                        test_point = ee.Geometry.Point(test_lon, test_lat)
                        
                        # Test this point with reduceRegion
                        test_values = test_img.reduceRegion(
                            reducer=ee.Reducer.first(),
                            geometry=test_point,
                            scale=11132,
                            bestEffort=True
                        ).getInfo()
                        
                        # Check if we got any data values
                        if any(test_values.values()):
                            sample_point = test_point
                            distance = self.haversine_distance(lat, lon, test_lat, test_lon)
                            print(f"  Found land at offset point: {test_lat}, {test_lon} (Distance: {distance:.2f} km)")
                            found_land = True
                            break
            else:
                found_land = True
                print(f"  Station is on land. Using original coordinates.")
        except Exception as e:
            print(f"  Error checking land/ocean status: {e}")
        
        if not found_land:
            print(f"  WARNING: No land found within search radius. Data may be unavailable.")
            return "Error: No land found near station coordinates with valid ERA5 data"
        
        # Get ERA5 collection for the specified time range and variables
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
            .filterDate(start_date_ee, end_date_ee) \
            .select(['total_precipitation', 'u_component_of_wind_10m', 
                    'v_component_of_wind_10m', 'surface_pressure', 'temperature_2m'])
        
        # Create a feature collection with ERA5 data at the sample point
        def extract_at_point(image):
            # Get timestamp (in milliseconds)
            timestamp = image.get('system:time_start')
            
            # Extract values at the point using reduceRegion
            values = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=sample_point,
                scale=11132,
                bestEffort=True
            )
            
            # Create a feature with just the values (no geometry)
            # First create a dictionary with timestamp
            properties = ee.Dictionary().set('timestamp', timestamp)
            
            # Add each band's value to the properties
            properties = properties.combine(values)
            
            # Create feature with NULL geometry to avoid .geo field in output
            return ee.Feature(None, properties)
        
        # Map the function over the collection
        point_collection = era5.map(extract_at_point)
        
        # Set a description that includes station name and coordinates
        description = f'ERA5_for_{station_code}'
        
        # Export to Google Drive
        task = ee.batch.Export.table.toDrive(
            collection=point_collection,
            description=description,
            folder='GESLA_ERA5',
            fileNamePrefix=f'{station_code}_ERA5',
            fileFormat='CSV',
            selectors=['system:index', 'total_precipitation', 'u_component_of_wind_10m', 
                        'v_component_of_wind_10m', 'surface_pressure', 'temperature_2m']
        )
        
        # Start the export task
        task.start()
        
        location_info = "at station coordinates" if using_original_point else "at nearest land point"
        
        return f"Export started - task ID: {task.id} ({location_info})"

# Main execution
if __name__ == "__main__":
    # Get base path - current directory
    base_path = os.path.abspath(os.getcwd())
    print(f"Using base path: {base_path}")
    
    # Initialize the processor
    processor = EfficientCompoundFloodingProcessor(base_path)
    
    # Initialize the ERA5 extractor
    min_years = 20
    extractor = GESLA_ERA5_Extractor(processor, min_years=min_years)
    
    # Add command line argument support for resume option
    import argparse
    parser = argparse.ArgumentParser(description='Extract ERA5 data for GESLA stations')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--export-to-drive', action='store_true', help='Export to Google Drive')
    args = parser.parse_args()
    
    print(f"Running with resume={args.resume}")
    
    # Process all stations
    results = extractor.process_all_stations(
        export_to_drive=args.export_to_drive, 
        resume=args.resume
    )
    
    # Print overall status
    print("\nExtraction completed!")
    print(f"Results saved to {os.path.join(base_path, 'ERA5_data')}")