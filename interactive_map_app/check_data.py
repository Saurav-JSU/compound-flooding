import os
import pandas as pd
import sys

def check_station_data():
    """Check if station data can be loaded correctly"""
    print("Checking station data...")
    
    # Check if station list exists
    if not os.path.exists('station_list.txt'):
        print("ERROR: station_list.txt not found!")
        return False
    
    try:
        # Read station list
        with open('station_list.txt', 'r') as f:
            station_ids = [line.strip() for line in f.readlines()]
        print(f"Found {len(station_ids)} stations in station_list.txt")
    except Exception as e:
        print(f"ERROR reading station_list.txt: {e}")
        return False
    
    # Check if metadata exists
    metadata_path = os.path.join('compound_flooding', 'data', 'GESLA', 'usa_metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"WARNING: Metadata file not found at {metadata_path}")
        return True
    
    # Read metadata
    try:
        metadata = pd.read_csv(metadata_path)
        # Filter metadata to include only stations in our station list
        filtered_metadata = metadata[metadata['SITE CODE'].isin(station_ids)]
        print(f"Found {len(filtered_metadata)} stations in metadata (out of {len(metadata)} total)")
        
        # Check for missing coordinates
        missing_coords = filtered_metadata[filtered_metadata['LATITUDE'].isna() | filtered_metadata['LONGITUDE'].isna()]
        if len(missing_coords) > 0:
            print(f"WARNING: {len(missing_coords)} stations have missing coordinates")
            
        return True
    except Exception as e:
        print(f"ERROR reading metadata: {e}")
        return False

def find_png_files(directory):
    """Find all PNG files in a directory and its subdirectories"""
    png_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.png'):
                    png_files.append(os.path.join(root, file))
    return png_files

def find_station_png_files(directory, station_id):
    """Find all PNG files for a specific station in a directory and its subdirectories"""
    png_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.png') and station_id in file:
                    png_files.append(os.path.join(root, file))
    return png_files

def check_visualizations():
    """Check if visualizations can be found"""
    print("\nChecking visualizations...")
    
    # Get station IDs
    try:
        with open('station_list.txt', 'r') as f:
            station_ids = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"ERROR reading station_list.txt: {e}")
        return False
    
    # Check tier1 visualizations
    tier1_path = os.path.join('outputs', 'visualizations', 'tier1')
    if not os.path.exists(tier1_path):
        print(f"WARNING: Tier1 directory not found at {tier1_path}")
        tier1_files = []
    else:
        tier1_files = find_png_files(tier1_path)
        print(f"Found {len(tier1_files)} tier1 visualization files")
        
        # Check subdirectories
        tier1_subdirs = [d for d in os.listdir(tier1_path) if os.path.isdir(os.path.join(tier1_path, d))]
        print(f"Tier1 subdirectories: {', '.join(tier1_subdirs)}")
        
        # Count files in each subdirectory
        for subdir in tier1_subdirs:
            subdir_path = os.path.join(tier1_path, subdir)
            subdir_files = find_png_files(subdir_path)
            print(f"  - {subdir}: {len(subdir_files)} files")
        
        # Check station-specific files
        station_files_count = 0
        for station_id in station_ids:
            station_files = find_station_png_files(tier1_path, station_id)
            if station_files:
                station_files_count += 1
        print(f"Found tier1 visualizations for {station_files_count} stations")
    
    # Check tier2 visualizations
    tier2_path = os.path.join('outputs', 'visualizations', 'tier2')
    if not os.path.exists(tier2_path):
        print(f"WARNING: Tier2 directory not found at {tier2_path}")
        tier2_files = []
    else:
        tier2_files = find_png_files(tier2_path)
        print(f"Found {len(tier2_files)} tier2 visualization files")
        
        # Check subdirectories
        tier2_subdirs = [d for d in os.listdir(tier2_path) if os.path.isdir(os.path.join(tier2_path, d))]
        print(f"Tier2 subdirectories: {', '.join(tier2_subdirs)}")
        
        # Count files in each subdirectory
        for subdir in tier2_subdirs:
            subdir_path = os.path.join(tier2_path, subdir)
            subdir_files = find_png_files(subdir_path)
            print(f"  - {subdir}: {len(subdir_files)} files")
        
        # Check station-specific files
        station_files_count = 0
        for station_id in station_ids:
            station_files = find_station_png_files(tier2_path, station_id)
            if station_files:
                station_files_count += 1
        print(f"Found tier2 visualizations for {station_files_count} stations")
    
    # Check tier3 visualizations
    tier3_path = os.path.join('outputs', 'visualizations', 'tier3')
    if not os.path.exists(tier3_path):
        print(f"WARNING: Tier3 directory not found at {tier3_path}")
        tier3_dirs = []
    else:
        tier3_dirs = [d for d in os.listdir(tier3_path) if os.path.isdir(os.path.join(tier3_path, d))]
        print(f"Found {len(tier3_dirs)} tier3 station directories")
        
        # Check if all stations in station_list have tier3 directories
        missing_stations = []
        for station_id in station_ids:
            if station_id not in tier3_dirs:
                missing_stations.append(station_id)
        
        if missing_stations:
            print(f"WARNING: {len(missing_stations)} stations in station_list.txt don't have tier3 directories")
        
        # Count total tier3 visualization files
        tier3_files_count = 0
        stations_with_files = 0
        for station_dir in tier3_dirs:
            station_path = os.path.join(tier3_path, station_dir)
            files = find_png_files(station_path)
            tier3_files_count += len(files)
            if len(files) > 0:
                stations_with_files += 1
        
        print(f"Found {tier3_files_count} tier3 visualization files across {stations_with_files} stations")
    
    # Check maps visualizations
    maps_path = os.path.join('outputs', 'visualizations', 'maps')
    if not os.path.exists(maps_path):
        print(f"WARNING: Maps directory not found at {maps_path}")
        maps_files = []
    else:
        maps_files = find_png_files(maps_path)
        print(f"Found {len(maps_files)} map visualization files")
    
    # Check publication visualizations
    pub_path = os.path.join('outputs', 'visualizations', 'publication')
    if not os.path.exists(pub_path):
        print(f"WARNING: Publication directory not found at {pub_path}")
        pub_files = []
    else:
        pub_files = find_png_files(pub_path)
        print(f"Found {len(pub_files)} publication visualization files")
    
    # Check events visualizations
    events_path = os.path.join('outputs', 'visualizations', 'events')
    if not os.path.exists(events_path):
        print(f"WARNING: Events directory not found at {events_path}")
        events_files = []
    else:
        events_files = find_png_files(events_path)
        print(f"Found {len(events_files)} events visualization files")
    
    # Calculate total visualizations
    total_files = len(tier1_files) + len(tier2_files) + tier3_files_count + len(maps_files) + len(pub_files) + len(events_files)
    print(f"\nTotal visualization files found: {total_files}")
    
    return True

if __name__ == "__main__":
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    print(f"Current working directory: {os.getcwd()}")
    
    station_check = check_station_data()
    viz_check = check_visualizations()
    
    if station_check and viz_check:
        print("\nAll checks completed. The app should be able to run.")
    else:
        print("\nSome checks failed. The app may not work correctly.")
        sys.exit(1) 