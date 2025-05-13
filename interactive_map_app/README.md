# Compound Flooding Interactive Map

This interactive web application displays stations from the compound flooding analysis on a map of the USA. Users can click on stations to view available visualizations from different tiers of analysis.

## Features

- Interactive map of USA showing all stations
- Station information display when a station is clicked
- Visualization browser organized by tier (Tier 1, Tier 2, Tier 3)
- Image preview for selected visualizations
- Automatic detection of all PNG files in all subdirectories for each station

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the application directory:

```bash
cd interactive_map_app
```

2. Run the application:

```bash
python app.py
```

3. Open a web browser and go to:

```
http://localhost:8050
```

## Data Structure

The application expects the following data structure:

- `station_list.txt`: List of station IDs
- `compound_flooding/data/GESLA/usa_metadata.csv`: Metadata file with station coordinates
- `outputs/visualizations/tier1/`: Tier 1 visualizations (including subdirectories like stations, extremes, joint)
- `outputs/visualizations/tier2/`: Tier 2 visualizations (including subdirectories like stations, dependence, joint_returns, copulas)
- `outputs/visualizations/tier3/`: Tier 3 station visualizations organized by station ID
- `outputs/visualizations/maps/`: Maps showing all stations
- `outputs/visualizations/publication/`: Publication-ready visualizations
- `outputs/visualizations/events/`: Event-specific visualizations

## Visualization Types

### Station-specific Visualizations
These are organized by station ID and can be accessed by clicking on a station on the map:
- Tier 1: Basic statistics and time series
- Tier 2: Dependency analysis and return periods
- Tier 3: Detailed analysis and model results

## Troubleshooting

### Checking Data and Visualizations

You can run the check_data.py script to verify that all data and visualizations are correctly detected:

```bash
python interactive_map_app/check_data.py
```

### Common Issues

1. **Callback errors when clicking on visualizations**:
   - Make sure the visualization files exist and are readable
   - Check that the file paths are correct
   - Restart the application after making changes

2. **Missing images or stations**:
   - Verify that all directories exist with the expected structure
   - Check that the station_list.txt file contains all the expected station IDs
   - Ensure that the metadata file has coordinates for all stations

3. **No stations appear on the map**:
   - Check that the metadata file is correctly formatted
   - Verify that the stations have valid latitude and longitude coordinates

4. **Application crashes on startup**:
   - Check the console for error messages
   - Verify that all required packages are installed
   - Make sure you're running the application from the correct directory

5. **Not all visualizations are showing**:
   - Check that the PNG files have the station ID in their filename (for tier1 and tier2)
   - For tier3, make sure the files are in the correct station directory
   - Run the check_data.py script to verify that all visualizations are detected

## Notes

- The application will automatically detect available visualizations for each station and in all subdirectories
- Visualizations are grouped by their parent directory for easier navigation
- If station metadata cannot be loaded, the application will still run but without coordinates
- Make sure the `assets` directory is in the same directory as `app.py` for proper styling 