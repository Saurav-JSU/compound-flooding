import os
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import base64
import json
import re
import logging
from dash.exceptions import PreventUpdate
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("interactive_map_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("interactive_map_app")

# Get the absolute path to the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"Root directory: {ROOT_DIR}")

# Read station list and metadata
def load_station_data():
    # Read station list
    station_list_path = os.path.join(ROOT_DIR, 'station_list.txt')
    with open(station_list_path, 'r') as f:
        station_ids = [line.strip() for line in f.readlines()]
    
    # Read metadata
    metadata_path = os.path.join(ROOT_DIR, 'compound_flooding', 'data', 'GESLA', 'usa_metadata.csv')
    try:
        metadata = pd.read_csv(metadata_path)
        # Filter metadata to include only stations in our station list
        metadata = metadata[metadata['SITE CODE'].isin(station_ids)]
        
        # Create a clean dataframe with required information
        stations_df = pd.DataFrame({
            'station_id': metadata['SITE CODE'],
            'station_name': metadata['SITE NAME'],
            'latitude': metadata['LATITUDE'],
            'longitude': metadata['LONGITUDE']
        })
        
        return stations_df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        # Create a dummy dataframe if metadata can't be loaded
        return pd.DataFrame({
            'station_id': station_ids,
            'station_name': station_ids,
            'latitude': [None] * len(station_ids),
            'longitude': [None] * len(station_ids)
        })

# Find all PNG files for a specific station in a directory and its subdirectories
def find_station_png_files(directory, station_id):
    png_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.png') and station_id in file:
                    png_files.append(os.path.join(root, file))
    return png_files

# Find all PNG files in a directory and its subdirectories
def find_png_files(directory):
    png_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.png'):
                    png_files.append(os.path.join(root, file))
    return png_files

# Read station list
def get_station_ids():
    try:
        station_list_path = os.path.join(ROOT_DIR, 'station_list.txt')
        with open(station_list_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error reading station_list.txt: {e}")
        return []

# Find available visualizations for each station
def get_available_visualizations():
    visualizations = {}
    
    # Get all station IDs
    station_ids = get_station_ids()
    
    # Check tier1 visualizations
    tier1_path = os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'tier1')
    if os.path.exists(tier1_path):
        # Process all stations
        for station_id in station_ids:
            if station_id not in visualizations:
                visualizations[station_id] = {'tier1': [], 'tier2': [], 'tier3': []}
            
            # Find all PNG files for this station in tier1
            station_files = find_station_png_files(tier1_path, station_id)
            visualizations[station_id]['tier1'].extend(station_files)
    
    # Check tier2 visualizations
    tier2_path = os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'tier2')
    if os.path.exists(tier2_path):
        # Process all stations
        for station_id in station_ids:
            if station_id not in visualizations:
                visualizations[station_id] = {'tier1': [], 'tier2': [], 'tier3': []}
            
            # Find all PNG files for this station in tier2
            station_files = find_station_png_files(tier2_path, station_id)
            visualizations[station_id]['tier2'].extend(station_files)
    
    # Check tier3 visualizations
    tier3_path = os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'tier3')
    if os.path.exists(tier3_path):
        for station_id in station_ids:
            if station_id not in visualizations:
                visualizations[station_id] = {'tier1': [], 'tier2': [], 'tier3': []}
            
            station_path = os.path.join(tier3_path, station_id)
            if os.path.isdir(station_path):
                # Find all PNG files in this station's directory
                station_files = find_png_files(station_path)
                visualizations[station_id]['tier3'].extend(station_files)
    
    return visualizations

# Load station data
stations_df = load_station_data()
visualizations = get_available_visualizations()

# Read station list for direct access
station_ids = get_station_ids()

# Create Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define app layout
app.layout = html.Div([
    html.H1("Compound Flooding Interactive Map", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            dcc.Graph(id='usa-map', style={'height': '70vh'})
        ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Station Information"),
            html.Div(id='station-info'),
            
            html.Div([
                html.H3("Visualization Options"),
                dcc.Tabs(id='tier-tabs', value='tier1', children=[
                    dcc.Tab(label='Tier 1', value='tier1'),
                    dcc.Tab(label='Tier 2', value='tier2'),
                    dcc.Tab(label='Tier 3', value='tier3'),
                ]),
                html.Div(id='visualization-options')
            ], id='visualization-container', style={'display': 'none'})
        ], style={'width': '28%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'})
    ]),
    
    html.Div([
        html.H3("Visualization Preview", style={'textAlign': 'center'}),
        html.Div([
            html.Img(
                id='visualization-preview',
                style={
                    'max-width': '100%',
                    'max-height': '70vh',
                    'display': 'block',
                    'margin': '0 auto',
                    'border': '1px solid #ddd'
                }
            )
        ], style={'textAlign': 'center', 'padding': '10px'})
    ], id='preview-container', style={'display': 'none', 'margin-top': '20px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
    
    # Hidden div to store the last clicked button
    dcc.Store(id='last-clicked-button', data=None),
    
    # Debug info - can be removed in production
    html.Div(id='debug-info', style={'margin-top': '20px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #ddd', 'borderRadius': '5px'})
])

# Callback to update the map
@app.callback(
    Output('usa-map', 'figure'),
    Input('usa-map', 'id')
)
def update_map(_):
    # Filter out stations with missing coordinates
    valid_stations = stations_df.dropna(subset=['latitude', 'longitude'])
    
    # Create the map
    fig = px.scatter_mapbox(
        valid_stations,
        lat='latitude',
        lon='longitude',
        hover_name='station_name',
        hover_data={'station_id': True, 'latitude': True, 'longitude': True},
        zoom=3,
        center={'lat': 39.8, 'lon': -98.5},  # Center of USA
        height=600
    )
    
    # Update the map style
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

# Callback to display station info when a point is clicked
@app.callback(
    [Output('station-info', 'children'),
     Output('visualization-container', 'style')],
    [Input('usa-map', 'clickData')]
)
def display_station_info(click_data):
    if click_data is None:
        return html.Div("Click on a station to view details"), {'display': 'none'}
    
    # Get station ID from click data
    station_id = click_data['points'][0]['customdata'][0]
    station_name = click_data['points'][0]['hovertext']
    lat = click_data['points'][0]['customdata'][1]
    lon = click_data['points'][0]['customdata'][2]
    
    # Check if this station has visualizations
    has_visualizations = station_id in visualizations
    
    # Create station info display
    info = html.Div([
        html.P(f"Station ID: {station_id}"),
        html.P(f"Station Name: {station_name}"),
        html.P(f"Latitude: {lat}"),
        html.P(f"Longitude: {lon}"),
        html.P(f"Visualizations Available: {has_visualizations}")
    ])
    
    # Show visualization options if available
    if has_visualizations:
        return info, {'display': 'block'}
    else:
        return info, {'display': 'none'}

# Callback to update visualization options based on selected tier
@app.callback(
    Output('visualization-options', 'children'),
    [Input('tier-tabs', 'value'),
     Input('usa-map', 'clickData')]
)
def update_visualization_options(selected_tier, click_data):
    if click_data is None:
        return []
    
    # Get station ID from click data
    station_id = click_data['points'][0]['customdata'][0]
    logger.info(f"Getting visualization options for station: {station_id}, tier: {selected_tier}")
    
    # For tier1 and tier2, we need to find all visualizations for this station
    if selected_tier in ['tier1', 'tier2']:
        # Get the base path for this tier
        tier_path = os.path.join(ROOT_DIR, 'outputs', 'visualizations', selected_tier)
        logger.info(f"Searching in path: {tier_path}")
        
        # Find all PNG files for this station in this tier
        all_files = []
        if os.path.exists(tier_path):
            for root, dirs, files in os.walk(tier_path):
                for file in files:
                    if file.endswith('.png') and station_id in file:
                        all_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(all_files)} files for station {station_id} in {selected_tier}")
        
        # Group files by parent directory
        files_by_dir = {}
        for file_path in all_files:
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if parent_dir not in files_by_dir:
                files_by_dir[parent_dir] = []
            files_by_dir[parent_dir].append(file_path)
        
        # Create visualization options
        options = []
        for dir_name, files in sorted(files_by_dir.items()):
            # Add directory name as header
            options.append(html.Div(html.H5(dir_name), style={'marginTop': '15px', 'marginBottom': '5px'}))
            
            # Add files
            for i, file_path in enumerate(sorted(files)):
                file_name = os.path.basename(file_path)
                # Remove the .png extension for the button ID to avoid issues
                button_path = file_path[:-4] if file_path.endswith('.png') else file_path
                logger.info(f"Adding button for file: {file_name}, path: {button_path}")
                options.append(
                    html.Div([
                        html.Button(
                            file_name,
                            id={'type': 'viz-button', 'index': button_path},
                            n_clicks=0,
                            style={'margin': '5px', 'padding': '5px', 'width': '100%', 'textAlign': 'left'}
                        )
                    ])
                )
        
        if not options:
            return html.Div(f"No {selected_tier} visualizations found for station {station_id}")
        
        return options
    
    # For tier3, we check the specific station directory
    elif selected_tier == 'tier3':
        tier3_path = os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'tier3', station_id)
        logger.info(f"Searching in tier3 path: {tier3_path}")
        
        if not os.path.exists(tier3_path):
            logger.warning(f"Tier3 path does not exist: {tier3_path}")
            return html.Div(f"No tier3 visualizations found for station {station_id}")
        
        # Get all PNG files in this directory and its subdirectories
        all_files = find_png_files(tier3_path)
        logger.info(f"Found {len(all_files)} files for station {station_id} in tier3")
        
        # Group files by parent directory
        files_by_dir = {}
        for file_path in all_files:
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if parent_dir not in files_by_dir:
                files_by_dir[parent_dir] = []
            files_by_dir[parent_dir].append(file_path)
        
        # Create visualization options
        options = []
        for dir_name, files in sorted(files_by_dir.items()):
            # Add directory name as header if it's not the station directory itself
            if dir_name != station_id:
                options.append(html.Div(html.H5(dir_name), style={'marginTop': '15px', 'marginBottom': '5px'}))
            
            # Add files
            for i, file_path in enumerate(sorted(files)):
                file_name = os.path.basename(file_path)
                # Remove the .png extension for the button ID to avoid issues
                button_path = file_path[:-4] if file_path.endswith('.png') else file_path
                logger.info(f"Adding tier3 button for file: {file_name}, path: {button_path}")
                options.append(
                    html.Div([
                        html.Button(
                            file_name,
                            id={'type': 'viz-button', 'index': button_path},
                            n_clicks=0,
                            style={'margin': '5px', 'padding': '5px', 'width': '100%', 'textAlign': 'left'}
                        )
                    ])
                )
        
        if not options:
            return html.Div(f"No tier3 visualizations found for station {station_id}")
        
        return options
    
    else:
        return html.Div(f"Unknown tier: {selected_tier}")

# Callback to track which button was clicked last
@app.callback(
    Output('last-clicked-button', 'data'),
    [Input({'type': 'viz-button', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State({'type': 'viz-button', 'index': dash.dependencies.ALL}, 'id')]
)
def track_clicked_button(n_clicks, button_ids):
    # Find which button was clicked
    ctx = dash.callback_context
    
    if not ctx.triggered or len(ctx.triggered) == 0:
        logger.info("No button was clicked")
        return None
    
    # Get the triggered input's property ID
    prop_id = ctx.triggered[0]['prop_id']
    logger.info(f"Button clicked: {prop_id}")
    
    # Extract the index from the property ID
    # prop_id format is like: {"type":"viz-button","index":"path/to/file"}.n_clicks
    try:
        # Extract the index part from the property ID
        import re
        match = re.search(r'"index":"([^"]+)"', prop_id)
        if match:
            # Get the file path
            file_path = match.group(1)
            
            # Make sure the path ends with .png
            if not file_path.endswith('.png'):
                file_path = file_path + '.png'
            
            logger.info(f"Selected file path from prop_id: {file_path}")
            logger.info(f"File exists: {os.path.exists(file_path)}")
            return file_path
        
        # If we couldn't extract the path, try the old method as fallback
        if n_clicks and any(n_clicks):
            # Find the index of the button with the highest n_clicks
            max_clicks = max(n_clicks)
            max_clicks_indices = [i for i, clicks in enumerate(n_clicks) if clicks == max_clicks]
            
            # Use the last button with max clicks (most recently clicked)
            if max_clicks_indices:
                max_clicks_idx = max_clicks_indices[-1]
                if max_clicks_idx < len(button_ids):
                    clicked_path = button_ids[max_clicks_idx]['index']
                    
                    # Make sure the path ends with .png
                    if not clicked_path.endswith('.png'):
                        clicked_path = clicked_path + '.png'
                    
                    logger.info(f"Selected file path (fallback): {clicked_path}")
                    logger.info(f"File exists: {os.path.exists(clicked_path)}")
                    return clicked_path
        
        logger.warning(f"Could not determine which button was clicked")
        logger.warning(f"prop_id: {prop_id}")
        logger.warning(f"n_clicks: {n_clicks}")
        return None
    except Exception as e:
        logger.error(f"Error tracking clicked button: {e}")
        logger.error(f"prop_id: {prop_id}")
        return None

# Callback to display the selected visualization
@app.callback(
    [Output('visualization-preview', 'src'),
     Output('visualization-preview', 'style'),
     Output('preview-container', 'style'),
     Output('debug-info', 'children')],
    [Input('last-clicked-button', 'data'),
     Input('tier-tabs', 'value')]
)
def display_visualization(button_id, tier_value):
    # Check if a button was clicked
    if not button_id:
        return None, {'display': 'none'}, {'display': 'none'}, html.Div("No visualization selected")
    
    # The button_id is the file path
    file_path = button_id
    logger.info(f"Attempting to display file: {file_path}")
    
    # Check if file exists and get its size
    file_exists = os.path.exists(file_path)
    file_size = os.path.getsize(file_path) if file_exists else 0
    
    # Debug info
    debug_info = html.Div([
        html.H4("Debug Information"),
        html.P(f"Selected file: {file_path}"),
        html.P(f"Absolute path: {os.path.abspath(file_path)}"),
        html.P(f"File exists: {file_exists}"),
        html.P(f"File size: {file_size} bytes"),
        html.P(f"Current tier: {tier_value}"),
        html.P(f"Root directory: {ROOT_DIR}")
    ])
    
    # Read and encode the image
    try:
        # Ensure the file path is valid
        if not file_exists:
            error_msg = f"File does not exist: {file_path}"
            logger.error(error_msg)
            return None, {'display': 'none'}, {'display': 'block'}, html.Div([html.P(error_msg), debug_info])
        
        # Check if file size is reasonable
        if file_size == 0:
            error_msg = f"File is empty: {file_path}"
            logger.error(error_msg)
            return None, {'display': 'none'}, {'display': 'block'}, html.Div([html.P(error_msg), debug_info])
        
        # Read and encode the image
        with open(file_path, 'rb') as image_file:
            logger.info(f"Successfully opened file: {file_path}")
            file_content = image_file.read()
            logger.info(f"Read {len(file_content)} bytes from file")
            encoded_image = base64.b64encode(file_content).decode('ascii')
            logger.info(f"Successfully encoded image: {file_path}")
            
            # Add timestamp to prevent caching issues
            timestamp = int(time.time())
            
            # Return the image with a timestamp to prevent caching
            image_src = f'data:image/png;base64,{encoded_image}'
            logger.info(f"Image src length: {len(image_src)}")
            
            # Set image display style
            img_style = {
                'max-width': '100%',
                'max-height': '70vh',
                'display': 'block',
                'margin': '0 auto',
                'border': '1px solid #ddd'
            }
            
            return image_src, img_style, {'display': 'block'}, debug_info
    except Exception as e:
        error_msg = f"Error loading image: {e}"
        logger.error(f"{error_msg} - {file_path}")
        return None, {'display': 'none'}, {'display': 'block'}, html.Div([html.P(error_msg), debug_info])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) 