from src.compound_flooding.visualization.export import create_all_publication_figures
from src.compound_flooding.visualization.base import (
    load_tier1_results, 
    load_tier2_results, 
    load_station_metadata
)

# Load your analysis results
tier1_data = load_tier1_results('outputs/tier1')
tier2_data = load_tier2_results('outputs/tier2')
metadata_df = load_station_metadata('compound_flooding/data/GESLA/usa_metadata.csv')

# Generate publication figures
create_all_publication_figures(
    tier1_data=tier1_data,
    tier2_data=tier2_data,
    metadata_df=metadata_df,
    output_dir='outputs/visualipublication'
)