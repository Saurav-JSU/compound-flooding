"""
Module: bn_core.py
Responsibilities:
- Define core Bayesian Network functionality
- Store shared constants for BN structure
- Provide utilities for graph operations
- Save and load BN models
"""
import os
import logging
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for node names
SEA_LEVEL_STATE = "Sea_Level_State"
PRECIPITATION_STATE = "Precipitation_State"
WIND_SPEED = "Wind_Speed"
WIND_DIRECTION = "Wind_Direction"
PRESSURE_ANOMALY = "Pressure_Anomaly"
SEASON = "Season"
COMPOUND_FLOOD_RISK = "Compound_Flood_Risk"

# Constants for node states
SEA_LEVEL_STATES = ["Below_90th", "90th_99th", "Above_99th"]
PRECIPITATION_STATES = ["Below_90th", "90th_99th", "Above_99th"]
WIND_SPEED_STATES = ["Low", "Moderate", "High"]
WIND_DIRECTION_STATES = ["Offshore", "Alongshore", "Onshore"]
PRESSURE_ANOMALY_STATES = ["High", "Normal", "Low"]
SEASON_STATES = ["Winter", "Spring", "Summer", "Fall"]
COMPOUND_FLOOD_RISK_STATES = ["Low", "Moderate", "High"]

# Node state mappings
NODE_STATES = {
    SEA_LEVEL_STATE: SEA_LEVEL_STATES,
    PRECIPITATION_STATE: PRECIPITATION_STATES,
    WIND_SPEED: WIND_SPEED_STATES,
    WIND_DIRECTION: WIND_DIRECTION_STATES,
    PRESSURE_ANOMALY: PRESSURE_ANOMALY_STATES,
    SEASON: SEASON_STATES,
    COMPOUND_FLOOD_RISK: COMPOUND_FLOOD_RISK_STATES,
}

# Default network structure as defined in the design
DEFAULT_EDGES = [
    (SEASON, PRESSURE_ANOMALY),
    (PRESSURE_ANOMALY, WIND_SPEED),
    (PRESSURE_ANOMALY, WIND_DIRECTION),
    (WIND_SPEED, SEA_LEVEL_STATE),
    (WIND_DIRECTION, SEA_LEVEL_STATE),
    (PRECIPITATION_STATE, COMPOUND_FLOOD_RISK),
    (SEA_LEVEL_STATE, COMPOUND_FLOOD_RISK),
    (WIND_DIRECTION, COMPOUND_FLOOD_RISK),
]

def build_dag(edges: List[Tuple[str, str]] = None) -> DiscreteBayesianNetwork:
    """
    Build a Directed Acyclic Graph (DAG) for the Bayesian Network.
    """
    if edges is None:
        edges = DEFAULT_EDGES
        
    # First check if the edges form a DAG using networkx
    G = nx.DiGraph(edges)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The provided edges create a cyclic graph, not a DAG")
    
    # Create a DiscreteBayesianNetwork with the given edges
    model = DiscreteBayesianNetwork(edges)
    
    logger.info(f"Built DAG with {len(model.nodes())} nodes and {len(model.edges())} edges")
    return model

def save_bn_xmlbif(model: DiscreteBayesianNetwork, output_path: str) -> None:
    """
    Save a Bayesian Network model to XMLBIF format.
    
    Parameters
    ----------
    model : BayesianNetwork
        The pgmpy BayesianNetwork to save
    output_path : str
        Path where the XMLBIF file will be saved
        
    Returns
    -------
    None
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the model to XMLBIF format
    writer = XMLBIFWriter(model)
    writer.write_xmlbif(output_path)
    
    logger.info(f"Saved Bayesian Network to {output_path}")

def load_bn(input_path: str) -> DiscreteBayesianNetwork:
    """
    Load a Bayesian Network model from XMLBIF format.
    
    Parameters
    ----------
    input_path : str
        Path to the XMLBIF file
        
    Returns
    -------
    BayesianNetwork
        The loaded pgmpy BayesianNetwork
        
    Raises
    ------
    FileNotFoundError
        If the input file does not exist
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"BN model file not found: {input_path}")
    
    # Read the model from XMLBIF format
    reader = XMLBIFReader(input_path)
    model = reader.get_model()
    
    logger.info(f"Loaded Bayesian Network from {input_path}")
    logger.debug(f"Model has {len(model.nodes())} nodes and {len(model.edges())} edges")
    
    return model

def node_state_mapping_to_json(node_states: Dict[str, List[str]], thresholds: Dict[str, Dict[str, float]]) -> Dict:
    """
    Convert node state definitions and thresholds to a JSON-serializable format.
    
    Parameters
    ----------
    node_states : Dict[str, List[str]]
        Dictionary mapping node names to their possible states
    thresholds : Dict[str, Dict[str, float]]
        Dictionary mapping node names to threshold dictionaries
        
    Returns
    -------
    Dict
        JSON-serializable dictionary with node state information
    """
    result = {}
    
    for node, states in node_states.items():
        node_info = {"states": states}
        
        # Add thresholds if available
        if node in thresholds:
            node_info["thresholds"] = thresholds[node]
            
        result[node] = node_info
    
    return result