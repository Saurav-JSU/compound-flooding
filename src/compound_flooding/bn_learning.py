"""
Module: bn_learning.py
Responsibilities:
- Discretize continuous data for Bayesian Network
- Fit conditional probability tables (CPTs)
- Implement hierarchical parameter learning
- Handle parameter estimation with sparse data
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.factors.discrete import TabularCPD

# Import from bn_core
from src.compound_flooding.bn_core import (
    NODE_STATES, SEA_LEVEL_STATE, PRECIPITATION_STATE, WIND_SPEED,
    WIND_DIRECTION, PRESSURE_ANOMALY, SEASON, COMPOUND_FLOOD_RISK
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def discretise_df(
    df: pd.DataFrame,
    node_mappings: Dict[str, Dict[str, Union[float, List[float]]]],
    random_state: Optional[int] = None,
    handle_missing: str = 'fill'  # New parameter
) -> pd.DataFrame:
    """
    Discretize continuous variables in a DataFrame for Bayesian Network analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with continuous variables
    node_mappings : Dict[str, Dict[str, Union[float, List[float]]]]
        Dictionary mapping node names to their discretization thresholds
    random_state : int, optional
        Random state for reproducibility
    handle_missing : str, optional
        Strategy for missing values: 'drop' to remove rows with any missing values,
        'fill' to use default values for missing data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with discretized variables
    """
    # Create a copy of the DataFrame to avoid modifying the original
    discretized_df = df.copy()
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Track missing values by column
    missing_by_column = {}
    
    # Discretize each variable based on its mapping
    for node, mapping in node_mappings.items():
        if node not in df.columns:
            logger.warning(f"Variable {node} not found in DataFrame")
            continue
            
        # Check how many missing values in this column
        missing_count = df[node].isna().sum()
        missing_by_column[node] = missing_count
        
        if missing_count > 0:
            logger.debug(f"Column {node} has {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")
            
        # Check if node requires discretization
        if node in [SEA_LEVEL_STATE, PRECIPITATION_STATE]:
            # These nodes use 3 states based on percentile thresholds
            if "threshold_90th" in mapping and "threshold_99th" in mapping:
                threshold_90th = mapping["threshold_90th"]
                threshold_99th = mapping["threshold_99th"]
                
                # Create categorical values
                conditions = [
                    df[node] < threshold_90th,
                    (df[node] >= threshold_90th) & (df[node] < threshold_99th),
                    df[node] >= threshold_99th
                ]
                choices = NODE_STATES[node]
                
                discretized_df[node] = np.select(conditions, choices, default=None)
                logger.debug(f"Discretized {node} with thresholds {threshold_90th:.4f} and {threshold_99th:.4f}")
                
        elif node == WIND_SPEED:
            # Wind speed uses 3 states based on different percentile thresholds
            if "threshold_75th" in mapping and "threshold_95th" in mapping:
                threshold_75th = mapping["threshold_75th"]
                threshold_95th = mapping["threshold_95th"]
                
                conditions = [
                    df[node] < threshold_75th,
                    (df[node] >= threshold_75th) & (df[node] < threshold_95th),
                    df[node] >= threshold_95th
                ]
                choices = NODE_STATES[node]
                
                discretized_df[node] = np.select(conditions, choices, default=None)
                logger.debug(f"Discretized {node} with thresholds {threshold_75th:.4f} and {threshold_95th:.4f}")
                
        elif node == WIND_DIRECTION:
            # Wind direction is based on angles relative to coastline
            if "offshore_angles" in mapping and "alongshore_angles" in mapping:
                offshore = mapping["offshore_angles"]
                alongshore = mapping["alongshore_angles"]
                
                # Determine direction based on angle ranges
                # All angles not in offshore or alongshore are considered onshore
                conditions = [
                    df[node].apply(lambda x: any(start <= x <= end for start, end in offshore)),
                    df[node].apply(lambda x: any(start <= x <= end for start, end in alongshore))
                ]
                choices = [NODE_STATES[node][0], NODE_STATES[node][1]]  # Offshore, Alongshore
                
                discretized_df[node] = np.select(conditions, choices, default=NODE_STATES[node][2])  # Default to Onshore
                
        elif node == PRESSURE_ANOMALY:
            # Pressure anomaly is based on deviation from mean
            if "high_threshold" in mapping and "low_threshold" in mapping:
                high_threshold = mapping["high_threshold"]
                low_threshold = mapping["low_threshold"]
                
                conditions = [
                    df[node] > high_threshold,
                    df[node] < low_threshold
                ]
                choices = [NODE_STATES[node][0], NODE_STATES[node][2]]  # High, Low
                
                discretized_df[node] = np.select(conditions, choices, default=NODE_STATES[node][1])  # Default to Normal
                
        elif node == SEASON:
            # Season is derived from datetime
            if "datetime_col" in mapping:
                datetime_col = mapping["datetime_col"]
                
                if datetime_col in df.columns:
                    # Extract month from datetime
                    months = pd.DatetimeIndex(df[datetime_col]).month
                    
                    # Assign seasons (Northern Hemisphere convention)
                    conditions = [
                        (months >= 12) | (months <= 2),  # Winter: Dec-Feb
                        (months >= 3) & (months <= 5),   # Spring: Mar-May
                        (months >= 6) & (months <= 8),   # Summer: Jun-Aug
                        (months >= 9) & (months <= 11)   # Fall: Sep-Nov
                    ]
                    choices = NODE_STATES[node]
                    
                    discretized_df[node] = np.select(conditions, choices, default=None)
                else:
                    logger.warning(f"Datetime column {datetime_col} not found in DataFrame")
        
        elif node == COMPOUND_FLOOD_RISK:
            # This is the target node that will be derived from others
            # We'll leave it empty for now if it's not already in the dataframe
            if node not in df.columns:
                logger.debug(f"Target node {node} will be derived from other variables")
                
    # Handle missing values strategy
    if handle_missing == 'fill':
        # Fill missing values with defaults for each column
        for node in discretized_df.columns:
            if node in NODE_STATES and node in discretized_df.columns:
                # Fill with the first state (usually the most common state)
                default_state = NODE_STATES[node][0]
                missing_mask = discretized_df[node].isna()
                missing_count = missing_mask.sum()
                
                if missing_count > 0:
                    discretized_df.loc[missing_mask, node] = default_state
                    logger.debug(f"Filled {missing_count} missing values in {node} with default state '{default_state}'")
    
    # Log missing value statistics
    na_counts_by_column = discretized_df.isna().sum()
    na_count_before = na_counts_by_column.sum()
    
    if na_count_before > 0:
        logger.debug("Missing values by column after discretization:")
        for col, count in na_counts_by_column.items():
            if count > 0:
                logger.debug(f"  {col}: {count} missing values ({count/len(discretized_df)*100:.2f}%)")
    
    # Remove rows with missing values if any remain
    n_rows_before = len(discretized_df)
    discretized_df = discretized_df.dropna()
    n_rows_after = len(discretized_df)
    rows_dropped = n_rows_before - n_rows_after
    
    if rows_dropped > 0:
        logger.warning(f"Removed {rows_dropped} rows with missing values after discretization")
        logger.warning(f"Remaining data points: {n_rows_after} out of {len(df)} ({n_rows_after/len(df)*100:.2f}%)")
    
    return discretized_df

def compute_quantile_thresholds(
    df: pd.DataFrame, 
    variables: List[str], 
    quantiles: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute station-specific quantile thresholds for variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with continuous variables
    variables : List[str]
        List of variable names to compute thresholds for
    quantiles : Dict[str, List[float]]
        Dictionary mapping variable names to lists of quantiles to compute
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping variable names to threshold dictionaries
        
    Example
    -------
    >>> df = pd.DataFrame({'sea_level': [1, 2, 3, 4, 5]})
    >>> variables = ['sea_level']
    >>> quantiles = {'sea_level': [0.9, 0.99]}
    >>> compute_quantile_thresholds(df, variables, quantiles)
    {'sea_level': {'threshold_90th': 4.6, 'threshold_99th': 4.96}}
    """
    thresholds = {}
    
    for var in variables:
        if var not in df.columns:
            logger.warning(f"Variable {var} not found in DataFrame")
            continue
            
        var_quantiles = quantiles.get(var, [0.9, 0.99])
        var_thresholds = {}
        
        # Compute quantiles
        for q in var_quantiles:
            # Format the quantile as a percentage for the key name
            q_pct = int(q * 100)
            key = f"threshold_{q_pct}th"
            
            try:
                var_thresholds[key] = float(df[var].quantile(q))
            except Exception as e:
                logger.error(f"Error computing {q} quantile for {var}: {e}")
                
        thresholds[var] = var_thresholds
    
    return thresholds
def fit_cpts(
    model: DiscreteBayesianNetwork, 
    data: pd.DataFrame, 
    method: str = 'bayes',
    pseudocount: Optional[float] = 1.0,
    state_names: Optional[Dict[str, List[str]]] = None,  # We'll handle this differently
    random_state: Optional[int] = None
) -> DiscreteBayesianNetwork:
    """Fit conditional probability tables (CPTs) for a Bayesian Network."""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Filter the data to only include columns that correspond to nodes in the model
    model_nodes = list(model.nodes())
    data_columns = list(data.columns)
    
    # Check which model nodes are in the data
    available_nodes = [node for node in model_nodes if node in data_columns]
    missing_nodes = [node for node in model_nodes if node not in data_columns]
    
    if missing_nodes:
        logger.warning(f"The following nodes are missing from the data: {missing_nodes}")
        
    # Filter data to include only node columns
    filtered_data = data[available_nodes].copy()
    
    if filtered_data.empty:
        raise ValueError("No data available after filtering for model nodes")
    
    # Continue with the rest of the function using filtered_data
    if method == 'mle':
        # Maximum Likelihood Estimation
        logger.info("Fitting CPTs using Maximum Likelihood Estimation")
        estimator = MaximumLikelihoodEstimator(model, filtered_data)
        
        # Fit CPTs for each node
        for node in model.nodes():
            if node in available_nodes:
                try:
                    # In pgmpy 1.0.0, we don't use state_names parameter
                    cpd = estimator.estimate_cpd(node)
                    model.add_cpds(cpd)
                    logger.debug(f"Fitted CPT for {node} using MLE")
                except Exception as e:
                    logger.error(f"Error fitting CPT for {node}: {e}")
                    raise
    
    elif method == 'bayes':
        # Bayesian Estimation with Dirichlet prior
        logger.info(f"Fitting CPTs using Bayesian Estimation with pseudocount={pseudocount}")
        estimator = BayesianEstimator(model, filtered_data)
        
        # Fit CPTs for each node
        for node in model.nodes():
            if node in available_nodes:
                try:
                    # In pgmpy 1.0.0, the API is different
                    cpd = estimator.estimate_cpd(node, prior_type='dirichlet', 
                                                 pseudo_counts=pseudocount)
                    model.add_cpds(cpd)
                    logger.debug(f"Fitted CPT for {node} using Bayesian estimation")
                except Exception as e:
                    logger.error(f"Error fitting CPT for {node}: {e}")
                    raise
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mle' or 'bayes'.")
    
    # Check if the model is valid
    try:
        if not model.check_model():
            logger.warning("The model is not valid after fitting CPTs")
        else:
            logger.info("Successfully fitted all CPTs")
    except Exception as e:
        logger.warning(f"Error checking model: {e}")
        
    return model

def hierarchical_fit(
    models: Dict[str, DiscreteBayesianNetwork], 
    datas: Dict[str, pd.DataFrame],
    regional_groups: Dict[str, List[str]],
    method: str = 'bayes',
    pseudocount: float = 1.0,
    state_names: Optional[Dict[str, List[str]]] = None,
    random_state: Optional[int] = None
) -> Dict[str, DiscreteBayesianNetwork]:
    """
    Perform hierarchical parameter learning by sharing information across similar stations.
    
    Parameters
    ----------
    models : Dict[str, BayesianNetwork]
        Dictionary mapping station codes to BayesianNetwork models
    datas : Dict[str, pd.DataFrame]
        Dictionary mapping station codes to discretized data
    regional_groups : Dict[str, List[str]]
        Dictionary mapping region names to lists of station codes
    method : str, optional
        Estimation method: 'mle' for Maximum Likelihood or 'bayes' for Bayesian estimation
    pseudocount : float, optional
        Base pseudocount for Bayesian estimation
    state_names : Dict[str, List[str]], optional
        Dictionary mapping node names to their possible states
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    Dict[str, BayesianNetwork]
        Dictionary mapping station codes to fitted BayesianNetwork models
        
    Notes
    -----
    This function implements a hierarchical approach where stations in the same region
    share information. For each station, the CPTs are estimated using data from all
    stations in the same region, weighted by the pseudocount parameter.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if state_names is None:
        state_names = NODE_STATES
    
    # Create a reverse mapping from station to region
    station_to_region = {}
    for region, stations in regional_groups.items():
        for station in stations:
            station_to_region[station] = region
    
    fitted_models = {}
    
    # Process each station
    for station, model in models.items():
        logger.info(f"Hierarchical fitting for station {station}")
        
        # Get the region for this station
        region = station_to_region.get(station)
        if region is None:
            logger.warning(f"Station {station} is not assigned to any region, using local data only")
            # Fit using only local data
            fitted_models[station] = fit_cpts(
                model, datas[station], method=method, 
                pseudocount=pseudocount, state_names=state_names,
                random_state=random_state
            )
            continue
        
        # Get all stations in the same region
        regional_stations = regional_groups[region]
        logger.debug(f"Station {station} belongs to region {region} with {len(regional_stations)} stations")
        
        # For Bayesian estimation with hierarchical priors
        if method == 'bayes':
            # Combine data from all stations in the region with appropriate weights
            # Weight is higher for the target station and lower for other stations
            combined_data = []
            
            for s in regional_stations:
                if s in datas:
                    # If this is the target station, add it with full weight
                    if s == station:
                        combined_data.append(datas[s])
                    # Otherwise, add it with reduced weight based on regional pseudocount
                    else:
                        # Take a subsample proportional to the pseudocount
                        if len(datas[s]) > 0:
                            regional_pseudocount = pseudocount / len(regional_stations)
                            sample_size = max(1, int(len(datas[s]) * regional_pseudocount))
                            if random_state is not None:
                                combined_data.append(datas[s].sample(n=sample_size, random_state=random_state))
                            else:
                                combined_data.append(datas[s].sample(n=sample_size))
            
            # Combine all data
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                
                # Fit the model using the combined data
                fitted_models[station] = fit_cpts(
                    model, combined_df, method=method, 
                    pseudocount=pseudocount, state_names=state_names,
                    random_state=random_state
                )
            else:
                logger.warning(f"No data available for station {station} or its region")
                fitted_models[station] = model
        
        # For MLE, just use the local data
        elif method == 'mle':
            logger.warning("Hierarchical fitting is not supported for MLE, using local data only")
            fitted_models[station] = fit_cpts(
                model, datas[station], method=method, 
                pseudocount=pseudocount, state_names=state_names,
                random_state=random_state
            )
    
    return fitted_models