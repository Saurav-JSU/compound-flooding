"""
Module: bn_diagnostics.py
Responsibilities:
- Evaluate Bayesian Network performance
- Perform k-fold cross-validation
- Calculate Brier scores for probabilistic predictions
- Conduct sensitivity analysis to identify influential variables
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.metrics import log_likelihood_score
from pgmpy.inference import VariableElimination
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

# Import from bn_core
from src.compound_flooding.bn_core import (
    NODE_STATES, COMPOUND_FLOOD_RISK
)
from src.compound_flooding.bn_learning import fit_cpts

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kfold_cv(
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    k: int = 5,
    method: str = 'bayes',
    pseudocount: float = 1.0,
    target_node: str = COMPOUND_FLOOD_RISK,
    random_state: Optional[int] = None,
    stratify: bool = True  # New parameter for stratified CV
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for a Bayesian Network.
    
    Parameters
    ----------
    model : BayesianNetwork
        The Bayesian Network structure to evaluate
    data : pd.DataFrame
        Discretized data for cross-validation
    k : int, optional
        Number of folds for cross-validation
    method : str, optional
        Estimation method: 'mle' for Maximum Likelihood or 'bayes' for Bayesian estimation
    pseudocount : float, optional
        Pseudocount for Bayesian estimation
    target_node : str, optional
        The target node for prediction evaluation, defaults to COMPOUND_FLOOD_RISK
    random_state : int, optional
        Random state for reproducibility
    stratify : bool, optional
        Whether to use stratified k-fold cross-validation, defaults to True
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with cross-validation results including:
        - log_likelihood: Mean log-likelihood across folds
        - accuracy: Mean accuracy across folds
        - brier_score: Mean Brier score across folds
        - fold_results: Detailed results for each fold
        
    Notes
    -----
    For each fold, the function:
    1. Trains the model on the training set
    2. Evaluates the model on the test set
    3. Computes log-likelihood, accuracy, and Brier score
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize k-fold cross-validation
    if stratify and target_node in data.columns:
        # Use StratifiedKFold to ensure all risk levels appear in train and test sets
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        split_indices = kf.split(data, data[target_node])
        logger.info(f"Using stratified {k}-fold cross-validation")
    else:
        # Fall back to regular KFold if stratification is not requested or target_node is not in data
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        split_indices = kf.split(data)
        if stratify and target_node not in data.columns:
            logger.warning(f"Cannot use stratified CV: target node {target_node} not in data. Falling back to standard KFold.")
        else:
            logger.info(f"Using standard {k}-fold cross-validation")
    
    # Initialize results storage
    fold_results = []
    log_likelihoods = []
    accuracies = []
    brier_scores = []
    
    # Ensure target node is in the model
    if target_node not in model.nodes:
        raise ValueError(f"Target node {target_node} not found in the model")
    
    # Get the possible states for the target node
    target_states = NODE_STATES.get(target_node, [])
    if not target_states:
        raise ValueError(f"No states defined for target node {target_node}")
    
    # Perform k-fold cross-validation
    logger.info(f"Performing {k}-fold cross-validation")
    for fold, (train_idx, test_idx) in enumerate(split_indices):
        logger.debug(f"Processing fold {fold+1}/{k}")
        
        # Split data into training and test sets
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Check class distribution in train and test sets if stratifying
        if stratify and target_node in data.columns:
            logger.debug(f"Train set {target_node} distribution: {train_data[target_node].value_counts().to_dict()}")
            logger.debug(f"Test set {target_node} distribution: {test_data[target_node].value_counts().to_dict()}")
        
        # Fit the model on the training data
        try:
            train_model = model.copy()
            train_model = fit_cpts(
                train_model, train_data, method=method, 
                pseudocount=pseudocount, random_state=random_state
            )
            
            # Create an inference engine
            inference = VariableElimination(train_model)
            
            # Evaluate on test data
            log_likelihood = 0.0
            correct_predictions = 0
            brier_score_sum = 0.0
            
            for _, test_row in test_data.iterrows():
                # Create evidence dictionary from all nodes except the target
                evidence = {
                    node: test_row[node]
                    for node in train_model.nodes
                    if node != target_node and node in test_row
                }
                
                # Perform inference to get probability distribution for target node
                try:
                    query_result = inference.query(variables=[target_node], evidence=evidence)
                    predicted_probs = query_result.values
                    
                    # Compute log-likelihood
                    actual_state = test_row[target_node]
                    actual_idx = target_states.index(actual_state)
                    if 0 <= actual_idx < len(predicted_probs):
                        prob = predicted_probs[actual_idx]
                        if prob > 0:
                            log_likelihood += np.log(prob)
                    
                    # Get the most likely state
                    predicted_idx = np.argmax(predicted_probs)
                    predicted_state = target_states[predicted_idx]
                    
                    # Check if prediction is correct
                    if predicted_state == actual_state:
                        correct_predictions += 1
                    
                    # Compute Brier score component
                    # Create one-hot encoding of actual state
                    actual_one_hot = np.zeros(len(target_states))
                    actual_one_hot[actual_idx] = 1
                    
                    # Calculate squared difference between predicted probs and actual one-hot
                    brier_score_sum += np.sum((predicted_probs - actual_one_hot) ** 2)
                    
                except Exception as e:
                    logger.warning(f"Inference error for row: {e}")
                    continue
            
            # Calculate metrics for this fold
            accuracy = correct_predictions / len(test_data) if len(test_data) > 0 else 0
            mean_log_likelihood = log_likelihood / len(test_data) if len(test_data) > 0 else float('-inf')
            mean_brier_score = brier_score_sum / len(test_data) if len(test_data) > 0 else float('inf')
            
            # Store results for this fold
            fold_results.append({
                'fold': fold + 1,
                'log_likelihood': mean_log_likelihood,
                'accuracy': accuracy,
                'brier_score': mean_brier_score,
                'train_size': len(train_data),
                'test_size': len(test_data)
            })
            
            # Update aggregate metrics
            log_likelihoods.append(mean_log_likelihood)
            accuracies.append(accuracy)
            brier_scores.append(mean_brier_score)
            
            logger.debug(f"Fold {fold+1} results: log_likelihood={mean_log_likelihood:.4f}, "
                         f"accuracy={accuracy:.4f}, brier_score={mean_brier_score:.4f}")
        
        except Exception as e:
            logger.error(f"Error in fold {fold+1}: {e}")
            continue
    
    # Calculate mean metrics across all folds
    mean_log_likelihood = np.mean(log_likelihoods) if log_likelihoods else float('-inf')
    mean_accuracy = np.mean(accuracies) if accuracies else 0
    mean_brier_score = np.mean(brier_scores) if brier_scores else float('inf')
    
    logger.info(f"Cross-validation complete: mean_log_likelihood={mean_log_likelihood:.4f}, "
                f"mean_accuracy={mean_accuracy:.4f}, mean_brier_score={mean_brier_score:.4f}")
    
    # Return results
    return {
        'log_likelihood': mean_log_likelihood,
        'accuracy': mean_accuracy,
        'brier_score': mean_brier_score,
        'fold_results': fold_results
    }

def brier_score(
    model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    target_node: str = COMPOUND_FLOOD_RISK
) -> float:
    """
    Calculate the Brier score for probabilistic predictions.
    
    Parameters
    ----------
    model : BayesianNetwork
        Trained Bayesian Network model
    data : pd.DataFrame
        Discretized test data
    target_node : str, optional
        The target node for prediction evaluation
        
    Returns
    -------
    float
        Brier score (lower is better)
        
    Notes
    -----
    The Brier score measures the accuracy of probabilistic predictions.
    It is the mean squared difference between predicted probabilities
    and the actual outcomes (represented as one-hot vectors).
    """
    # Ensure target node is in the model
    if target_node not in model.nodes:
        raise ValueError(f"Target node {target_node} not found in the model")
    
    # Get the possible states for the target node
    target_states = NODE_STATES.get(target_node, [])
    if not target_states:
        raise ValueError(f"No states defined for target node {target_node}")
    
    # Create an inference engine
    inference = VariableElimination(model)
    
    brier_score_sum = 0.0
    n_valid = 0
    
    # Evaluate each data point
    for _, row in data.iterrows():
        # Create evidence dictionary from all nodes except the target
        evidence = {
            node: row[node]
            for node in model.nodes
            if node != target_node and node in row
        }
        
        # Skip if target node value is missing
        if target_node not in row or pd.isna(row[target_node]):
            continue
        
        # Perform inference to get probability distribution for target node
        try:
            query_result = inference.query(variables=[target_node], evidence=evidence)
            predicted_probs = query_result.values
            
            # Get the actual state and create one-hot encoding
            actual_state = row[target_node]
            actual_idx = target_states.index(actual_state)
            actual_one_hot = np.zeros(len(target_states))
            actual_one_hot[actual_idx] = 1
            
            # Calculate squared difference between predicted probs and actual one-hot
            brier_score_sum += np.sum((predicted_probs - actual_one_hot) ** 2)
            n_valid += 1
            
        except Exception as e:
            logger.warning(f"Inference error: {e}")
            continue
    
    # Calculate mean Brier score
    mean_brier_score = brier_score_sum / n_valid if n_valid > 0 else float('inf')
    
    logger.info(f"Brier score: {mean_brier_score:.4f} (calculated on {n_valid} valid samples)")
    
    return mean_brier_score

def sensitivity(
    model: DiscreteBayesianNetwork,
    target_node: str = COMPOUND_FLOOD_RISK,
    target_state: Optional[str] = None,
    n_samples: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Perform sensitivity analysis to identify influential variables.
    
    Parameters
    ----------
    model : BayesianNetwork
        Trained Bayesian Network model
    target_node : str, optional
        The target node for sensitivity analysis
    target_state : str, optional
        Specific state of the target node to analyze, if None uses all states
    n_samples : int, optional
        Number of samples to generate for analysis
    random_state : int, optional
        Random state for reproducibility
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping parent nodes to their influence metrics
        
    Notes
    -----
    This function:
    1. Randomly samples evidence configurations
    2. For each parent of the target node, measures the change in target probability
       when the parent state is varied
    3. Quantifies the sensitivity as the average change in probability
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Ensure target node is in the model
    if target_node not in model.nodes:
        raise ValueError(f"Target node {target_node} not found in the model")
    
    # Get the possible states for the target node
    target_states = NODE_STATES.get(target_node, [])
    if not target_states:
        raise ValueError(f"No states defined for target node {target_node}")
    
    # If target_state is specified, ensure it's valid
    if target_state is not None and target_state not in target_states:
        raise ValueError(f"Target state {target_state} not found in {target_states}")
    
    # Use all states if target_state is not specified
    target_indices = [target_states.index(target_state)] if target_state else range(len(target_states))
    
    # Get the parents of the target node
    parents = list(model.get_parents(target_node))
    
    # Verify and fix CPTs if they don't sum to 1
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is not None:
            values = cpd.values
            if len(values.shape) == 1:
                if abs(np.sum(values) - 1.0) > 1e-6:
                    # Normalize and update
                    values = values / np.sum(values)
                    # Create new CPD with normalized values
                    evidence = cpd.variables[1:] if len(cpd.variables) > 1 else []
                    evidence_card = [model.get_cardinality(v) for v in evidence] if evidence else []
                    new_cpd = TabularCPD(
                        variable=cpd.variable,
                        variable_card=cpd.variable_card,
                        values=values.reshape(cpd.variable_card, -1),
                        evidence=evidence,
                        evidence_card=evidence_card,
                        state_names=cpd.state_names
                    )
                    model.add_cpds(new_cpd)
            else:
                # For multi-dimensional values, check each column
                values_2d = values
                if len(values.shape) > 2:
                    # Reshape multi-dimensional array to 2D
                    evidence = cpd.variables[1:] if len(cpd.variables) > 1 else []
                    evidence_card = [model.get_cardinality(v) for v in evidence] if evidence else []
                    evidence_size = np.prod(evidence_card) if evidence_card else 1
                    values_2d = values.reshape(cpd.variable_card, evidence_size)
                
                # Check if any column doesn't sum to 1
                needs_fixing = False
                for i in range(values_2d.shape[1]):
                    if abs(np.sum(values_2d[:, i]) - 1.0) > 1e-6:
                        needs_fixing = True
                        values_2d[:, i] = values_2d[:, i] / np.sum(values_2d[:, i])
                
                # Update CPD if needed
                if needs_fixing:
                    evidence = cpd.variables[1:] if len(cpd.variables) > 1 else []
                    evidence_card = [model.get_cardinality(v) for v in evidence] if evidence else []
                    new_cpd = TabularCPD(
                        variable=cpd.variable,
                        variable_card=cpd.variable_card,
                        values=values_2d,
                        evidence=evidence,
                        evidence_card=evidence_card,
                        state_names=cpd.state_names
                    )
                    model.add_cpds(new_cpd)
    
    # Create an inference engine
    inference = VariableElimination(model)
    
    # Initialize sensitivity metrics
    sensitivity_metrics = {parent: {state: 0.0 for state in NODE_STATES.get(parent, [])} for parent in parents}
    
    # Generate random evidence configurations
    for _ in range(n_samples):
        # Generate a random evidence configuration (excluding the target and its parents)
        evidence = {}
        for node in model.nodes:
            if node != target_node and node not in parents:
                states = NODE_STATES.get(node, [])
                if states:
                    evidence[node] = np.random.choice(states)
        
        try:
            # Base query: probability of target with random evidence
            base_query = inference.query(variables=[target_node], evidence=evidence)
            base_probs = base_query.values
            
            # Ensure base probabilities sum to 1
            if abs(np.sum(base_probs) - 1.0) > 1e-6:
                base_probs = base_probs / np.sum(base_probs)
            
            # For each parent, measure the impact of changing its state
            for parent in parents:
                parent_states = NODE_STATES.get(parent, [])
                if not parent_states:
                    continue
                
                # Try each state of the parent
                for state in parent_states:
                    # Add parent state to evidence
                    evidence_with_parent = evidence.copy()
                    evidence_with_parent[parent] = state
                    
                    # Query with parent state
                    query_with_parent = inference.query(variables=[target_node], evidence=evidence_with_parent)
                    probs_with_parent = query_with_parent.values
                    
                    # Ensure probabilities sum to 1 (normalize if needed)
                    if abs(np.sum(probs_with_parent) - 1.0) > 1e-6:
                        probs_with_parent = probs_with_parent / np.sum(probs_with_parent)
                    
                    # Calculate absolute change in probability for the target state(s)
                    for idx in target_indices:
                        prob_diff = abs(probs_with_parent[idx] - base_probs[idx])
                        
                        # Update sensitivity metric
                        sensitivity_metrics[parent][state] += prob_diff / len(target_indices)
        except Exception as e:
            logger.warning(f"Error during sensitivity analysis iteration: {e}")
            continue
    
    # Normalize by the number of samples
    for parent in sensitivity_metrics:
        for state in sensitivity_metrics[parent]:
            sensitivity_metrics[parent][state] /= n_samples
    
    # Compute an overall influence score for each parent
    parent_influence = {}
    for parent in parents:
        if sensitivity_metrics[parent]:
            # Average impact across all states
            avg_impact = sum(sensitivity_metrics[parent].values()) / len(sensitivity_metrics[parent])
            max_impact = max(sensitivity_metrics[parent].values())
            
            parent_influence[parent] = {
                'average_impact': avg_impact,
                'max_impact': max_impact,
                'state_impacts': sensitivity_metrics[parent]
            }
    
    # Sort parents by influence
    sorted_influence = dict(sorted(parent_influence.items(), 
                                  key=lambda x: x[1]['average_impact'], 
                                  reverse=True))
    
    logger.info(f"Sensitivity analysis complete for {target_node}")
    for parent, metrics in sorted_influence.items():
        logger.info(f"  {parent}: avg_impact={metrics['average_impact']:.4f}, max_impact={metrics['max_impact']:.4f}")
    
    return sorted_influence

def extract_cpt_statistics(model: DiscreteBayesianNetwork) -> Dict[str, Any]:
    """Extract enhanced CPT statistics from a Bayesian Network model."""
    import scipy.stats as stats
    
    cpt_stats = {}
    
    # Process each node's CPT
    for node in model.nodes():
        node_stats = {}
        
        try:
            # Get the CPD for this node
            cpd = model.get_cpds(node)
            
            if cpd is None:
                continue
                
            # Extract state names
            states = cpd.state_names[node]
            
            # Get the values and ensure proper shape for analysis
            values = cpd.values
            
            # Ensure values is a 2D array
            if len(values.shape) == 1:
                values = values.reshape(-1, 1)
            elif len(values.shape) > 2:
                # For multi-dimensional arrays, reshape to 2D
                # First dimension is the variable card (number of states for the node)
                # Second dimension is product of evidence cards
                evidence = cpd.variables[1:] if len(cpd.variables) > 1 else []
                evidence_card = [model.get_cardinality(v) for v in evidence] if evidence else []
                evidence_size = np.prod(evidence_card) if evidence_card else 1
                variable_card = cpd.variable_card
                values = values.reshape(variable_card, evidence_size)
                logger.debug(f"Reshaped values for {node} from multi-dimensional to 2D: {values.shape}")
            
            # Normalize each column to ensure it sums to 1
            for i in range(values.shape[1]):
                col_sum = np.sum(values[:, i])
                if abs(col_sum - 1.0) > 1e-6:
                    values[:, i] = values[:, i] / col_sum
            
            # Get the parents of the node
            parents = cpd.variables[1:] if len(cpd.variables) > 1 else []
            
            # Most likely states analysis
            most_likely_states = {}
            entropy_values = {}
            
            # If the node has parents
            if parents:
                # Get parent state combinations
                parent_states = []
                
                for parent in parents:
                    if parent in cpd.state_names:
                        parent_states.append(cpd.state_names[parent])
                    else:
                        # Handle missing parent state names
                        logger.warning(f"Parent {parent} state names not found for {node}")
                        parent_states.append([f"State_{i}" for i in range(values.shape[1])])
                
                # Generate all combinations of parent states
                import itertools
                # Check if the parent_states list is not empty and all elements are valid
                if parent_states and all(len(states) > 0 for states in parent_states):
                    parent_combinations = list(itertools.product(*parent_states))
                    
                    # Store parent combinations for reference
                    parent_combos_ref = {}
                    for i, combo in enumerate(parent_combinations):
                        if i < values.shape[1]:  # Ensure we don't exceed array bounds
                            parent_combo_name = "_".join([f"{parent}={state}" for parent, state in zip(parents, combo)])
                            parent_combos_ref[i] = parent_combo_name
                    
                    # For each parent combination, analyze the CPT
                    for i in range(min(values.shape[1], len(parent_combinations))):
                        probs = values[:, i]
                        most_likely_idx = np.argmax(probs)
                        most_likely_state = states[most_likely_idx]
                        max_prob = float(probs[most_likely_idx])
                        
                        # Calculate entropy
                        entropy = float(stats.entropy(probs))
                        
                        # Store results
                        parent_combo = parent_combos_ref.get(i, f"combo_{i}")
                        most_likely_states[parent_combo] = {
                            'state': most_likely_state,
                            'probability': max_prob
                        }
                        entropy_values[parent_combo] = entropy
                else:
                    # Fallback for invalid parent states
                    for i in range(values.shape[1]):
                        probs = values[:, i]
                        most_likely_idx = np.argmax(probs)
                        most_likely_state = states[most_likely_idx]
                        max_prob = float(probs[most_likely_idx])
                        entropy = float(stats.entropy(probs))
                        
                        most_likely_states[f"combo_{i}"] = {
                            'state': most_likely_state,
                            'probability': max_prob
                        }
                        entropy_values[f"combo_{i}"] = entropy
            else:
                # No parents, just analyze marginal distribution
                probs = values.ravel()
                most_likely_idx = np.argmax(probs)
                most_likely_state = states[most_likely_idx]
                max_prob = float(probs[most_likely_idx])
                
                # Calculate entropy
                entropy = float(stats.entropy(probs))
                
                # Store results
                most_likely_states['marginal'] = {
                    'state': most_likely_state,
                    'probability': max_prob
                }
                entropy_values['marginal'] = entropy
            
            # Calculate parent-child relationships
            parent_influence = {}
            if parents:
                # For each parent, estimate its influence
                for parent_idx, parent in enumerate(parents):
                    if parent_idx >= len(parent_states) or not parent_states[parent_idx]:
                        continue  # Skip if parent states not available
                        
                    # Estimate parent influence using a simplified metric
                    variance_sum = 0
                    
                    # Number of states for this parent
                    parent_state_count = len(parent_states[parent_idx])
                    
                    for child_state_idx in range(len(states)):
                        # Extract probabilities for this child state across all parent states
                        # This is a safe approach that avoids index errors
                        probs_across_parent = []
                        
                        # Only consider valid indices
                        for i in range(values.shape[1]):
                            probs_across_parent.append(values[child_state_idx, i])
                        
                        # Variance of these probabilities is a measure of parent influence
                        if probs_across_parent:
                            variance_sum += np.var(probs_across_parent)
                    
                    parent_influence[parent] = {
                        'influence_score': float(variance_sum / len(states)),
                        'num_states': parent_state_count
                    }
            
            # Store all statistics for this node
            node_stats['most_likely_states'] = most_likely_states
            node_stats['entropy'] = entropy_values
            node_stats['parent_influence'] = parent_influence
            
            cpt_stats[node] = node_stats
        
        except Exception as e:
            logger.warning(f"Error extracting CPT statistics for node {node}: {e}")
            cpt_stats[node] = {"error": str(e)}
    
    return cpt_stats