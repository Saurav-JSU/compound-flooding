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
from pgmpy.metrics import log_likelihood_score
from pgmpy.inference import VariableElimination
from sklearn.model_selection import KFold
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
    random_state: Optional[int] = None
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
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
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
    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        logger.debug(f"Processing fold {fold+1}/{k}")
        
        # Split data into training and test sets
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
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
        
        # Base query: probability of target with random evidence
        base_query = inference.query(variables=[target_node], evidence=evidence)
        base_probs = base_query.values
        
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
                
                # Calculate absolute change in probability for the target state(s)
                for idx in target_indices:
                    prob_diff = abs(probs_with_parent[idx] - base_probs[idx])
                    
                    # Update sensitivity metric
                    sensitivity_metrics[parent][state] += prob_diff / len(target_indices)
    
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