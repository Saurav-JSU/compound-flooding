# src/compound_flooding/copula_selection.py

"""
Module: copula_selection.py
Responsibilities:
- Model selection via AIC, BIC, or other criteria
- Cross-validation for copula selection
- Parallel model fitting and selection
- Ensemble model approaches
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import concurrent.futures
from statsmodels.distributions.copula.api import GaussianCopula

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False
    logger.warning("joblib not available, parallel processing will use concurrent.futures")


def fit_and_compute_criterion(
    name: str,
    fit_func: Callable,
    u: np.ndarray,
    v: np.ndarray,
    criterion: str = 'aic'
) -> Tuple[str, Any, float]:
    """
    Fit a copula and compute selection criterion.
    
    Parameters
    ----------
    name : str
        Name of the copula
    fit_func : Callable
        Function to fit the copula
    u, v : np.ndarray
        Pseudo-observations
    criterion : str, optional
        Selection criterion ('aic', 'bic', 'll')
        
    Returns
    -------
    Tuple[str, Any, float]
        (name, fitted_copula, criterion_value)
    """
    try:
        # Fit the copula
        cop = fit_func(u, v)
        
        # Compute log-likelihood
        uv = np.column_stack((u, v))
        ll = np.sum(cop.logpdf(uv))
        
        # Number of parameters
        k = 2 if name == 'StudentT' else 1
        if name == 'Independence':
            k = 0
            
        # Compute criterion
        n = len(u)
        if criterion.lower() == 'aic':
            value = 2 * k - 2 * ll
        elif criterion.lower() == 'bic':
            value = k * np.log(n) - 2 * ll
        elif criterion.lower() == 'll':
            value = -ll  # Negative log-likelihood (lower is better)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
        return name, cop, value
        
    except Exception as e:
        logger.warning(f"Error fitting {name} copula: {str(e)}")
        return name, None, np.inf


def select_best_copula(
    u: np.ndarray,
    v: np.ndarray,
    candidates: Optional[Dict[str, Callable]] = None,
    criterion: str = 'aic',
    parallel: bool = False,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Select the best copula by AIC or other criterion.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    candidates : Dict[str, Callable], optional
        Dictionary of candidate copulas (name -> fit_function)
    criterion : str, optional
        Selection criterion ('aic', 'bic', 'll')
    parallel : bool, optional
        Whether to fit candidates in parallel
    n_jobs : int, optional
        Number of parallel jobs (-1 for all cores)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with best copula info
    """
    from src.compound_flooding.copula_fit import (
        fit_gumbel, fit_frank, fit_student, fit_gaussian, 
        fit_clayton, fit_independence
    )
    
    # Default candidates if none provided
    if candidates is None:
        candidates = {
            'Gumbel': fit_gumbel,
            'Frank': fit_frank,
            'StudentT': fit_student,
            'Gaussian': fit_gaussian,
            'Clayton': fit_clayton,
            'Independence': fit_independence
        }
    
    # Process in parallel or sequential
    if parallel:
        results = _select_copula_parallel(u, v, candidates, criterion, n_jobs)
    else:
        results = _select_copula_sequential(u, v, candidates, criterion)
    
    # Find best model (minimize criterion)
    valid_results = [(name, cop, value) for name, cop, value in results 
                     if cop is not None and np.isfinite(value)]
    
    if not valid_results:
        logger.warning("No valid copula fits, using independence")
        return {
            'name': 'Independence',
            'copula': GaussianCopula(corr=np.eye(2)),
            'criterion': np.inf,
            'criterion_name': criterion,
            'all_results': {name: (None, np.inf) for name, _, _ in results}
        }
    
    # Find best model (lowest criterion value)
    best_name, best_cop, best_value = min(valid_results, key=lambda x: x[2])
    
    # Create results dictionary
    all_results = {name: (cop, value) for name, cop, value in results}
    
    return {
        'name': best_name,
        'copula': best_cop,
        'criterion': best_value,
        'criterion_name': criterion,
        'all_results': all_results
    }


def _select_copula_sequential(
    u: np.ndarray,
    v: np.ndarray,
    candidates: Dict[str, Callable],
    criterion: str
) -> List[Tuple[str, Any, float]]:
    """
    Sequential implementation of copula selection.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    candidates : Dict[str, Callable]
        Dictionary of candidate copulas
    criterion : str
        Selection criterion
        
    Returns
    -------
    List[Tuple[str, Any, float]]
        List of (name, copula, criterion_value) for each candidate
    """
    results = []
    
    for name, fit_func in candidates.items():
        result = fit_and_compute_criterion(name, fit_func, u, v, criterion)
        results.append(result)
        
    return results


def _select_copula_parallel(
    u: np.ndarray,
    v: np.ndarray,
    candidates: Dict[str, Callable],
    criterion: str,
    n_jobs: int
) -> List[Tuple[str, Any, float]]:
    """
    Parallel implementation of copula selection.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    candidates : Dict[str, Callable]
        Dictionary of candidate copulas
    criterion : str
        Selection criterion
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    List[Tuple[str, Any, float]]
        List of (name, copula, criterion_value) for each candidate
    """
    if HAVE_JOBLIB:
        # Use joblib for parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_and_compute_criterion)(name, fit_func, u, v, criterion)
            for name, fit_func in candidates.items()
        )
    else:
        # Use concurrent.futures
        items = list(candidates.items())
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=(None if n_jobs == -1 else n_jobs)
        ) as executor:
            futures = [
                executor.submit(fit_and_compute_criterion, name, fit_func, u, v, criterion)
                for name, fit_func in items
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return results


def cross_validate_copula(
    u: np.ndarray,
    v: np.ndarray,
    candidates: Optional[Dict[str, Callable]] = None,
    n_folds: int = 5,
    criterion: str = 'aic',
    parallel: bool = False,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    Perform cross-validation for copula selection.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    candidates : Dict[str, Callable], optional
        Dictionary of candidate copulas
    n_folds : int, optional
        Number of cross-validation folds
    criterion : str, optional
        Selection criterion
    parallel : bool, optional
        Whether to process folds in parallel
    n_jobs : int, optional
        Number of parallel jobs
        
    Returns
    -------
    Dict[str, Any]
        Cross-validation results
    """
    from src.compound_flooding.copula_fit import (
        fit_gumbel, fit_frank, fit_student, fit_gaussian, fit_clayton
    )
    
    # Default candidates if none provided
    if candidates is None:
        candidates = {
            'Gumbel': fit_gumbel,
            'Frank': fit_frank,
            'StudentT': fit_student,
            'Gaussian': fit_gaussian,
            'Clayton': fit_clayton
        }
    
    n = len(u)
    if n < n_folds:
        logger.warning(f"Sample size {n} is less than n_folds {n_folds}, using LOO CV")
        n_folds = n
    
    # Create folds
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(n)
    fold_size = n // n_folds
    
    # Initialize results
    cv_results = {name: [] for name in candidates.keys()}
    
    # Process folds
    for i in range(n_folds):
        # Test indices for this fold
        if i < n_folds - 1:
            test_idx = indices[i * fold_size:(i + 1) * fold_size]
        else:
            # Last fold may have more elements
            test_idx = indices[i * fold_size:]
            
        # Train indices (everything except test)
        train_idx = np.setdiff1d(indices, test_idx)
        
        # Split data
        u_train, v_train = u[train_idx], v[train_idx]
        u_test, v_test = u[test_idx], v[test_idx]
        
        # Fit on train, evaluate on test
        for name, fit_func in candidates.items():
            try:
                # Fit on training data
                cop = fit_func(u_train, v_train)
                
                # Evaluate on test data
                uv_test = np.column_stack((u_test, v_test))
                test_ll = np.sum(cop.logpdf(uv_test))
                
                # Number of parameters
                k = 2 if name == 'StudentT' else 1
                
                # Compute criterion
                n_test = len(u_test)
                if criterion.lower() == 'aic':
                    value = 2 * k - 2 * test_ll
                elif criterion.lower() == 'bic':
                    value = k * np.log(n_test) - 2 * test_ll
                elif criterion.lower() == 'll':
                    value = -test_ll
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
                    
                cv_results[name].append(value)
                
            except Exception as e:
                logger.warning(f"Error in fold {i} for {name}: {str(e)}")
                cv_results[name].append(np.inf)
    
    # Compute mean and std of criterion values
    cv_summary = {}
    for name in candidates.keys():
        values = cv_results[name]
        valid_values = [v for v in values if np.isfinite(v)]
        
        if valid_values:
            mean_value = np.mean(valid_values)
            std_value = np.std(valid_values)
        else:
            mean_value = np.inf
            std_value = np.inf
            
        cv_summary[name] = {
            'mean': mean_value,
            'std': std_value,
            'values': values,
            'n_valid': len(valid_values)
        }
    
    # Find best model (lowest mean criterion)
    valid_names = [name for name, stats in cv_summary.items() 
                  if stats['n_valid'] > 0]
    
    if not valid_names:
        logger.warning("No valid models in cross-validation")
        best_name = None
    else:
        best_name = min(valid_names, key=lambda name: cv_summary[name]['mean'])
    
    # Fit final model on all data
    if best_name is not None:
        try:
            best_fit_func = candidates[best_name]
            best_copula = best_fit_func(u, v)
        except Exception as e:
            logger.error(f"Error fitting final {best_name} copula: {str(e)}")
            best_copula = None
    else:
        best_copula = None
    
    return {
        'best_name': best_name,
        'best_copula': best_copula,
        'cv_summary': cv_summary,
        'cv_results': cv_results,
        'n_folds': n_folds,
        'criterion': criterion
    }


def compare_copula_fits(
    u: np.ndarray,
    v: np.ndarray,
    candidates: Optional[Dict[str, Callable]] = None,
    criterion: str = 'aic'
) -> pd.DataFrame:
    """
    Fit and compare multiple copula models.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    candidates : Dict[str, Callable], optional
        Dictionary of candidate copulas
    criterion : str, optional
        Selection criterion
        
    Returns
    -------
    pd.DataFrame
        Comparison of copula fits
    """
    import pandas as pd
    from src.compound_flooding.copula_fit import (
        fit_gumbel, fit_frank, fit_student, fit_gaussian, 
        fit_clayton, compute_tail_dependence
    )
    
    # Default candidates if none provided
    if candidates is None:
        candidates = {
            'Gumbel': fit_gumbel,
            'Frank': fit_frank,
            'StudentT': fit_student,
            'Gaussian': fit_gaussian,
            'Clayton': fit_clayton
        }
    
    # Initialize results
    results = []
    
    # Fit each model
    for name, fit_func in candidates.items():
        try:
            # Fit the model
            cop = fit_func(u, v)
            
            # Compute log-likelihood
            uv = np.column_stack((u, v))
            ll = np.sum(cop.logpdf(uv))
            
            # Number of parameters
            k = 2 if name == 'StudentT' else 1
            
            # Compute metrics
            n = len(u)
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            
            # Get parameters
            params = {}
            if hasattr(cop, 'theta'):
                params['theta'] = cop.theta
            if hasattr(cop, 'corr') and hasattr(cop.corr, 'shape') and cop.corr.shape == (2, 2):
                params['rho'] = cop.corr[0, 1]
            if hasattr(cop, 'df'):
                params['df'] = cop.df
                
            # Compute tail dependence
            lambda_l, lambda_u = compute_tail_dependence(cop)
            
            # Add to results
            result = {
                'name': name,
                'log_likelihood': ll,
                'aic': aic,
                'bic': bic,
                'n_params': k,
                'lambda_lower': lambda_l,
                'lambda_upper': lambda_u
            }
            
            # Add parameters
            for param_name, value in params.items():
                result[f'param_{param_name}'] = value
                
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Error fitting {name} copula: {str(e)}")
            
    # Convert to DataFrame
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    # Sort by selection criterion
    if criterion.lower() == 'aic':
        df = df.sort_values('aic')
    elif criterion.lower() == 'bic':
        df = df.sort_values('bic')
    elif criterion.lower() == 'll':
        df = df.sort_values('log_likelihood', ascending=False)
        
    return df


if __name__ == "__main__":
    # Simple test/demonstration
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau, norm
    import pandas as pd
    
    print("Testing copula selection functions")
    
    # Generate some bivariate data with known dependence
    n = 1000
    rho = 0.7
    
    # Generate correlated normals
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    x, y = np.random.multivariate_normal(mean, cov, n).T
    
    # Convert to uniform
    u = norm.cdf(x)
    v = norm.cdf(y)
    
    # Compute Kendall's tau
    tau, _ = kendalltau(u, v)
    print(f"Kendall's tau: {tau:.4f}")
    
    # Import copula fitting functions
    from src.compound_flooding.copula_fit import (
        fit_gumbel, fit_frank, fit_student, fit_gaussian, fit_clayton
    )
    
    # Define candidates
    candidates = {
        'Gumbel': fit_gumbel,
        'Frank': fit_frank,
        'StudentT': fit_student,
        'Gaussian': fit_gaussian,
        'Clayton': fit_clayton
    }
    
    # Select best copula
    print("\nSelecting best copula by AIC...")
    result_aic = select_best_copula(u, v, candidates, criterion='aic')
    print(f"Best model (AIC): {result_aic['name']}")
    print(f"AIC: {result_aic['criterion']:.4f}")
    
    print("\nSelecting best copula by BIC...")
    result_bic = select_best_copula(u, v, candidates, criterion='bic')
    print(f"Best model (BIC): {result_bic['name']}")
    print(f"BIC: {result_bic['criterion']:.4f}")
    
    # Compare copula fits
    print("\nComparing all copula fits:")
    comp_df = compare_copula_fits(u, v, candidates)
    print(comp_df.to_string(index=False))
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_results = cross_validate_copula(u, v, candidates, n_folds=5)
    print(f"Best model (CV): {cv_results['best_name']}")
    
    # Print CV summary
    print("\nCross-validation summary:")
    cv_summary = cv_results['cv_summary']
    for name in candidates:
        if name in cv_summary:
            stats = cv_summary[name]
            print(f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, valid={stats['n_valid']}/{cv_results['n_folds']}")
    
    # Parallel selection
    print("\nSelecting best copula in parallel...")
    result_parallel = select_best_copula(u, v, candidates, criterion='aic', parallel=True)
    print(f"Best model (parallel): {result_parallel['name']}")
    print(f"AIC: {result_parallel['criterion']:.4f}")
    
    # Plot comparison of models
    plt.figure(figsize=(10, 6))
    metrics = ['aic', 'bic', 'log_likelihood']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        if metric == 'log_likelihood':
            # For log-likelihood, higher is better
            values = comp_df['log_likelihood']
            plt.bar(comp_df['name'], values)
            plt.axhline(values.max(), linestyle='--', color='r')
        else:
            # For AIC/BIC, lower is better
            values = comp_df[metric]
            plt.bar(comp_df['name'], values)
            plt.axhline(values.min(), linestyle='--', color='r')
            
        plt.title(metric.upper())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()