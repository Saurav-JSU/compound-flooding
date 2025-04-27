# src/compound_flooding/copula_utils.py

"""
Module: copula_utils.py
Responsibilities:
- Main entry points for copula analysis operations
- Delegates to specialized modules for implementation
- Provides simplified API for common copula operations
- Maintains backward compatibility where needed
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

try:
    from statsmodels.distributions.copula.api import (
        GumbelCopula,
        FrankCopula,
        StudentTCopula,
        GaussianCopula,
        IndependenceCopula
    )
except ImportError:
    logging.warning("statsmodels copula API not available. Some functionality may be limited.")

# Import from specialized helper modules
from src.compound_flooding.copula_params import (
    estimate_theta_gumbel,
    estimate_theta_frank,
    estimate_corr_student_t,
    estimate_rho_gaussian
)

from src.compound_flooding.copula_fit import (
    _validate_pobs,
    create_pseudo_observations,
    fit_gumbel,
    fit_frank,
    fit_student,
    fit_gaussian,
    fit_independence,
    compute_joint_exceedance,
    compute_conditional_exceedance
)

from src.compound_flooding.copula_selection import (
    select_best_copula
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fit_copula(
    input_nc: str,
    u_var: str,
    v_var: str,
    method: str,
    output_json: str
) -> None:
    """
    Open a single NetCDF, extract u_var and v_var, fit a copula, and write JSON.
    
    Parameters
    ----------
    input_nc : str
        Path to NetCDF file
    u_var : str
        First variable name
    v_var : str
        Second variable name
    method : str
        Which copula to fit ('auto', 'gumbel', 'frank', 'student', 'gaussian')
    output_json : str
        Output JSON file path
    """
    import xarray as xr
    import os
    import json
    from scipy.stats import kendalltau
    
    logger.info(f"Opening dataset {input_nc}...")
    ds = xr.open_dataset(input_nc)

    var1 = ds[u_var].values
    var2 = ds[v_var].values

    # drop NaNs and Infs
    valid = np.isfinite(var1) & np.isfinite(var2)
    n_drop = len(var1) - valid.sum()
    if n_drop:
        logger.warning(f"Dropping {n_drop} missing or invalid values before copula fitting.")
    u, v = var1[valid], var2[valid]
    
    # Check if we have enough valid data
    if len(u) < 30:
        raise ValueError(f"Insufficient valid data points ({len(u)}) for copula fitting. Minimum required: 30")

    # pseudo‐obs conversion
    u_p, v_p = create_pseudo_observations(u, v)
    logger.info("Converting to pseudo-observations...")

    # fit copula
    logger.info(f"Fitting copula (method={method})...")
    from src.compound_flooding.copula_fit import fit_copula_with_metrics
    
    fit_result = fit_copula_with_metrics(u_p, v_p, method=method)
    cop = fit_result['copula']
    method_used = fit_result['method']

    # compute metrics
    uv = np.column_stack((u_p, v_p))
    tau, _ = kendalltau(u, v, nan_policy='omit')

    # information criterion
    aic = fit_result.get('aic')
    log_lik = fit_result.get('log_likelihood')

    # Extract parameters
    params = fit_result.get('params', {})
    
    # Get tail dependence
    tail_dep = fit_result.get('tail_dependence', {})

    result = {
        'copula': method_used,
        'params': params,
        'kendall_tau': float(tau),
        'n_obs': len(u),
        'log_likelihood': log_lik,
        'aic': float(aic) if aic is not None else None
    }
    
    # Add tail dependence if available
    if tail_dep:
        result['tail_dependence'] = tail_dep
    
    # Add joint exceedance at 95% level
    try:
        joint_exc = compute_joint_exceedance(cop, 0.95, 0.95)
        indep_exc = (1 - 0.95) * (1 - 0.95)
        cpr = joint_exc / indep_exc if indep_exc > 0 else np.nan
        
        result['joint_exceedance'] = {
            'level': 0.95,
            'probability': float(joint_exc),
            'independent_probability': float(indep_exc),
            'cpr': float(cpr)
        }
    except Exception as e:
        logger.warning(f"Failed to compute joint exceedance: {e}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    # Save result
    with open(output_json, 'w') as f:
        json.dump(convert_numpy(result), f, indent=2)

    logger.info(f"Results saved to {output_json}")
    
    # Close dataset
    ds.close()


# Legacy function for backward compatibility
def _validate_pobs_legacy(u, v):
    """
    Legacy function maintained for backward compatibility.
    Delegates to the new implementation in copula_fit module.
    """
    return _validate_pobs(u, v)


# Extended smoke test
if __name__ == '__main__':
    print("Running enhanced copula_utils tests...")
    np.random.seed(0)
    
    # Test on random uniforms (should pick Gaussian independence)
    u = np.random.rand(200)
    v = np.random.rand(200)
    
    print("\nTesting individual copula fits on independent data:")
    for func in [fit_gumbel, fit_frank, fit_student, fit_gaussian]:
        cop = func(u, v)
        print(f"{cop.__class__.__name__} logpdf mean: {np.mean(cop.logpdf(np.column_stack((u, v)))): .4f}")
    
    # Test AIC selection
    selected = select_best_copula(u, v)
    print(f"\nAuto-selected copula for independent data: {selected['name']}")
    
    # Test on correlated data
    print("\nTesting on correlated data:")
    # Generate correlated normals
    n = 1000
    rho_true = 0.7
    
    mean = [0, 0]
    cov = [[1, rho_true], [rho_true, 1]]
    x, y = np.random.multivariate_normal(mean, cov, n).T
    
    # Convert to uniform
    from scipy.stats import norm
    u_corr = norm.cdf(x)
    v_corr = norm.cdf(y)
    
    # Test fits
    for func in [fit_gumbel, fit_frank, fit_student, fit_gaussian]:
        cop = func(u_corr, v_corr)
        print(f"{cop.__class__.__name__} logpdf mean: {np.mean(cop.logpdf(np.column_stack((u_corr, v_corr)))): .4f}")
    
    # Test AIC selection
    selected = select_best_copula(u_corr, v_corr)
    print(f"\nAuto-selected copula for correlated data: {selected['name']}")
    
    # Test tail dependence and joint probabilities
    cop = selected['copula']
    from src.compound_flooding.copula_params import compute_tail_dependence
    lambda_l, lambda_u = compute_tail_dependence(cop)
    print(f"\nTail dependence - Lower: {lambda_l:.4f}, Upper: {lambda_u:.4f}")
    
    # Test joint exceedance
    joint_prob = compute_joint_exceedance(cop, 0.95, 0.95)
    indep_prob = 0.05 * 0.05
    print(f"Joint exceedance P(U>0.95, V>0.95): {joint_prob:.6f}")
    print(f"Independent case: {indep_prob:.6f}")
    print(f"CPR: {joint_prob/indep_prob:.4f}")
    
    # Test conditional exceedance
    cond_prob = compute_conditional_exceedance(cop, 0.5, 0.95, conditional_var='v|u')
    print(f"Conditional P(V>0.95 | U=0.5): {cond_prob:.6f}")
    
    print("\nAll enhanced tests completed.")