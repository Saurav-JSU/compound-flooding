# src/compound_flooding/copula_fit.py

"""
Module: copula_fit.py
Responsibilities:
- Robust fitting of various copula families
- Computing joint probabilities and return periods
- Computing conditional exceedance probabilities
- Handling fit failures with appropriate fallbacks
- Computing tail dependence coefficients
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings

from statsmodels.distributions.copula.api import (
    GumbelCopula,
    FrankCopula,
    StudentTCopula,
    GaussianCopula,
    ClaytonCopula,
    IndependenceCopula
)

from src.compound_flooding.copula_params import (
    estimate_theta_gumbel,
    estimate_theta_frank,
    estimate_corr_student_t,
    estimate_rho_gaussian,
    estimate_clayton_theta,
    compute_tail_dependence
)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _validate_pobs(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate pseudo-observations u and v.
    - Checks for consistent shapes
    - Ensures values are in (0, 1)
    - Checks for NaN values
    - Ensures there is sufficient variability

    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations to validate

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Validated pseudo-observations

    Raises
    ------
    ValueError
        If inputs fail validation criteria
    """
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Check shapes
    if u.shape != v.shape:
        raise ValueError(f"u and v must have same shape, got {u.shape} vs {v.shape}")
    
    # Check for NaN values
    if np.isnan(u).any() or np.isnan(v).any():
        raise ValueError("Inputs contain NaN values")
    
    # Check for values outside (0, 1)
    if np.any(u <= 0) or np.any(u >= 1) or np.any(v <= 0) or np.any(v >= 1):
        raise ValueError("Pseudo-observations must lie in the open interval (0, 1)")
    
    # Check for sufficient variability
    if len(np.unique(u)) < 3 or len(np.unique(v)) < 3:
        raise ValueError("Inputs have insufficient variability")
    
    return u, v


def create_pseudo_observations(
    x: np.ndarray, 
    y: np.ndarray, 
    ties_method: str = 'average'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw data to pseudo-observations (uniform ranks).
    
    Parameters
    ----------
    x, y : np.ndarray
        Raw data arrays
    ties_method : str, optional
        Method for handling ties ('average', 'min', 'max', 'dense', 'ordinal')
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pseudo-observations (uniform ranks)
    """
    # Validate inputs
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")
    
    # Check for NaN values
    x_nan = np.isnan(x)
    y_nan = np.isnan(y)
    
    if x_nan.any() or y_nan.any():
        # Find valid indices (non-NaN in both series)
        valid = ~(x_nan | y_nan)
        if not valid.any():
            raise ValueError("All data points contain at least one NaN")
        
        # Extract valid data
        x_valid = x[valid]
        y_valid = y[valid]
    else:
        x_valid = x
        y_valid = y
    
    # Create DataFrame for ranking
    n = len(x_valid)
    df = pd.DataFrame({'x': x_valid, 'y': y_valid})
    
    # Convert to ranks and scale to (0,1)
    u = df['x'].rank(method=ties_method) / (n + 1)
    v = df['y'].rank(method=ties_method) / (n + 1)
    
    # Avoid exact 0 or 1 values (causes issues with some copulas)
    eps = 1e-10
    u = u.clip(eps, 1-eps)
    v = v.clip(eps, 1-eps)
    
    return u.values, v.values


def fit_independence(
    u: np.ndarray, 
    v: np.ndarray, 
    validate: bool = True
) -> IndependenceCopula:
    """
    Fit an independence copula (returns fixed model).
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations (not used, included for API consistency)
    validate : bool, optional
        Whether to validate inputs
        
    Returns
    -------
    IndependenceCopula
        The independence copula
    """
    if validate:
        u, v = _validate_pobs(u, v)
    
    return IndependenceCopula()


def fit_gumbel(
    u: np.ndarray, 
    v: np.ndarray, 
    validate: bool = True, 
    method: str = 'kendall'
) -> GumbelCopula:
    """
    Fit a Gumbel copula to (u, v).
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    validate : bool, optional
        Whether to validate inputs
    method : str, optional
        Method for parameter estimation ('kendall', 'mle')
        
    Returns
    -------
    GumbelCopula
        Fitted Gumbel copula
        
    Notes
    -----
    For negative dependence, returns a copula with theta near 1 (independence).
    """
    if validate:
        u, v = _validate_pobs(u, v)
    
    # For Gumbel, we primarily use the Kendall's tau method
    # MLE could be added in the future
    if method == 'mle':
        logger.warning("MLE estimation not implemented for Gumbel copula, using 'kendall' instead")
    
    from scipy.stats import kendalltau
    tau, _ = kendalltau(u, v)
    
    # Handle NaN tau (can occur with constant data)
    if np.isnan(tau):
        logger.warning("NaN tau detected in fit_gumbel, using independence")
        tau = 0.0
    
    theta = estimate_theta_gumbel(tau)
    
    # Create the copula
    try:
        return GumbelCopula(theta=theta)
    except Exception as e:
        logger.error(f"Failed to create Gumbel copula with theta={theta}: {str(e)}")
        # Fallback to independence
        return GumbelCopula(theta=1.01)


def fit_frank(
    u: np.ndarray, 
    v: np.ndarray, 
    validate: bool = True, 
    method: str = 'kendall'
) -> Union[FrankCopula, GaussianCopula]:
    """
    Fit a Frank copula to (u, v).
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    validate : bool, optional
        Whether to validate inputs
    method : str, optional
        Method for parameter estimation ('kendall', 'mle')
        
    Returns
    -------
    Union[FrankCopula, GaussianCopula]
        Fitted Frank copula, or Gaussian independence copula on failure
        
    Notes
    -----
    Falls back to independence Gaussian copula if theta == 0 or fitting fails.
    """
    if validate:
        u, v = _validate_pobs(u, v)
    
    # For Frank, we primarily use the Kendall's tau method
    if method == 'mle':
        logger.warning("MLE estimation not implemented for Frank copula, using 'kendall' instead")
    
    from scipy.stats import kendalltau
    tau, _ = kendalltau(u, v)
    
    # Handle NaN tau (can occur with constant data)
    if np.isnan(tau):
        logger.warning("NaN tau detected in fit_frank, using independence")
        tau = 0.0
    
    theta = estimate_theta_frank(tau)
    
    # If theta is very close to zero, use independence
    if abs(theta) < 1e-6:
        return GaussianCopula(corr=np.eye(2))
    
    # Create the copula
    try:
        return FrankCopula(theta=theta)
    except Exception as e:
        logger.error(f"Failed to create Frank copula with theta={theta}: {str(e)}")
        # Fallback to independence
        return GaussianCopula(corr=np.eye(2))


def fit_clayton(
    u: np.ndarray, 
    v: np.ndarray, 
    validate: bool = True, 
    method: str = 'kendall'
) -> Union[ClaytonCopula, GaussianCopula]:
    """
    Fit a Clayton copula to (u, v).
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    validate : bool, optional
        Whether to validate inputs
    method : str, optional
        Method for parameter estimation ('kendall', 'mle')
        
    Returns
    -------
    Union[ClaytonCopula, GaussianCopula]
        Fitted Clayton copula, or Gaussian independence copula on failure
        
    Notes
    -----
    Falls back to independence Gaussian copula if theta <= 0 (except theta=-1)
    or if fitting fails.
    """
    if validate:
        u, v = _validate_pobs(u, v)
    
    # For Clayton, we primarily use the Kendall's tau method
    if method == 'mle':
        logger.warning("MLE estimation not implemented for Clayton copula, using 'kendall' instead")
    
    from scipy.stats import kendalltau
    tau, _ = kendalltau(u, v)
    
    # Handle NaN tau (can occur with constant data)
    if np.isnan(tau):
        logger.warning("NaN tau detected in fit_clayton, using independence")
        tau = 0.0
    
    theta = estimate_clayton_theta(tau)
    
    # If theta is very close to zero, use independence
    if abs(theta) < 1e-6:
        return GaussianCopula(corr=np.eye(2))
    
    # Create the copula
    try:
        return ClaytonCopula(theta=theta)
    except Exception as e:
        logger.error(f"Failed to create Clayton copula with theta={theta}: {str(e)}")
        # Fallback to independence
        return GaussianCopula(corr=np.eye(2))


def fit_student(
    u: np.ndarray, 
    v: np.ndarray, 
    validate: bool = True, 
    method: str = 'kendall',
    df: float = 5.0
) -> Union[StudentTCopula, GaussianCopula]:
    """
    Fit a Student-T copula to (u, v).
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    validate : bool, optional
        Whether to validate inputs
    method : str, optional
        Method for correlation estimation ('kendall', 'spearman', 'pearson')
    df : float, optional
        Degrees of freedom (fixed parameter, not estimated)
        
    Returns
    -------
    Union[StudentTCopula, GaussianCopula]
        Fitted Student-T copula, or Gaussian copula on failure
        
    Notes
    -----
    Falls back to Gaussian copula with the same correlation if StudentT fails.
    Future enhancement: estimate df via MLE.
    """
    if validate:
        u, v = _validate_pobs(u, v)
    
    # Estimate correlation matrix
    try:
        corr, _ = estimate_corr_student_t(u, v, df_init=df, method=method)
    except Exception as e:
        logger.error(f"Error estimating Student-T correlation: {str(e)}")
        return GaussianCopula(corr=np.eye(2))
    
    # Handle case where corr is not positive definite
    if not np.all(np.linalg.eigvals(corr) > 0):
        logger.warning("Non-positive definite correlation matrix in fit_student")
        return GaussianCopula(corr=np.eye(2))
    
    # Handle degenerate data
    if len(np.unique(u)) < 3 or len(np.unique(v)) < 3:
        logger.warning("Degenerate data detected in fit_student")
        return GaussianCopula(corr=np.eye(2))
    
    # Create the copula
    try:
        return StudentTCopula(corr=corr, df=df)
    except Exception as e:
        logger.warning(f"Failed to create Student-T copula: {str(e)}")
        # Fallback to Gaussian with same correlation
        try:
            return GaussianCopula(corr=corr)
        except Exception:
            # Ultimate fallback
            return GaussianCopula(corr=np.eye(2))


def fit_gaussian(
    u: np.ndarray, 
    v: np.ndarray, 
    validate: bool = True,
    method: str = 'kendall'
) -> GaussianCopula:
    """
    Fit a Gaussian copula to (u, v).
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    validate : bool, optional
        Whether to validate inputs
    method : str, optional
        Method for correlation estimation ('kendall', 'spearman', 'pearson')
        
    Returns
    -------
    GaussianCopula
        Fitted Gaussian copula, or independence copula on failure
    """
    if validate:
        u, v = _validate_pobs(u, v)
    
    # Estimate correlation
    try:
        rho = estimate_rho_gaussian(u, v, method=method)
    except Exception as e:
        logger.error(f"Error estimating Gaussian correlation: {str(e)}")
        return GaussianCopula(corr=np.eye(2))
    
    # Handle NaN rho
    if np.isnan(rho):
        logger.warning("NaN correlation detected in fit_gaussian, using independence")
        return GaussianCopula(corr=np.eye(2))
    
    # Create correlation matrix
    corr = np.array([[1.0, rho], [rho, 1.0]])
    
    # Create the copula
    try:
        return GaussianCopula(corr=corr)
    except Exception as e:
        logger.warning(f"Failed to create Gaussian copula: {str(e)}")
        # Fallback to independence
        return GaussianCopula(corr=np.eye(2))


def fit_with_fallbacks(
    u: np.ndarray, 
    v: np.ndarray, 
    fit_func: Callable, 
    fallback_sequence: List[Callable]
) -> Any:
    """
    Apply a fitting function with progressive fallbacks on failure.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    fit_func : Callable
        Primary fitting function
    fallback_sequence : List[Callable]
        Sequence of fallback fitting functions to try on failure
        
    Returns
    -------
    Any
        Fitted copula from primary function or fallbacks
    """
    try:
        return fit_func(u, v)
    except Exception as e:
        logger.warning(f"Primary fitting failed: {str(e)}")
        
        # Try fallbacks in sequence
        for i, fallback in enumerate(fallback_sequence):
            try:
                return fallback(u, v)
            except Exception as e2:
                logger.warning(f"Fallback {i+1} failed: {str(e2)}")
                
        # Ultimate fallback: independence copula
        return GaussianCopula(corr=np.eye(2))


def compute_copula_likelihood(
    cop: Any, 
    u: np.ndarray, 
    v: np.ndarray
) -> float:
    """
    Compute log-likelihood of data under a copula.
    
    Parameters
    ----------
    cop : Any
        Fitted copula object
    u, v : np.ndarray
        Pseudo-observations
        
    Returns
    -------
    float
        Log-likelihood sum
    """
    try:
        uv = np.column_stack((u, v))
        ll = np.sum(cop.logpdf(uv))
        return ll
    except Exception as e:
        logger.error(f"Error computing copula likelihood: {str(e)}")
        return -np.inf

def compute_joint_exceedance(
    copula: Any, 
    u_levels: Union[float, np.ndarray], 
    v_levels: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute joint exceedance probability P(U>u, V>v) for a fitted copula.
    
    Parameters
    ----------
    copula : Any
        Fitted copula object
    u_levels, v_levels : Union[float, np.ndarray]
        Uniform levels for exceedance (single value or array)
        
    Returns
    -------
    Union[float, np.ndarray]
        Joint exceedance probability or array of probabilities
        
    Notes
    -----
    Uses the survival copula formula: P(U>u, V>v) = 1 - u - v + C(u,v)
    Falls back to Monte Carlo sampling if CDF is not available in closed form.
    """
    # Handle scalar inputs
    u_is_scalar = np.isscalar(u_levels)
    v_is_scalar = np.isscalar(v_levels)
    
    # Convert to numpy arrays
    u = np.atleast_1d(u_levels)
    v = np.atleast_1d(v_levels)
    
    # Check if inputs are valid
    if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
        logger.warning(f"Invalid probability levels: u_levels={u_levels}, v_levels={v_levels}")
        if u_is_scalar and v_is_scalar:
            return 0.0
        else:
            return np.zeros_like(u if len(u) >= len(v) else v)
    
    # Define a function to compute the joint exceedance via Monte Carlo if CDF fails
    def compute_mc_joint_exceedance(copula, u_val, v_val, n_samples=10000):
        try:
            # Generate samples from the copula
            samples = copula.rvs(n_samples)
            u_samples, v_samples = samples[:, 0], samples[:, 1]
            
            # Count joint exceedances
            joint_exceed = (u_samples > u_val) & (v_samples > v_val)
            prob = np.mean(joint_exceed)
            return prob
        except Exception as e:
            logger.error(f"Monte Carlo joint exceedance failed: {e}")
            return 0.0
    
    try:
        # Try using the copula's CDF method first
        
        # Single pair case (both scalars)
        if len(u) == 1 and len(v) == 1:
            try:
                # Try using the analytical CDF
                uv = np.array([[u[0], v[0]]])
                c_uv = copula.cdf(uv)[0]
                joint_exc = 1.0 - u[0] - v[0] + c_uv
            except Exception as e:
                # If analytical CDF fails, use Monte Carlo
                logger.warning(f"CDF not available in closed form, using Monte Carlo: {e}")
                joint_exc = compute_mc_joint_exceedance(copula, u[0], v[0])
                
            return float(max(0.0, joint_exc)) if u_is_scalar and v_is_scalar else np.array([max(0.0, joint_exc)])
        
        # Multiple pairs case
        else:
            # For multiple values, handle each pair individually for robustness
            if len(u) == len(v):
                result = np.zeros(len(u))
                for i in range(len(u)):
                    try:
                        # Try analytical CDF
                        c_uv = copula.cdf(np.array([[u[i], v[i]]]))[0]
                        result[i] = max(0.0, 1.0 - u[i] - v[i] + c_uv)
                    except Exception:
                        # Fall back to Monte Carlo
                        result[i] = compute_mc_joint_exceedance(copula, u[i], v[i])
                return result
            
            # Handle one scalar, one array case
            elif len(u) == 1:
                result = np.zeros(len(v))
                u_val = u[0]
                for i in range(len(v)):
                    try:
                        c_uv = copula.cdf(np.array([[u_val, v[i]]]))[0]
                        result[i] = max(0.0, 1.0 - u_val - v[i] + c_uv)
                    except Exception:
                        result[i] = compute_mc_joint_exceedance(copula, u_val, v[i])
                return result
                
            elif len(v) == 1:
                result = np.zeros(len(u))
                v_val = v[0]
                for i in range(len(u)):
                    try:
                        c_uv = copula.cdf(np.array([[u[i], v_val]]))[0]
                        result[i] = max(0.0, 1.0 - u[i] - v_val + c_uv)
                    except Exception:
                        result[i] = compute_mc_joint_exceedance(copula, u[i], v_val)
                return result
                
            # Different length arrays
            else:
                # Create all combinations and compute individually
                result = np.zeros((len(u), len(v)))
                for i in range(len(u)):
                    for j in range(len(v)):
                        try:
                            c_uv = copula.cdf(np.array([[u[i], v[j]]]))[0]
                            result[i, j] = max(0.0, 1.0 - u[i] - v[j] + c_uv)
                        except Exception:
                            result[i, j] = compute_mc_joint_exceedance(copula, u[i], v[j])
                return result
    
    except Exception as e:
        logger.error(f"Error computing joint exceedance: {str(e)}")
        # Return appropriate shape based on inputs
        if u_is_scalar and v_is_scalar:
            return 0.0
        elif u_is_scalar:
            return np.zeros_like(v)
        elif v_is_scalar:
            return np.zeros_like(u)
        else:
            return np.zeros((len(u), len(v)))

def compute_joint_return_period(
    copula: Any,
    u_level: float,
    v_level: float,
    type: str = 'and'
) -> float:
    """
    Compute joint return period for a bivariate event.
    
    Parameters
    ----------
    copula : Any
        Fitted copula object
    u_level, v_level : float
        Uniform levels for exceedance
    type : str, optional
        Type of joint event: 'and' (both exceed) or 'or' (either exceeds)
        
    Returns
    -------
    float
        Joint return period
        
    Notes
    -----
    For 'and' type: T_AND = 1/P(U>u, V>v)
    For 'or' type: T_OR = 1/P(U>u or V>v) = 1/(P(U>u) + P(V>v) - P(U>u, V>v))
    """
    if u_level < 0 or u_level > 1 or v_level < 0 or v_level > 1:
        raise ValueError("Levels must be in [0, 1]")
        
    # Probability that both exceed threshold
    joint_and_prob = compute_joint_exceedance(copula, u_level, v_level)
    
    # For AND case, return period is inverse of joint probability
    if type.lower() == 'and':
        if joint_and_prob <= 0:
            return float('inf')  # Infinite return period
        return 1.0 / joint_and_prob
        
    # For OR case, we need probability that either exceeds
    elif type.lower() == 'or':
        # P(U>u or V>v) = P(U>u) + P(V>v) - P(U>u, V>v)
        prob_u = 1.0 - u_level
        prob_v = 1.0 - v_level
        joint_or_prob = prob_u + prob_v - joint_and_prob
        
        if joint_or_prob <= 0:
            return float('inf')  # Infinite return period
        return 1.0 / joint_or_prob
        
    else:
        raise ValueError(f"Unknown joint return period type: {type}. Use 'and' or 'or'.")

def compute_conditional_exceedance(
    copula: Any,
    u_value: float,
    v_level: float,
    conditional_var: str = 'v|u'
) -> float:
    """
    Compute conditional exceedance probability.
    
    Parameters
    ----------
    copula : Any
        Fitted copula object
    u_value : float
        Conditioning value (value that U=u_value or V=u_value)
    v_level : float
        Exceedance level for other variable
    conditional_var : str, optional
        Which conditional to compute: 'v|u' for P(V>v|U=u) or 'u|v' for P(U>u|V=v)
        
    Returns
    -------
    float
        Conditional exceedance probability
    """
    if u_value < 0 or u_value > 1 or v_level < 0 or v_level > 1:
        logger.warning(f"Invalid probability levels: u_value={u_value}, v_level={v_level}")
        return np.nan
    
    # Define a function to estimate conditional probability using Monte Carlo
    def estimate_conditional_mc(copula, u_val, v_val, conditional_var, n_samples=50000):
        try:
            # Generate samples from the copula
            samples = copula.rvs(n_samples)
            u_samples, v_samples = samples[:, 0], samples[:, 1]
            
            # For P(V>v|U=u), find samples where U is close to u
            if conditional_var == 'v|u':
                # Use a small bandwidth around the conditioning value
                bandwidth = 0.02
                u_indices = (u_samples > u_val - bandwidth) & (u_samples < u_val + bandwidth)
                if np.sum(u_indices) == 0:
                    return np.nan
                
                # Compute how many of those have V > v
                prob = np.mean(v_samples[u_indices] > v_val)
                return prob
                
            # For P(U>u|V=v)
            elif conditional_var == 'u|v':
                bandwidth = 0.02
                v_indices = (v_samples > v_val - bandwidth) & (v_samples < v_val + bandwidth)
                if np.sum(v_indices) == 0:
                    return np.nan
                
                prob = np.mean(u_samples[v_indices] > u_val)
                return prob
            
            else:
                raise ValueError(f"Unknown conditional: {conditional_var}")
        except Exception as e:
            logger.error(f"Monte Carlo conditional probability estimation failed: {e}")
            return np.nan
    
    try:
        if conditional_var == 'v|u':
            # We want P(V>v|U=u)
            # First try numerical differentiation if the copula has a CDF
            try:
                # Compute partial derivative of C(u,v) w.r.t. u at the point (u_value, v_level)
                delta = 1e-4
                u_up = min(u_value + delta, 0.9999)
                u_down = max(u_value - delta, 0.0001)
                
                # Compute C(u+δ, v) - C(u-δ, v)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c_up = copula.cdf(np.array([[u_up, v_level]]))[0]
                    c_down = copula.cdf(np.array([[u_down, v_level]]))[0]
                    
                partial_u = (c_up - c_down) / (u_up - u_down)
                
                # h₁(v|u) = ∂C(u,v)/∂u
                h_v_given_u = partial_u
                
                # P(V>v|U=u) = 1 - h₁(v|u)
                return max(0.0, min(1.0, 1.0 - h_v_given_u))
            except Exception as e:
                logger.warning(f"Numerical differentiation failed: {e}, falling back to Monte Carlo")
                return estimate_conditional_mc(copula, u_value, v_level, conditional_var)
                
        elif conditional_var == 'u|v':
            # We want P(U>u|V=v)
            # Similar to above, but with respect to v
            try:
                delta = 1e-4
                v_up = min(v_level + delta, 0.9999)
                v_down = max(v_level - delta, 0.0001)
                
                # Compute C(u, v+δ) - C(u, v-δ)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c_up = copula.cdf(np.array([[u_value, v_up]]))[0]
                    c_down = copula.cdf(np.array([[u_value, v_down]]))[0]
                    
                partial_v = (c_up - c_down) / (v_up - v_down)
                
                # h₂(u|v) = ∂C(u,v)/∂v
                h_u_given_v = partial_v
                
                # P(U>u|V=v) = 1 - h₂(u|v)
                return max(0.0, min(1.0, 1.0 - h_u_given_v))
            except Exception as e:
                logger.warning(f"Numerical differentiation failed: {e}, falling back to Monte Carlo")
                return estimate_conditional_mc(copula, u_value, v_level, conditional_var)
                
        else:
            raise ValueError(f"Unknown conditional: {conditional_var}. Use 'v|u' or 'u|v'.")
            
    except Exception as e:
        logger.error(f"Error computing conditional exceedance: {str(e)}")
        return np.nan
    
def compute_conditional_exceeding_given_exceeding(
    copula: Any,
    u_level: float,
    v_level: float,
    conditional_var: str = 'v|u'
) -> float:
    """
    Compute conditional exceedance probability given that the conditioning
    variable also exceeds its threshold.
    
    Parameters
    ----------
    copula : Any
        Fitted copula object
    u_level, v_level : float
        Exceedance levels
    conditional_var : str, optional
        Which conditional to compute: 'v|u' for P(V>v|U>u) or 'u|v' for P(U>u|V>v)
        
    Returns
    -------
    float
        Conditional probability
    """
    if u_level < 0 or u_level > 1 or v_level < 0 or v_level > 1:
        logger.warning(f"Invalid probability levels: u_level={u_level}, v_level={v_level}")
        return np.nan
    
    # Define a Monte Carlo fallback method
    def estimate_conditional_exceeding_mc(copula, u_level, v_level, conditional_var, n_samples=10000):
        try:
            # Generate samples from the copula
            samples = copula.rvs(n_samples)
            u_samples, v_samples = samples[:, 0], samples[:, 1]
            
            if conditional_var == 'v|u':
                # Find samples where U > u_level
                u_exceed = u_samples > u_level
                if np.sum(u_exceed) == 0:
                    return np.nan
                
                # Compute how many of those also have V > v_level
                prob = np.mean(v_samples[u_exceed] > v_level)
                return prob
                
            elif conditional_var == 'u|v':
                # Find samples where V > v_level
                v_exceed = v_samples > v_level
                if np.sum(v_exceed) == 0:
                    return np.nan
                
                # Compute how many of those also have U > u_level
                prob = np.mean(u_samples[v_exceed] > u_level)
                return prob
            
            else:
                raise ValueError(f"Unknown conditional: {conditional_var}")
        except Exception as e:
            logger.error(f"Monte Carlo conditional probability estimation failed: {e}")
            return np.nan
    
    try:
        # First try to compute using the joint exceedance formula
        # P(V>v|U>u) = P(V>v, U>u) / P(U>u)
        try:
            # Compute joint exceedance P(U>u, V>v)
            joint_prob = compute_joint_exceedance(copula, u_level, v_level)
            
            if conditional_var == 'v|u':
                # The probability that U exceeds u_level
                p_u_exceed = 1.0 - u_level
                
                if p_u_exceed <= 0:
                    return np.nan  # Cannot condition on zero probability event
                
                return max(0.0, min(1.0, joint_prob / p_u_exceed))
                
            elif conditional_var == 'u|v':
                # The probability that V exceeds v_level
                p_v_exceed = 1.0 - v_level
                
                if p_v_exceed <= 0:
                    return np.nan  # Cannot condition on zero probability event
                
                return max(0.0, min(1.0, joint_prob / p_v_exceed))
                
            else:
                raise ValueError(f"Unknown conditional: {conditional_var}. Use 'v|u' or 'u|v'.")
        except Exception as e:
            logger.warning(f"Formula-based conditional probability failed: {e}, using Monte Carlo")
            return estimate_conditional_exceeding_mc(copula, u_level, v_level, conditional_var)
            
    except Exception as e:
        logger.error(f"Error computing conditional exceeding given exceeding: {str(e)}")
        return np.nan

def compute_conditional_exceeding_given_exceeding(
    copula: Any,
    u_level: float,
    v_level: float,
    conditional_var: str = 'v|u'
) -> float:
    """
    Compute conditional exceedance probability given that the conditioning
    variable also exceeds its threshold.
    
    Parameters
    ----------
    copula : Any
        Fitted copula object
    u_level, v_level : float
        Exceedance levels
    conditional_var : str, optional
        Which conditional to compute: 'v|u' for P(V>v|U>u) or 'u|v' for P(U>u|V>v)
        
    Returns
    -------
    float
        Conditional probability
    """
    if u_level < 0 or u_level > 1 or v_level < 0 or v_level > 1:
        raise ValueError(f"Levels must be in [0, 1], got u_level={u_level}, v_level={v_level}")
    
    # P(V>v|U>u) = P(V>v, U>u) / P(U>u)
    # P(U>u|V>v) = P(U>u, V>v) / P(V>v)
    
    try:
        # Compute joint exceedance P(U>u, V>v)
        joint_prob = compute_joint_exceedance(copula, u_level, v_level)
        
        if conditional_var == 'v|u':
            # We want P(V>v|U>u)
            # The probability that U exceeds u
            p_u_exceed = 1.0 - u_level
            
            if p_u_exceed <= 0:
                return np.nan  # Cannot condition on zero probability event
            
            return joint_prob / p_u_exceed
            
        elif conditional_var == 'u|v':
            # We want P(U>u|V>v)
            # The probability that V exceeds v
            p_v_exceed = 1.0 - v_level
            
            if p_v_exceed <= 0:
                return np.nan  # Cannot condition on zero probability event
            
            return joint_prob / p_v_exceed
            
        else:
            raise ValueError(f"Unknown conditional: {conditional_var}. Use 'v|u' or 'u|v'.")
            
    except Exception as e:
        logger.error(f"Error computing conditional exceeding given exceeding: {str(e)}")
        return np.nan

def compute_cpr(
    u: np.ndarray, 
    v: np.ndarray, 
    u_level: float, 
    v_level: float, 
    copula: Optional[Any] = None
) -> float:
    """
    Compute the Conditional Probability Ratio (CPR) for bivariate data.
    
    CPR = P(U>u, V>v) / [P(U>u) * P(V>v)]
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    u_level, v_level : float
        Exceedance thresholds
    copula : Optional[Any], optional
        Fitted copula object. If None, computes from empirical data.
        
    Returns
    -------
    float
        CPR value (>1 indicates positive dependence)
    """
    if u_level < 0 or u_level > 1 or v_level < 0 or v_level > 1:
        raise ValueError("Levels must be in [0, 1]")
    
    # Calculate empirical or model-based CPR
    if copula is None:
        # Empirical calculation from data
        exceed_u = (u > u_level)
        exceed_v = (v > v_level)
        
        # Joint probability P(U>u, V>v)
        p_joint = np.mean(exceed_u & exceed_v)
        
        # Marginal probabilities
        p_u = np.mean(exceed_u)
        p_v = np.mean(exceed_v)
        
        # Independence assumption probability
        p_ind = p_u * p_v
    else:
        # Calculate from fitted copula
        # Joint probability P(U>u, V>v)
        p_joint = compute_joint_exceedance(copula, u_level, v_level)
        
        # Marginal probabilities
        p_u = 1.0 - u_level
        p_v = 1.0 - v_level
        
        # Independence assumption probability
        p_ind = p_u * p_v
    
    # Calculate CPR
    if p_ind <= 0:
        return np.nan  # Cannot calculate CPR for zero probability
        
    cpr = p_joint / p_ind
    
    return cpr


def fit_copula_with_metrics(
    u: np.ndarray,
    v: np.ndarray,
    method: str = 'auto',
    candidates: Optional[Dict[str, Callable]] = None,
    compute_diagnostics: bool = True
) -> Dict[str, Any]:
    """
    Fit a copula to bivariate data and compute additional metrics.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations
    method : str, optional
        Copula type or 'auto' for automatic selection
    candidates : Optional[Dict[str, Callable]], optional
        Dictionary of copula fitting functions if method='auto'
    compute_diagnostics : bool, optional
        Whether to compute additional diagnostics
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with fitted copula and metrics
    """
    from src.compound_flooding.copula_selection import select_best_copula
    
    # Validate inputs
    try:
        u, v = _validate_pobs(u, v)
    except Exception as e:
        logger.error(f"Validation error in fit_copula_with_metrics: {str(e)}")
        return {
            'error': str(e),
            'copula': None,
            'method': 'independence',
            'success': False
        }
    
    # Dictionary of fitting functions
    default_candidates = {
        'Gumbel': fit_gumbel,
        'Frank': fit_frank,
        'StudentT': fit_student,
        'Gaussian': fit_gaussian,
        'Clayton': fit_clayton
    }
    
    if candidates is None:
        candidates = default_candidates
    
    # Fit the copula
    if method == 'auto':
        try:
            # Select best copula
            copula_info = select_best_copula(u, v, candidates)
            cop = copula_info['copula']
            method = copula_info['name']
        except Exception as e:
            logger.error(f"Error in automatic copula selection: {str(e)}")
            cop = GaussianCopula(corr=np.eye(2))
            method = 'independence'
    else:
        # Use specified copula type
        if method == 'gumbel':
            cop = fit_gumbel(u, v)
        elif method == 'frank':
            cop = fit_frank(u, v)
        elif method == 'student':
            cop = fit_student(u, v)
        elif method == 'gaussian':
            cop = fit_gaussian(u, v)
        elif method == 'clayton':
            cop = fit_clayton(u, v)
        elif method == 'independence':
            cop = fit_independence(u, v)
        else:
            logger.error(f"Unknown copula type: {method}")
            cop = fit_independence(u, v)
            method = 'independence'
    
    # Compute additional metrics if requested
    results = {
        'copula': cop,
        'method': method,
        'success': True
    }
    
    if compute_diagnostics:
        # Compute log-likelihood
        uv = np.column_stack((u, v))
        try:
            ll = np.sum(cop.logpdf(uv))
            
            # Get parameter count for AIC/BIC
            k = 2 if method == 'StudentT' else 1
            if method == 'independence':
                k = 0
            
            aic = 2 * k - 2 * ll
            bic = k * np.log(len(u)) - 2 * ll
            
            results['log_likelihood'] = float(ll)
            results['aic'] = float(aic)
            results['bic'] = float(bic)
            
            # Compute tail dependence
            lambda_l, lambda_u = compute_tail_dependence(cop)
            results['tail_dependence'] = {
                'lower': float(lambda_l),
                'upper': float(lambda_u)
            }
            
            # Compute key parameter(s)
            if hasattr(cop, 'theta'):
                results['params'] = {'theta': float(cop.theta)}
            elif hasattr(cop, 'corr'):
                results['params'] = {'rho': float(cop.corr[0, 1])}
                if hasattr(cop, 'df'):
                    results['params']['df'] = float(cop.df)
            else:
                results['params'] = {}
            
        except Exception as e:
            logger.error(f"Error computing diagnostics: {str(e)}")
            results['diagnostics_error'] = str(e)
    
    return results


if __name__ == "__main__":
    # Simple test/demonstration
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau
    
    print("Testing copula fitting functions")
    
    # Generate some bivariate data with known dependence
    n = 1000
    rho = 0.7
    
    # Generate correlated normals
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    x, y = np.random.multivariate_normal(mean, cov, n).T
    
    # Convert to uniform
    u, v = create_pseudo_observations(x, y)
    
    # Compute Kendall's tau
    tau, _ = kendalltau(u, v)
    print(f"Kendall's tau: {tau:.4f}")
    
    # Fit various copulas
    fit_results = fit_copula_with_metrics(u, v, method='auto')
    best_copula = fit_results['copula']
    best_method = fit_results['method']
    
    print(f"Best copula: {best_method}")
    print(f"Log-likelihood: {fit_results.get('log_likelihood', 'N/A')}")
    print(f"AIC: {fit_results.get('aic', 'N/A')}")
    print(f"Parameters: {fit_results.get('params', {})}")
    print(f"Tail dependence: {fit_results.get('tail_dependence', {})}")
    
    # Create a grid for visualizing the copula density
    grid_size = 50
    grid_u = np.linspace(0.01, 0.99, grid_size)
    grid_v = np.linspace(0.01, 0.99, grid_size)
    uu, vv = np.meshgrid(grid_u, grid_v)
    
    # Compute the copula density
    uv_grid = np.column_stack((uu.flatten(), vv.flatten()))
    try:
        density = best_copula.pdf(uv_grid).reshape(grid_size, grid_size)
    except:
        density = np.ones((grid_size, grid_size))  # Fallback if density computation fails
    
    # Plot data and fitted copula
    plt.figure(figsize=(12, 10))
    
    # Plot the scatter of pseudo-observations
    plt.subplot(2, 2, 1)
    plt.scatter(u, v, alpha=0.5, s=5)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title(f'Pseudo-observations (tau={tau:.4f})')
    plt.grid(True)
    
    # Plot the copula density contours
    plt.subplot(2, 2, 2)
    plt.contourf(uu, vv, density, cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title(f'Fitted {best_method} Copula Density')
    plt.grid(True)
    
    # Plot joint exceedance probabilities
    plt.subplot(2, 2, 3)
    exceedance_levels = np.linspace(0.5, 0.99, grid_size)
    joint_exc = np.zeros((grid_size, grid_size))
    
    for i, u_level in enumerate(exceedance_levels):
        for j, v_level in enumerate(exceedance_levels):
            joint_exc[i, j] = compute_joint_exceedance(best_copula, u_level, v_level)
    
    plt.contourf(exceedance_levels, exceedance_levels, joint_exc, cmap='viridis')
    plt.colorbar(label='P(U>u, V>v)')
    plt.xlabel('u threshold')
    plt.ylabel('v threshold')
    plt.title('Joint Exceedance Probabilities')
    plt.grid(True)
    
    # Plot CPR
    plt.subplot(2, 2, 4)
    cpr_values = np.zeros((grid_size, grid_size))
    
    for i, u_level in enumerate(exceedance_levels):
        for j, v_level in enumerate(exceedance_levels):
            cpr_values[i, j] = compute_cpr(u, v, u_level, v_level, best_copula)
    
    plt.contourf(exceedance_levels, exceedance_levels, cpr_values, 
                levels=[0, 0.5, 1, 1.5, 2, 3, 4, 5], cmap='viridis')
    plt.colorbar(label='CPR')
    plt.xlabel('u threshold')
    plt.ylabel('v threshold')
    plt.title('Conditional Probability Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()