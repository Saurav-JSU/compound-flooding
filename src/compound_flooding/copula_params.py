# src/compound_flooding/copula_params.py

"""
Module: copula_params.py
Responsibilities:
- Robust parameter estimation for various copulas
- Conversion between dependence measures (Kendall's tau, Spearman's rho, etc.)
- Numerical stability considerations for edge cases
- Fallback mechanisms for estimation failures
"""

import numpy as np
import logging
from typing import Tuple, Union, Optional, Callable
from scipy.optimize import root_scalar, minimize
from scipy.stats import kendalltau, spearmanr

try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_valid_correlation_matrix(corr: np.ndarray) -> bool:
    """
    Check if a matrix is a valid correlation matrix:
    - Symmetric
    - All diagonal elements are 1
    - All off-diagonal elements are in [-1, 1]
    - Positive semi-definite

    Parameters
    ----------
    corr : np.ndarray
        Correlation matrix to check

    Returns
    -------
    bool
        True if valid correlation matrix, False otherwise
    """
    if not isinstance(corr, np.ndarray):
        return False
    
    # Check shape (must be square)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        return False
    
    # Check symmetry
    if not np.allclose(corr, corr.T):
        return False
    
    # Check diagonal elements
    if not np.allclose(np.diag(corr), 1.0):
        return False
    
    # Check range of off-diagonal elements
    mask = ~np.eye(corr.shape[0], dtype=bool)
    if np.any(corr[mask] < -1.0) or np.any(corr[mask] > 1.0):
        return False
    
    # Check positive semi-definiteness (all eigenvalues ≥ 0)
    try:
        eigvals = np.linalg.eigvalsh(corr)
        return np.all(eigvals >= -1e-10)  # Allow for small numerical errors
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(corr: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """
    Find the nearest positive definite correlation matrix to the given matrix.
    Implemented using the algorithm from Higham (2002).

    Parameters
    ----------
    corr : np.ndarray
        Input correlation matrix
    epsilon : float, optional
        Small value to ensure positive definiteness

    Returns
    -------
    np.ndarray
        Nearest positive definite correlation matrix
    """
    if is_valid_correlation_matrix(corr):
        return corr
        
    n = corr.shape[0]
    
    # Ensure symmetry
    corr = (corr + corr.T) / 2
    
    # Get eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(corr)
    
    # Clip eigenvalues to be positive
    eigval = np.maximum(eigval, epsilon)
    
    # Reconstruct the matrix
    corr_psd = eigvec @ np.diag(eigval) @ eigvec.T
    
    # Ensure unit diagonal
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)
    
    # Final symmetrization due to numerical issues
    corr_psd = (corr_psd + corr_psd.T) / 2
    
    return corr_psd


def estimate_theta_gumbel(tau: float, min_theta: float = 1.01) -> float:
    """
    Estimate Gumbel copula parameter theta from Kendall's tau.
    For tau <= 0, returns the independence case (min_theta).
    For tau > 0, uses formula theta = 1/(1-tau) with numerical stabilization.

    Parameters
    ----------
    tau : float
        Kendall's tau correlation (-1 <= tau <= 1)
    min_theta : float, optional
        Minimum theta value (must be > 1 for Gumbel)
        
    Returns
    -------
    float
        Estimated theta parameter
        
    Notes
    -----
    The Gumbel copula is only defined for positive dependence (tau > 0).
    For negative dependence, this function returns the independence case.
    """
    # Validate input
    if not np.isfinite(tau):
        logger.warning("Non-finite tau provided to estimate_theta_gumbel, using independence")
        return min_theta
    
    # For negative or zero dependence, return independence
    if tau <= 0.0:
        return min_theta
    
    # Apply stabilization for tau near 1 (avoiding division by zero)
    if tau > 0.99:
        tau_reg = 0.99 + 0.01 * np.tanh(100 * (tau - 0.99))
        logger.info(f"Regularizing tau={tau:.6f} to {tau_reg:.6f} for numerical stability")
        tau = tau_reg
        
    # Compute theta using the standard formula
    theta = 1.0 / (1.0 - tau)
    
    # Ensure theta > 1 (Gumbel copula constraint)
    theta = max(theta, min_theta)
    
    return theta


def frank_tau(theta: float) -> float:
    """
    Compute Kendall's tau for a Frank copula with parameter theta.
    
    Parameters
    ----------
    theta : float
        Frank copula parameter
        
    Returns
    -------
    float
        Kendall's tau corresponding to theta
        
    Notes
    -----
    For theta = 0, returns tau = 0 (independence).
    For non-zero theta, computes using Debye function D₁.
    """
    if abs(theta) < 1e-10:
        return 0.0
    
    # Use Debye function D₁ formula
    # τ = 1 + 4/θ * [D₁(θ) - 1]
    # where D₁(θ) = (1/θ) * ∫₀ᶿ [t/(e^t - 1)] dt
    
    # Approximate Debye function using series expansion for small theta
    if abs(theta) < 0.1:
        # D₁(θ) ≈ 1 - θ/4 + θ²/72 - ...
        debye = 1 - theta/4 + (theta**2)/72 - (theta**4)/43200
    else:
        # For larger theta, use direct integration
        t = np.linspace(0, abs(theta), 1000)[1:]  # Avoid division by zero at t=0
        integrand = t / (np.exp(t) - 1)
        debye = np.trapz(integrand, t) / abs(theta)
    
    tau = 1.0 + 4.0 * (debye - 1.0) / theta
    
    # Ensure tau is in [-1, 1]
    return np.clip(tau, -1.0, 1.0)


def estimate_theta_frank(
    tau: float, 
    bracket: Tuple[float, float] = (-50, 50), 
    tol: float = 1e-6
) -> float:
    """
    Estimate Frank copula parameter theta from Kendall's tau using numerical root-finding.
    For small |tau| < 0.01, returns theta = 0 (independence).
    
    Parameters
    ----------
    tau : float
        Kendall's tau correlation (-1 <= tau <= 1)
    bracket : Tuple[float, float], optional
        Bracket for root-finding algorithm
    tol : float, optional
        Tolerance for convergence
        
    Returns
    -------
    float
        Estimated theta parameter
        
    Notes
    -----
    For very small tau values, returns independence (theta = 0).
    If root-finding fails, falls back to approximate methods.
    """
    # Validate input
    if not np.isfinite(tau) or tau < -1 or tau > 1:
        logger.warning(f"Invalid tau={tau} provided to estimate_theta_frank, using independence")
        return 0.0
    
    # For very small dependence, return independence
    if abs(tau) < 0.01:
        return 0.0
    
    # Define the objective function: frank_tau(theta) - tau = 0
    def objective(theta: float) -> float:
        return frank_tau(theta) - tau
    
    # Try root-finding first
    try:
        sol = root_scalar(objective, bracket=bracket, method='brentq', rtol=tol)
        if sol.converged:
            return sol.root
    except Exception as e:
        logger.warning(f"Root-finding failed in estimate_theta_frank: {str(e)}")
    
    # If root-finding fails, use approximate method based on sign of tau
    if tau > 0:
        # Positive dependence: try a few fixed values and pick closest
        candidates = [1.0, 5.0, 10.0, 20.0]
    else:
        # Negative dependence: try a few fixed values and pick closest
        candidates = [-1.0, -5.0, -10.0, -20.0]
    
    # Find candidate with closest tau
    tau_diffs = [abs(frank_tau(theta) - tau) for theta in candidates]
    best_idx = np.argmin(tau_diffs)
    
    logger.info(f"Using approximate theta={candidates[best_idx]} for tau={tau}")
    return candidates[best_idx]


def estimate_corr_from_kendall(tau: float) -> float:
    """
    Estimate linear correlation from Kendall's tau using the sin formula.
    
    Parameters
    ----------
    tau : float
        Kendall's tau correlation (-1 <= tau <= 1)
        
    Returns
    -------
    float
        Estimated Pearson correlation coefficient
        
    Notes
    -----
    Uses the formula ρ = sin(π*τ/2) which is exact for elliptical distributions
    and approximate for others.
    """
    # Validate input
    if not np.isfinite(tau) or tau < -1 or tau > 1:
        logger.warning(f"Invalid tau={tau} provided to estimate_corr_from_kendall, using 0")
        return 0.0
    
    # Apply sin formula: ρ = sin(π*τ/2)
    rho = np.sin(0.5 * np.pi * tau)
    
    return rho


def estimate_corr_student_t(
    u: np.ndarray, 
    v: np.ndarray,
    df_init: float = 5.0,
    method: str = 'kendall'
) -> Tuple[np.ndarray, float]:
    """
    Estimate correlation matrix and degrees of freedom for Student-T copula.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations (uniform data)
    df_init : float, optional
        Initial guess for degrees of freedom
    method : str, optional
        Method for correlation estimation ('kendall', 'spearman', 'pearson')
        
    Returns
    -------
    Tuple[np.ndarray, float]
        (correlation_matrix, degrees_of_freedom)
        
    Notes
    -----
    Currently only estimates the correlation matrix using the specified method.
    The df parameter is not estimated but returned as provided.
    Future enhancement: estimate df via MLE.
    """
    # Validate inputs
    if u.shape != v.shape:
        raise ValueError(f"u and v must have same shape, got {u.shape} vs {v.shape}")
        
    if len(u) < 3:
        logger.warning("Too few observations for reliable correlation estimation")
        return np.array([[1.0, 0.0], [0.0, 1.0]]), df_init
    
    # Estimate correlation using specified method
    if method == 'kendall':
        tau, _ = kendalltau(u, v, nan_policy='omit')
        rho = estimate_corr_from_kendall(tau)
    elif method == 'spearman':
        rho, _ = spearmanr(u, v, nan_policy='omit')
        # Convert Spearman's rho to Pearson's r using sin formula (a reasonable approximation)
        rho = 2 * np.sin(np.pi * rho / 6)
    elif method == 'pearson':
        # For Pearson, transform u,v to normal first
        from scipy.stats import norm
        x = norm.ppf(u)
        y = norm.ppf(v)
        rho = np.corrcoef(x, y)[0, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure rho is in valid range
    rho = np.clip(rho, -0.99, 0.99)
    
    # Create 2x2 correlation matrix
    corr = np.array([[1.0, rho], [rho, 1.0]])
    
    # Ensure the correlation matrix is valid
    if not is_valid_correlation_matrix(corr):
        logger.warning("Invalid correlation matrix detected, using nearest positive definite")
        corr = nearest_positive_definite(corr, epsilon=1e-6)
    
    return corr, df_init


def estimate_rho_gaussian(
    u: np.ndarray, 
    v: np.ndarray,
    method: str = 'kendall'
) -> float:
    """
    Estimate Gaussian copula correlation parameter.
    
    Parameters
    ----------
    u, v : np.ndarray
        Pseudo-observations (uniform data)
    method : str, optional
        Method for correlation estimation ('kendall', 'spearman', 'pearson')
        
    Returns
    -------
    float
        Estimated correlation coefficient
        
    Notes
    -----
    When using 'pearson', the data is first transformed to normal scores.
    """
    # Validate inputs
    if u.shape != v.shape:
        raise ValueError(f"u and v must have same shape, got {u.shape} vs {v.shape}")
        
    if len(u) < 3:
        logger.warning("Too few observations for reliable correlation estimation")
        return 0.0
    
    # Estimate correlation using specified method
    if method == 'kendall':
        tau, _ = kendalltau(u, v, nan_policy='omit')
        rho = estimate_corr_from_kendall(tau)
    elif method == 'spearman':
        rho, _ = spearmanr(u, v, nan_policy='omit')
        # Convert Spearman's rho to Pearson's r
        rho = 2 * np.sin(np.pi * rho / 6)
    elif method == 'pearson':
        # Transform to normal scores
        from scipy.stats import norm
        x = norm.ppf(np.clip(u, 0.001, 0.999))
        y = norm.ppf(np.clip(v, 0.001, 0.999))
        rho = np.corrcoef(x, y)[0, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Ensure rho is in valid range
    rho = np.clip(rho, -0.99, 0.99)
    
    return rho


def estimate_clayton_theta(tau: float) -> float:
    """
    Estimate Clayton copula parameter theta from Kendall's tau.
    
    Parameters
    ----------
    tau : float
        Kendall's tau correlation (-1 <= tau <= 1)
        
    Returns
    -------
    float
        Estimated theta parameter
        
    Notes
    -----
    Clayton copula only supports positive dependence except for theta=-1,
    which corresponds to the countermonotonicity case.
    """
    # Validate input
    if not np.isfinite(tau) or tau < -1 or tau > 1:
        logger.warning(f"Invalid tau={tau} provided to estimate_clayton_theta, using independence")
        return 0.0
    
    # Clayton only allows tau >= 0, except for countermonotonicity
    if tau <= -1:
        return -1.0  # Countermonotonicity
    
    if tau < 0:
        logger.warning(f"Clayton copula is only defined for tau >= 0, got {tau}. Using independence.")
        return 0.0
    
    # For tau ≈ 0, return independence
    if abs(tau) < 0.01:
        return 0.0
    
    # Apply formula: θ = 2τ/(1-τ)
    # Apply stabilization for tau near 1
    if tau > 0.99:
        tau_reg = 0.99 + 0.01 * np.tanh(100 * (tau - 0.99))
        logger.info(f"Regularizing tau={tau:.6f} to {tau_reg:.6f} for numerical stability")
        tau = tau_reg
    
    theta = 2 * tau / (1 - tau)
    
    return theta


def compute_tail_dependence(
    copula: object, 
    p: float = 0.99,
    samples: int = 10000
) -> Tuple[float, float]:
    """
    Compute upper and lower tail dependence coefficients.
    
    Parameters
    ----------
    copula : object
        Fitted copula object
    p : float, optional
        Probability level for empirical estimation (0 < p < 1)
    samples : int, optional
        Number of samples for empirical estimation
        
    Returns
    -------
    Tuple[float, float]
        (lower_tail_dependence, upper_tail_dependence)
        
    Notes
    -----
    For many copulas, the exact tail dependence is available analytically:
    - Gaussian: λᵤ = λₗ = 0 (if ρ < 1)
    - t: λᵤ = λₗ = 2*t_{v+1}(-sqrt((v+1)(1-ρ)/(1+ρ)))
    - Gumbel: λᵤ = 2 - 2^(1/θ), λₗ = 0
    - Clayton: λᵤ = 0, λₗ = 2^(-1/θ) (if θ > 0)
    
    This function uses a combination of analytical formulas and empirical
    estimation when analytical forms are not available.
    """
    # Extract copula type and parameters
    copula_type = copula.__class__.__name__
    
    # Compute tail dependence analytically where possible
    if copula_type == 'GaussianCopula':
        # Gaussian copula has no tail dependence except at ρ=1
        rho = copula.corr[0, 1]
        if abs(rho) >= 0.9999:
            return 1.0, 1.0
        return 0.0, 0.0
        
    elif copula_type == 'GumbelCopula':
        # Gumbel has upper tail dependence only
        theta = copula.theta
        lambda_u = 2 - 2**(1/theta)
        lambda_l = 0.0
        return lambda_l, lambda_u
        
    elif copula_type == 'FrankCopula':
        # Frank has no tail dependence
        return 0.0, 0.0
        
    elif copula_type == 'ClaytonCopula':
        # Clayton has lower tail dependence only (for θ > 0)
        theta = copula.theta
        if theta <= 0:
            return 0.0, 0.0
        lambda_l = 2**(-1/theta)
        lambda_u = 0.0
        return lambda_l, lambda_u
        
    elif copula_type == 'StudentTCopula':
        # Student's t has symmetric tail dependence
        rho = copula.corr[0, 1]
        df = copula.df
        
        from scipy.stats import t
        lambda_t = 2 * t.cdf(-np.sqrt((df+1)*(1-rho)/(1+rho)), df+1)
        return lambda_t, lambda_t
    
    # For other copulas, use empirical estimation
    # Generate samples from the copula
    u, v = copula.rvs(samples)
    
    # Compute empirical lower tail dependence: lim_{q→0} P(V<q | U<q)
    q_l = 1 - p  # Small quantile
    lower_u = u <= q_l
    lower_v = v <= q_l
    lambda_l = np.mean(lower_v[lower_u]) if np.sum(lower_u) > 0 else 0.0
    
    # Compute empirical upper tail dependence: lim_{q→1} P(V>q | U>q)
    q_u = p  # High quantile
    upper_u = u >= q_u
    upper_v = v >= q_u
    lambda_u = np.mean(upper_v[upper_u]) if np.sum(upper_u) > 0 else 0.0
    
    return lambda_l, lambda_u


# If Numba is available, create accelerated versions of key functions
if HAVE_NUMBA:
    @numba.njit
    def _compute_gumbel_theta_batch(tau_values):
        """Numba-accelerated batch computation of Gumbel theta from tau values."""
        result = np.empty_like(tau_values)
        for i, tau in enumerate(tau_values):
            if not np.isfinite(tau) or tau <= 0:
                result[i] = 1.01  # Independence
            else:
                # Apply regularization for numerical stability
                if tau > 0.99:
                    tau = 0.99 + 0.01 * np.tanh(100 * (tau - 0.99))
                result[i] = max(1.0 / (1.0 - tau), 1.01)
        return result
        
    def estimate_theta_gumbel_batch(tau_values: np.ndarray) -> np.ndarray:
        """
        Vectorized version of estimate_theta_gumbel for multiple tau values.
        
        Parameters
        ----------
        tau_values : np.ndarray
            Array of Kendall's tau values
            
        Returns
        -------
        np.ndarray
            Array of estimated theta values
        """
        return _compute_gumbel_theta_batch(tau_values)

if __name__ == "__main__":
    # Simple test/demonstration
    import matplotlib.pyplot as plt
    from scipy.stats import kendalltau
    
    print("Testing copula parameter estimation functions")
    
    # Test Gumbel parameter estimation
    taus = np.linspace(-0.5, 0.99, 100)
    thetas = [estimate_theta_gumbel(tau) for tau in taus]
    
    plt.figure(figsize=(10, 6))
    plt.plot(taus, thetas, label='Gumbel theta')
    plt.axhline(1.0, linestyle='--', color='red', label='Independence (theta=1)')
    plt.xlabel("Kendall's tau")
    plt.ylabel('Theta')
    plt.title('Gumbel Parameter Estimation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Test Frank parameter estimation
    thetas_frank = [estimate_theta_frank(tau) for tau in taus]
    
    plt.figure(figsize=(10, 6))
    plt.plot(taus, thetas_frank, label='Frank theta')
    plt.axhline(0.0, linestyle='--', color='red', label='Independence (theta=0)')
    plt.xlabel("Kendall's tau")
    plt.ylabel('Theta')
    plt.title('Frank Parameter Estimation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Generate correlated data and test correlation estimation
    from scipy.stats import norm
    n = 1000
    rho_true = 0.7
    
    # Generate correlated normals
    mean = [0, 0]
    cov = [[1, rho_true], [rho_true, 1]]
    x, y = np.random.multivariate_normal(mean, cov, n).T
    
    # Convert to uniform
    u = norm.cdf(x)
    v = norm.cdf(y)
    
    # Estimate using different methods
    tau, _ = kendalltau(u, v)
    corr_est, _ = estimate_corr_student_t(u, v)
    rho_est = estimate_rho_gaussian(u, v)
    
    print(f"True correlation: {rho_true}")
    print(f"Kendall's tau: {tau}")
    print(f"Estimated correlation (Student-t): {corr_est[0, 1]}")
    print(f"Estimated correlation (Gaussian): {rho_est}")
    
    # Test positive definite correction
    bad_corr = np.array([[1.0, 1.2], [1.2, 1.0]])
    print(f"Invalid correlation matrix: {bad_corr}")
    fixed_corr = nearest_positive_definite(bad_corr)
    print(f"Fixed correlation matrix: {fixed_corr}")
    print(f"Is valid correlation matrix: {is_valid_correlation_matrix(fixed_corr)}")