# src/compound_flooding/copula_utils.py (Professional Enhancement)

import numpy as np
from scipy.stats import kendalltau, norm
from scipy.optimize import root_scalar
from statsmodels.distributions.copula.api import (
    GumbelCopula,
    FrankCopula,
    StudentTCopula,
    GaussianCopula
)

# =============================
# Input Validation
# =============================

def _validate_pobs(u, v):
    """
    Validate pseudo-observations u and v.
    Raises ValueError if lengths mismatch or values not in (0,1).
    """
    u = np.asarray(u)
    v = np.asarray(v)
    if u.shape != v.shape:
        raise ValueError(f"u and v must have same shape, got {u.shape} vs {v.shape}")
    if np.any(u <= 0) or np.any(u >= 1) or np.any(v <= 0) or np.any(v >= 1):
        raise ValueError("Pseudo-observations u and v must lie in the open interval (0,1)")
    return u, v

# =============================
# Helper Functions (Parameter Estimation)
# =============================

def estimate_theta_gumbel(tau: float) -> float:
    """
    Estimate Gumbel copula parameter theta from Kendall's tau.
    Ensures theta > 1.01 to satisfy Gumbel constraints.
    """
    tau = max(tau, 0.01)
    theta = 1.0 / (1.0 - tau)
    return max(theta, 1.01)


def estimate_theta_frank(tau: float) -> float:
    """
    Numerically invert Kendall's tau for Frank copula parameter theta.
    If |tau| < 0.05, returns 0 for independence.
    Falls back to theta=5.0 if root finding fails.
    """
    if abs(tau) < 0.05:
        return 0.0

    def frank_tau(theta: float) -> float:
        if theta == 0:
            return 0.0
        num = 1.0 - (1.0 / theta) * (1.0 - np.exp(-theta))
        den = np.exp(-theta) - 1.0
        return 1.0 + den / num

    def objective(theta: float) -> float:
        return frank_tau(theta) - tau

    try:
        sol = root_scalar(objective, bracket=[-50, 50], method='bisect')
        if sol.converged:
            return sol.root
    except Exception:
        pass
    return 5.0


def estimate_corr_student_t(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Estimate correlation matrix for Student-T copula using Kendall's tau.
    Converts tau to linear correlation via sin(pi*tau/2).
    """
    tau, _ = kendalltau(u, v)
    rho = np.sin(0.5 * np.pi * tau)
    return np.array([[1.0, rho], [rho, 1.0]])


def estimate_rho_gaussian(u: np.ndarray, v: np.ndarray) -> float:
    """
    Estimate Gaussian copula correlation by computing Pearson correlation
    of the normal scores (inverse CDF of uniform data).
    """
    x = norm.ppf(u)
    y = norm.ppf(v)
    return np.corrcoef(x, y)[0, 1]

# =============================
# Copula Fitting Functions
# =============================

def fit_gumbel(u, v, validate: bool = True) -> GumbelCopula:
    """
    Fit a Gumbel copula to (u, v) using Kendall's tau.
    Returns an instantiated GumbelCopula.
    """
    if validate:
        u, v = _validate_pobs(u, v)
    tau, _ = kendalltau(u, v)
    theta = estimate_theta_gumbel(tau)
    return GumbelCopula(theta=theta)


def fit_frank(u, v, validate: bool = True):
    """
    Fit a Frank copula to (u, v) by numerical inversion of Kendall's tau.
    Falls back to independence Gaussian copula if theta==0.
    """
    if validate:
        u, v = _validate_pobs(u, v)
    tau, _ = kendalltau(u, v)
    theta = estimate_theta_frank(tau)
    if np.isclose(theta, 0.0):
        return GaussianCopula(corr=np.eye(2))
    return FrankCopula(theta=theta)


def fit_student(
    u: np.ndarray,
    v: np.ndarray,
    validate: bool = True,
    df: float = 5.0
) -> StudentTCopula:
    """
    Fit a Student-T copula to (u, v).
    Correlation estimated via Kendall's tau.
    df currently fixed; consider MLE estimation for df in future.
    If the estimated correlation matrix is invalid or the data have no
    variability, falls back to an independence Gaussian copula.
    """
    if validate:
        u, v = _validate_pobs(u, v)

    corr = estimate_corr_student_t(u, v)

    # Guard against invalid corr (NaN/Inf) or degenerate data
    if (
        not np.all(np.isfinite(corr)) or
        len(np.unique(u)) < 2 or
        len(np.unique(v)) < 2
    ):
        return GaussianCopula(corr=np.eye(2))

    # Attempt to construct Student-T; if it errors, fallback to Gaussian
    try:
        return StudentTCopula(corr=corr, df=df)
    except ValueError:
        return GaussianCopula(corr=corr)

def fit_gaussian(
    u: np.ndarray,
    v: np.ndarray,
    validate: bool = True
) -> GaussianCopula:
    """
    Fit a Gaussian copula to (u, v).
    Correlation estimated via Kendall's tau (converted to Pearson ρ).
    If the estimated correlation matrix is invalid or the data have no
    variability, falls back to the independence Gaussian copula.
    """
    if validate:
        u, v = _validate_pobs(u, v)

    # Estimate Kendall’s tau → Pearson’s rho
    tau, _ = kendalltau(u, v)
    rho = np.sin(0.5 * np.pi * tau)
    corr = np.array([[1.0, rho], [rho, 1.0]])

    # Guard against NaN/Inf or degenerate data
    if (
        not np.all(np.isfinite(corr)) or
        len(np.unique(u)) < 2 or
        len(np.unique(v)) < 2
    ):
        # Pure independence
        return GaussianCopula(corr=np.eye(2))

    # Try to construct the copula; fallback on numerical issues
    try:
        return GaussianCopula(corr=corr)
    except ValueError:
        return GaussianCopula(corr=np.eye(2))


# =============================
# Model Selection via AIC
# =============================

def select_best_copula(u, v, candidates=None) -> object:
    """
    Select the best copula by AIC among candidates.
    Candidates is a dict of name->fit_function. If None, uses all four.
    """
    if candidates is None:
        candidates = {
            'Gumbel': fit_gumbel,
            'Frank': fit_frank,
            'StudentT': fit_student,
            'Gaussian': fit_gaussian
        }
    u, v = _validate_pobs(u, v)
    uv = np.column_stack((u, v))
    best = None
    best_aic = np.inf
    for name, fit_func in candidates.items():
        cop = fit_func(u, v, validate=False)
        ll = np.sum(cop.logpdf(uv))
        # number of parameters: Gumbel=1, Frank=1, StudentT=2 (rho, df), Gaussian=1
        k = 2 if name == 'StudentT' else 1
        aic = 2 * k - 2 * ll
        if aic < best_aic:
            best_aic = aic
            best = cop
    return best

# =============================
# Extended Smoke Test
# =============================

if __name__ == '__main__':
    print("Running enhanced copula_utils extended tests...")
    np.random.seed(0)
    # Test on random uniforms (should pick Gaussian independence)
    u = np.random.rand(200)
    v = np.random.rand(200)
    for func in [fit_gumbel, fit_frank, fit_student, fit_gaussian]:
        cop = func(u, v)
        print(f"{cop.__class__.__name__} logpdf mean: {np.mean(cop.logpdf(np.column_stack((u, v)))): .4f}")
    # Test AIC selection
    selected = select_best_copula(u, v)
    print(f"Auto-selected copula: {selected.__class__.__name__}")

    # TODO: Add tests with known dependence structures
    print("All enhanced tests completed.")