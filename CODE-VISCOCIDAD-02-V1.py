from __future__ import annotations

"""
================================================================================
ARTICLE 2 LONG-MEMORY PIPELINE
================================================================================

This script implements a complete research-oriented numerical pipeline for the
comparative study of creep responses in viscoelastic materials, with emphasis
on long-memory behavior and tail diagnostics in logarithmic scales.

GENERAL PURPOSE OF THE CODE
---------------------------
The main objective of this program is to compare:

1. A trusted fractional Kelvin-Voigt creep response,
2. A best-fit classical Kelvin-Voigt model,
3. A best-fit same-family fractional refit,

under several observation horizons, in order to determine whether the
classical model can reproduce the long-time hereditary behavior of the
fractional model.

SCIENTIFIC IDEA BEHIND THE SCRIPT
---------------------------------
A fractional viscoelastic model can generate creep responses with long-memory
tails that decay approximately as power laws. By contrast, the classical
Kelvin-Voigt model produces exponential relaxation of the normalized deficit.
This difference is often difficult to appreciate on ordinary linear plots,
especially over short time windows. However, once the time horizon is extended
and the remaining distance to equilibrium is analyzed in log-log coordinates,
the difference becomes structurally evident.

MAIN TASKS PERFORMED BY THE SCRIPT
----------------------------------
The script is organized into the following conceptual stages:

A. Stable numerical evaluation of the Mittag-Leffler function on the negative
   real axis. This is essential because the fractional creep response depends
   directly on this special function.

B. Construction of fractional and classical Kelvin-Voigt creep curves.

C. Fitting of one-parameter classical and fractional models to a trusted
   fractional reference signal.

D. Computation of global, windowed, and tail-sensitive diagnostic metrics.

E. Construction of publication-ready figures and CSV summary tables.

F. Export of a manifest file that records all generated artifacts.

EXPECTED OUTPUTS
----------------
When the script is executed, it creates:
- CSV tables with global metrics, windowed metrics, and tail diagnostics.
- PNG figures summarizing the progressive horizon comparison.
- A JSON manifest file listing all generated tables and figures.

OUTPUT DIRECTORY STRUCTURE
--------------------------
The script writes results into a directory such as:

    /mnt/data/article2_outputs/
        figures/
        tables/
        manifest.json

This organization allows the numerical experiment to be reused in reports,
articles, or supplementary scientific material.

NOTES ABOUT STYLE AND DESIGN
----------------------------
This code is intentionally written in a research-friendly and didactic manner:
- The workflow is modular.
- Each block corresponds to a methodological stage of the study.
- Comments explain not only what is done, but also why it is done.
- The code is designed to remain readable, reproducible, and extensible.

================================================================================
"""

import json
import math
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import mpmath as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================================================================
# SECTION 1. STABLE MITTAG-LEFFLER EVALUATION ON THE NEGATIVE REAL AXIS
# ==============================================================================
#
# The fractional Kelvin-Voigt creep formula depends on the Mittag-Leffler
# function E_alpha(-x). In practice, this function is numerically delicate:
# - near the origin, a direct power series is convenient;
# - for larger values of x, an asymptotic inverse-power expansion is better.
#
# This section implements a hybrid evaluator that automatically switches
# between both approximations depending on the scale of x and the value of
# alpha.
#
# The main scientific reason for this block is that the conclusions of the
# article depend on long-time tail behavior, and such tails can be corrupted
# if the special function is evaluated poorly.
# ==============================================================================

def _safe_float(x: mp.mpf | float) -> float:
    """
    Safely convert an mpmath high-precision value to a standard Python float.

    WHY THIS FUNCTION EXISTS
    ------------------------
    During high-precision numerical computations, values may occasionally be
    represented internally in ways that cannot be converted cleanly to a
    regular float. Instead of allowing the entire workflow to crash, this
    helper uses a defensive strategy.

    PARAMETERS
    ----------
    x : mp.mpf or float
        Value to be converted.

    RETURNS
    -------
    float
        The converted floating-point value. If conversion fails, the function
        returns 0.0 as a safe fallback.

    COMMENT
    -------
    Returning 0.0 is not meant to be a mathematically exact substitute in all
    situations, but it prevents isolated conversion failures from breaking the
    entire article-generation pipeline.
    """
    try:
        return float(x)
    except Exception:
        return 0.0


@lru_cache(maxsize=256)
def _inv_gamma_coefficients(alpha: float, max_terms: int = 24, dps: int = 80) -> Tuple[float, ...]:
    """
    Precompute and cache the coefficients required by the asymptotic expansion
    of the Mittag-Leffler function.

    MATHEMATICAL CONTEXT
    --------------------
    For large x, the function E_alpha(-x) can be approximated through an
    inverse-power expansion involving coefficients of the form:

        1 / Gamma(1 - alpha * k)

    Since the same alpha values are repeatedly used throughout scenario fitting,
    plotting, and diagnostic calculations, it is efficient to compute these
    coefficients once and cache them.

    PARAMETERS
    ----------
    alpha : float
        Fractional order of the model.
    max_terms : int, default=24
        Maximum number of asymptotic terms to generate.
    dps : int, default=80
        Decimal precision used internally by mpmath.

    RETURNS
    -------
    Tuple[float, ...]
        A tuple of asymptotic coefficients suitable for repeated reuse.
    """
    mp.mp.dps = int(dps)
    a = mp.mpf(alpha)
    coeffs: List[float] = []

    for k in range(1, int(max_terms) + 1):
        # The asymptotic series uses coefficients based on Gamma(1 - alpha*k).
        arg = mp.mpf("1.0") - a * k
        try:
            value = 1.0 / mp.gamma(arg)
        except ValueError:
            # If the Gamma argument becomes invalid, the expansion is truncated.
            break

        if not mp.isfinite(value):
            # If the computed value is not finite, we stop safely.
            break

        coeffs.append(float(value))

    return tuple(coeffs)


def ml_power_series_negative_real(
    x: float,
    alpha: float,
    tol: float = 1e-15,
    max_terms: int = 12000,
    dps: int = 80,
) -> float:
    """
    Evaluate E_alpha(-x) through a high-precision power series.

    WHEN THIS BRANCH IS USED
    ------------------------
    This branch is especially useful for small or moderate x, where the power
    series converges reliably.

    MATHEMATICAL FORM
    -----------------
    The Mittag-Leffler function is defined as:

        E_alpha(z) = sum_{k=0}^\infty z^k / Gamma(alpha*k + 1)

    In this script, z = -x with x >= 0.

    PARAMETERS
    ----------
    x : float
        Positive real value appearing in E_alpha(-x).
    alpha : float
        Fractional order parameter.
    tol : float, default=1e-15
        Threshold used to stop the summation once the incremental term becomes
        negligibly small.
    max_terms : int, default=12000
        Maximum number of terms allowed in the series.
    dps : int, default=80
        Working decimal precision.

    RETURNS
    -------
    float
        Approximation of E_alpha(-x), clipped into [0, 1] for physical stability
        in the present creep context.
    """
    if x <= 0.0:
        return 1.0

    mp.mp.dps = int(dps)
    z = -mp.mpf(x)
    a = mp.mpf(alpha)

    term = mp.mpf("1.0")
    total = term

    for k in range(1, int(max_terms) + 1):
        term = (z ** k) / mp.gamma(a * k + 1)
        total += term

        # Once the newest term is smaller than the tolerance, the sum is
        # considered sufficiently converged for the present application.
        if mp.fabs(term) < tol:
            break

    value = _safe_float(total)

    # The clipping is used because, in this constitutive application,
    # E_alpha(-x) is expected to behave as a bounded positive quantity.
    return min(1.0, max(0.0, value))


def ml_asymptotic_negative_real(
    x: float,
    alpha: float,
    max_terms: int = 18,
    dps: int = 80,
) -> float:
    """
    Evaluate E_alpha(-x) through an inverse-power asymptotic expansion.

    WHEN THIS BRANCH IS USED
    ------------------------
    For larger x values, the direct power series can become inefficient or
    numerically unstable. In that regime, a truncated asymptotic expansion in
    inverse powers of x is more appropriate.

    PARAMETERS
    ----------
    x : float
        Positive argument in E_alpha(-x).
    alpha : float
        Fractional order.
    max_terms : int, default=18
        Number of asymptotic terms.
    dps : int, default=80
        Decimal precision used for coefficient generation.

    RETURNS
    -------
    float
        Approximation of E_alpha(-x), clipped to [0, 1].
    """
    if x <= 0.0:
        return 1.0

    coeffs = _inv_gamma_coefficients(float(alpha), max_terms=int(max_terms), dps=int(dps))

    total = 0.0
    sign = 1.0
    x_pow = float(x)

    for coeff in coeffs:
        total += sign * coeff / x_pow
        sign *= -1.0
        x_pow *= float(x)

    return min(1.0, max(0.0, float(total)))


def adaptive_series_switch(alpha: float) -> float:
    """
    Determine the crossover value between the power-series branch and the
    asymptotic branch.

    IDEA
    ----
    The region where the power series remains efficient depends on alpha.
    This function implements empirical thresholds that work well for the
    current study.

    PARAMETERS
    ----------
    alpha : float
        Fractional order.

    RETURNS
    -------
    float
        Threshold value x_switch such that:
        - if x < x_switch, use the power series,
        - otherwise, use the asymptotic expansion.
    """
    a = float(alpha)

    if a >= 0.90:
        return 8.0
    if a >= 0.80:
        return 6.0
    if a >= 0.60:
        return 4.0
    if a >= 0.40:
        return 3.0
    return 2.0


def ml_stable_hybrid_negative_real(
    x: float,
    alpha: float,
    series_tol: float = 1e-15,
    series_max_terms: int = 12000,
    asymp_max_terms: int = 18,
    dps: int = 80,
) -> float:
    """
    Hybrid stable evaluator for E_alpha(-x).

    WORKFLOW
    --------
    1. If x <= 0, the function returns 1.0, consistent with E_alpha(0)=1.
    2. It chooses a switching threshold depending on alpha.
    3. For smaller x values, it uses the power series.
    4. For larger x values, it uses the asymptotic branch.

    PARAMETERS
    ----------
    x : float
        Nonnegative real argument.
    alpha : float
        Fractional order.
    series_tol : float
        Tolerance for the power series branch.
    series_max_terms : int
        Maximum number of terms in the series.
    asymp_max_terms : int
        Maximum number of terms in the asymptotic expansion.
    dps : int
        Decimal precision.

    RETURNS
    -------
    float
        Stable approximation of E_alpha(-x).
    """
    if x <= 0.0:
        return 1.0

    x_switch = adaptive_series_switch(float(alpha))

    if x < x_switch:
        return ml_power_series_negative_real(
            x=x,
            alpha=alpha,
            tol=series_tol,
            max_terms=series_max_terms,
            dps=dps,
        )

    return ml_asymptotic_negative_real(
        x=x,
        alpha=alpha,
        max_terms=asymp_max_terms,
        dps=dps,
    )


# ==============================================================================
# SECTION 2. FRACTIONAL AND CLASSICAL KELVIN-VOIGT CREEP MODELS
# ==============================================================================
#
# This section contains the constitutive models used throughout the paper.
# The core comparison is between:
#
# - A fractional Kelvin-Voigt element:
#       epsilon(t) = (sigma0 / E) [1 - E_alpha(-(E/eta_alpha) t^alpha)]
#
# - A classical Kelvin-Voigt element:
#       epsilon(t) = (sigma0 / E) [1 - exp(-t/tau)]
#
# These functions generate the synthetic curves later used for fitting,
# diagnostics, and plotting.
# ==============================================================================

def creep_fractional_kelvin_voigt(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta_alpha: float,
    alpha: float,
    evaluator: Callable[[float, float], float],
) -> np.ndarray:
    """
    Compute the fractional Kelvin-Voigt creep response under step stress.

    MODEL
    -----
    The constitutive formula implemented here is:

        epsilon(t) = (sigma0 / E) * [1 - E_alpha(-(E/eta_alpha) * t^alpha)]

    where:
    - sigma0 is the step stress magnitude,
    - E is the elastic modulus,
    - eta_alpha is the fractional viscosity parameter,
    - alpha is the memory order,
    - E_alpha is the Mittag-Leffler function.

    PARAMETERS
    ----------
    t : np.ndarray
        Time vector.
    sigma0 : float
        Applied stress magnitude.
    E : float
        Elastic modulus.
    eta_alpha : float
        Fractional viscosity.
    alpha : float
        Fractional order.
    evaluator : callable
        Function used to evaluate E_alpha(-x).

    RETURNS
    -------
    np.ndarray
        Fractional creep strain history.

    COMMENT
    -------
    The loop is explicit rather than vectorized because each time point requires
    evaluating the special function with a nonlinear argument depending on t^alpha.
    """
    const = float(E) / float(eta_alpha)
    strain = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if ti <= 0.0:
            # At or before loading onset, creep strain is zero.
            strain[i] = 0.0
        else:
            # Dimensionless argument entering the Mittag-Leffler function.
            x = const * (float(ti) ** float(alpha))

            # Stable special-function evaluation.
            ml = evaluator(x, alpha)

            # Fractional creep formula.
            strain[i] = (float(sigma0) / float(E)) * (1.0 - ml)

    return strain


def creep_classic_kelvin_voigt(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta: float,
) -> np.ndarray:
    """
    Compute the classical Kelvin-Voigt creep response under step stress.

    MODEL
    -----
    The classical Kelvin-Voigt creep law is:

        epsilon(t) = (sigma0 / E) * [1 - exp(-t / tau)]

    with:
        tau = eta / E

    PARAMETERS
    ----------
    t : np.ndarray
        Time vector.
    sigma0 : float
        Applied stress magnitude.
    E : float
        Elastic modulus.
    eta : float
        Classical viscosity parameter.

    RETURNS
    -------
    np.ndarray
        Classical creep strain history.
    """
    tau = float(eta) / float(E)
    return (float(sigma0) / float(E)) * (1.0 - np.exp(-np.asarray(t, dtype=float) / tau))


# ==============================================================================
# SECTION 3. ERROR METRICS AND ONE-DIMENSIONAL PARAMETER FITTING
# ==============================================================================
#
# This block provides:
# - standard error metrics,
# - a simple but robust one-dimensional fitting routine based on:
#     1) coarse scan,
#     2) golden-section refinement.
#
# In the article, the fitted parameters are:
# - eta*      for the classical model,
# - eta_alpha* for the fractional refit.
# ==============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute three standard discrepancy metrics between two curves.

    METRICS
    -------
    - MSE  : Mean Squared Error
    - MAE  : Mean Absolute Error
    - MAXE : Maximum Absolute Error

    PARAMETERS
    ----------
    y_true : np.ndarray
        Reference signal.
    y_pred : np.ndarray
        Predicted or fitted signal.

    RETURNS
    -------
    Tuple[float, float, float]
        (mse, mae, maxe)
    """
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)

    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    maxe = float(np.max(np.abs(error)))

    return mse, mae, maxe


def golden_fit_1d(
    objective: Callable[[float], float],
    p_min: float,
    p_max: float,
    coarse_n: int = 80,
    golden_iter: int = 20,
) -> Tuple[float, float]:
    """
    Fit one scalar parameter using a two-stage derivative-free strategy.

    STAGE 1: COARSE GRID SEARCH
    ---------------------------
    A uniform grid is explored first to identify a promising interval.

    STAGE 2: GOLDEN-SECTION SEARCH
    ------------------------------
    The method then refines the optimum inside that interval.

    PARAMETERS
    ----------
    objective : callable
        Scalar objective function to minimize.
    p_min : float
        Lower bound of the parameter domain.
    p_max : float
        Upper bound of the parameter domain.
    coarse_n : int, default=80
        Number of coarse grid points.
    golden_iter : int, default=20
        Number of golden-section refinement iterations.

    RETURNS
    -------
    Tuple[float, float]
        (best_parameter, best_objective_value)
    """
    grid = np.linspace(float(p_min), float(p_max), int(coarse_n))
    values = np.array([objective(float(p)) for p in grid], dtype=float)

    idx = int(np.argmin(values))

    # Select a bracket around the best coarse point.
    if idx == 0:
        a, b = float(grid[0]), float(grid[1])
    elif idx == len(grid) - 1:
        a, b = float(grid[-2]), float(grid[-1])
    else:
        a, b = float(grid[idx - 1]), float(grid[idx + 1])

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    c = b - resphi * (b - a)
    d = a + resphi * (b - a)

    fc = objective(c)
    fd = objective(d)

    for _ in range(int(golden_iter)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - resphi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + resphi * (b - a)
            fd = objective(d)

    best_p = float(c if fc < fd else d)
    best_value = float(min(fc, fd))

    return best_p, best_value


# ==============================================================================
# SECTION 4. SCENARIO CONFIGURATION
# ==============================================================================
#
# The article compares three stress-test scenarios:
# - S1: short horizon,
# - S2: moderate horizon,
# - S3: long horizon.
#
# This section defines a lightweight immutable data structure for scenarios,
# along with a function that returns the default scenario list.
# ==============================================================================

@dataclass(frozen=True)
class Scenario:
    """
    Immutable container storing the configuration of one numerical scenario.

    FIELDS
    ------
    code : str
        Scenario identifier, e.g. "S1", "S2", "S3".
    title : str
        Descriptive scenario title.
    alpha_true : float
        Ground-truth fractional order used to generate the reference signal.
    t_max : float
        Maximum observation time.
    n_points : int
        Number of time discretization points.
    sigma0 : float
        Applied stress.
    E : float
        Elastic modulus.
    eta_alpha_true : float
        Ground-truth fractional viscosity.
    eta_class_bounds : tuple
        Search interval for the classical viscosity fit.
    eta_frac_bounds : tuple
        Search interval for the fractional refit viscosity.
    """
    code: str
    title: str
    alpha_true: float
    t_max: float
    n_points: int
    sigma0: float = 1.0
    E: float = 1000.0
    eta_alpha_true: float = 200.0
    eta_class_bounds: Tuple[float, float] = (25.0, 6000.0)
    eta_frac_bounds: Tuple[float, float] = (50.0, 500.0)


def default_scenarios() -> List[Scenario]:
    """
    Return the default sequence of scenarios used in the study.

    SCIENTIFIC PURPOSE
    ------------------
    The progression from S1 to S3 is designed to test whether the fitted
    classical model remains stable as the time horizon expands and the tail
    becomes more important.

    RETURNS
    -------
    List[Scenario]
        List of the three canonical scenarios.
    """
    return [
        Scenario(code="S1", title="Short horizon baseline", alpha_true=0.65, t_max=50.0, n_points=700),
        Scenario(code="S2", title="Moderate horizon stress test", alpha_true=0.55, t_max=200.0, n_points=900),
        Scenario(code="S3", title="Long horizon tail emphasis", alpha_true=0.25, t_max=2000.0, n_points=1200),
    ]


# ==============================================================================
# SECTION 5. HELPER CALCULATIONS AND DIAGNOSTIC TRANSFORMATIONS
# ==============================================================================
#
# This section groups the secondary calculations used across the analysis:
# - fitting both models for one scenario,
# - normalized strain,
# - normalized deficit,
# - definition of early/intermediate/late windows,
# - log-log tail regression,
# - surrogate regression for the classical exponential tail.
# ==============================================================================

def fit_models(cfg: Scenario) -> Dict[str, object]:
    """
    Run the full fitting cycle for a given scenario.

    WORKFLOW
    --------
    1. Create the time vector.
    2. Generate the trusted fractional reference curve.
    3. Fit the best classical Kelvin-Voigt viscosity eta.
    4. Fit the best same-family fractional viscosity eta_alpha.
    5. Reconstruct both fitted curves.

    PARAMETERS
    ----------
    cfg : Scenario
        Scenario configuration object.

    RETURNS
    -------
    Dict[str, object]
        Dictionary containing time vector, reference curve, fitted curves, and
        fitted parameter values.
    """
    t = np.linspace(0.0, cfg.t_max, cfg.n_points)

    # Generate the trusted reference signal from the fractional model using
    # the known ground-truth parameters of the scenario.
    eps_ref = creep_fractional_kelvin_voigt(
        t=t,
        sigma0=cfg.sigma0,
        E=cfg.E,
        eta_alpha=cfg.eta_alpha_true,
        alpha=cfg.alpha_true,
        evaluator=ml_stable_hybrid_negative_real,
    )

    # Objective function for the classical fit.
    obj_class = lambda eta: np.mean(
        (creep_classic_kelvin_voigt(t, cfg.sigma0, cfg.E, eta) - eps_ref) ** 2
    )

    # Objective function for the same-family fractional refit.
    obj_frac = lambda eta: np.mean(
        (
            creep_fractional_kelvin_voigt(
                t,
                cfg.sigma0,
                cfg.E,
                eta,
                cfg.alpha_true,
                ml_stable_hybrid_negative_real,
            ) - eps_ref
        ) ** 2
    )

    # Perform one-dimensional parameter fitting.
    eta_class, _ = golden_fit_1d(obj_class, cfg.eta_class_bounds[0], cfg.eta_class_bounds[1])
    eta_frac, _ = golden_fit_1d(obj_frac, cfg.eta_frac_bounds[0], cfg.eta_frac_bounds[1])

    # Rebuild fitted curves.
    eps_class = creep_classic_kelvin_voigt(t, cfg.sigma0, cfg.E, eta_class)
    eps_frac = creep_fractional_kelvin_voigt(
        t,
        cfg.sigma0,
        cfg.E,
        eta_frac,
        cfg.alpha_true,
        ml_stable_hybrid_negative_real,
    )

    return {
        "t": t,
        "eps_ref": eps_ref,
        "eps_class": eps_class,
        "eps_frac": eps_frac,
        "eta_class": eta_class,
        "eta_frac": eta_frac,
    }


def normalized_strain(eps: np.ndarray, sigma0: float, E: float) -> np.ndarray:
    """
    Convert strain into the normalized quantity E * epsilon / sigma0.

    This normalization is useful because it makes the equilibrium level equal
    to 1 for the present creep setting, simplifying interpretation.

    PARAMETERS
    ----------
    eps : np.ndarray
        Strain history.
    sigma0 : float
        Applied stress.
    E : float
        Elastic modulus.

    RETURNS
    -------
    np.ndarray
        Normalized strain.
    """
    return (E / sigma0) * np.asarray(eps, dtype=float)


def normalized_deficit(eps: np.ndarray, sigma0: float, E: float) -> np.ndarray:
    """
    Compute the normalized deficit:

        Delta(t) = 1 - E*epsilon(t)/sigma0

    INTERPRETATION
    --------------
    This is the remaining distance to the asymptotic equilibrium level.
    It is the key quantity used in the tail analysis of the article.

    A very small lower bound is imposed to prevent numerical problems when
    logarithms are later applied.

    PARAMETERS
    ----------
    eps : np.ndarray
        Strain history.
    sigma0 : float
        Applied stress.
    E : float
        Elastic modulus.

    RETURNS
    -------
    np.ndarray
        Normalized deficit clipped below by 1e-300.
    """
    return np.clip(1.0 - normalized_strain(eps, sigma0, E), 1e-300, None)


def relative_windows(t_max: float) -> List[Tuple[str, float, float]]:
    """
    Define the three relative windows used for local error analysis.

    WINDOWS
    -------
    W1 = [0, 0.1*t_max]
    W2 = [0.1*t_max, 0.5*t_max]
    W3 = [0.5*t_max, t_max]

    These windows allow the code to examine whether the model discrepancy is
    concentrated early, in the middle, or in the tail.

    PARAMETERS
    ----------
    t_max : float
        Maximum time horizon.

    RETURNS
    -------
    List[Tuple[str, float, float]]
        List of labeled intervals.
    """
    return [
        ("W1", 0.0, 0.1 * t_max),
        ("W2", 0.1 * t_max, 0.5 * t_max),
        ("W3", 0.5 * t_max, t_max),
    ]


def tail_regression_loglog(t: np.ndarray, deficit: np.ndarray, t_start: float) -> Dict[str, float]:
    """
    Perform a linear regression in log-log space over the tail interval.

    MODEL
    -----
    The regression approximates:

        log10(Delta(t)) = beta * log10(t) + intercept

    This is used to estimate the apparent power-law slope of the tail.

    PARAMETERS
    ----------
    t : np.ndarray
        Time vector (strictly positive in practice for log usage).
    deficit : np.ndarray
        Normalized deficit.
    t_start : float
        Start of the tail interval.

    RETURNS
    -------
    Dict[str, float]
        Dictionary containing:
        - slope
        - intercept
        - R2
    """
    mask = np.asarray(t) >= float(t_start)

    x = np.log10(np.asarray(t)[mask])
    y = np.log10(np.asarray(deficit)[mask])

    coeff = np.polyfit(x, y, 1)
    yhat = coeff[0] * x + coeff[1]

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return {"slope": float(coeff[0]), "intercept": float(coeff[1]), "R2": float(r2)}


def classical_logdef_regression(t: np.ndarray, tau: float, t_start: float) -> Dict[str, float]:
    """
    Build a surrogate regression summary for the classical exponential deficit.

    WHY THIS IS NEEDED
    ------------------
    The classical model does not possess a genuine stationary power-law tail.
    However, for comparison purposes, the code fits a linear trend in the
    transformed log-log domain to show how distorted the classical behavior is
    when forced into the same representation.

    PARAMETERS
    ----------
    t : np.ndarray
        Time vector.
    tau : float
        Classical retardation time eta / E.
    t_start : float
        Start of the tail interval.

    RETURNS
    -------
    Dict[str, float]
        Dictionary with surrogate slope, intercept, and R2.
    """
    mask = np.asarray(t) >= float(t_start)

    x = np.log10(np.asarray(t)[mask])

    # For the exponential deficit exp(-t/tau), taking log10 gives:
    # log10(exp(-t/tau)) = -(t/tau) / ln(10)
    y = -(np.asarray(t)[mask] / float(tau)) / math.log(10.0)

    coeff = np.polyfit(x, y, 1)
    yhat = coeff[0] * x + coeff[1]

    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return {"slope": float(coeff[0]), "intercept": float(coeff[1]), "R2": float(r2)}


# ==============================================================================
# SECTION 6. PLOTTING UTILITIES
# ==============================================================================
#
# This section generates the article-style figures:
# - progressive linear comparison,
# - absolute error comparison,
# - log-log tail view,
# - local slope magnitude,
# - parameter drift,
# - family of alpha-dependent tail curves.
#
# Each plotting routine creates one specific figure and saves it to disk.
# ==============================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    """
    Save a matplotlib figure with consistent layout and cleanup.

    PARAMETERS
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : pathlib.Path
        Target file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_progressive_linear(results: Dict[str, Dict[str, object]], outpath: Path) -> None:
    """
    Plot normalized creep curves in linear coordinates for all scenarios.

    INTERPRETATION
    --------------
    This figure answers the first visual question:
    do the models look similar under ordinary plotting conditions?

    PARAMETERS
    ----------
    results : dict
        Scenario results dictionary.
    outpath : Path
        Destination path for the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))

    for ax, (code, r) in zip(axes, results.items()):
        cfg = r["cfg"]
        t = r["t"]

        ax.plot(
            t,
            normalized_strain(r["eps_ref"], cfg.sigma0, cfg.E),
            label="Fractional reference",
            linewidth=2.1,
        )
        ax.plot(
            t,
            normalized_strain(r["eps_class"], cfg.sigma0, cfg.E),
            "--",
            label="Best classical fit",
            linewidth=2.1,
        )
        ax.plot(
            t,
            normalized_strain(r["eps_frac"], cfg.sigma0, cfg.E),
            ":",
            label="Best fractional refit",
            linewidth=2.4,
        )

        ax.set_title(f"{code}: α={cfg.alpha_true:.2f}, $t_{{max}}$={cfg.t_max:g} s", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Normalized strain $E\,\varepsilon/\sigma_0$")
        ax.grid(True, alpha=0.28)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    _save(fig, outpath)


def plot_progressive_error(results: Dict[str, Dict[str, object]], outpath: Path) -> None:
    """
    Plot absolute error histories on a semilogarithmic scale.

    INTERPRETATION
    --------------
    This figure reveals whether the classical and fractional fits differ by
    small or large error orders of magnitude across time.

    PARAMETERS
    ----------
    results : dict
        Scenario results dictionary.
    outpath : Path
        Destination path for the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))

    for ax, (code, r) in zip(axes, results.items()):
        cfg = r["cfg"]
        t = r["t"]

        err_class = np.abs(r["eps_class"] - r["eps_ref"])
        err_frac = np.abs(r["eps_frac"] - r["eps_ref"])

        ax.semilogy(t[1:], err_class[1:], "--", label="Classical abs. error", linewidth=2.0)
        ax.semilogy(t[1:], err_frac[1:], ":", label="Fractional abs. error", linewidth=2.3)

        ax.set_title(f"{code}: α={cfg.alpha_true:.2f}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Absolute error $|\varepsilon-\varepsilon^{ref}|$")
        ax.grid(True, alpha=0.28)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    _save(fig, outpath)


def plot_tail_loglog(results: Dict[str, Dict[str, object]], outpath: Path) -> None:
    """
    Plot the normalized deficit in log-log scale over the tail interval.

    INTERPRETATION
    --------------
    This is one of the most important figures of the article because it shows:
    - whether the fractional tail aligns with a power-law guide,
    - how rapidly the classical exponential tail collapses.

    PARAMETERS
    ----------
    results : dict
        Scenario results dictionary.
    outpath : Path
        Destination path for the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))

    for ax, (code, r) in zip(axes, results.items()):
        cfg = r["cfg"]

        t = r["t"].copy()

        # The first point may be zero, which is incompatible with logarithms.
        # Therefore it is replaced by a small positive value.
        t[0] = max(t[1] * 0.5, 1e-6)

        tail_mask = t >= 0.1 * cfg.t_max
        t_tail = t[tail_mask]

        d_ref = normalized_deficit(r["eps_ref"], cfg.sigma0, cfg.E)[tail_mask]
        d_class = np.exp(-t_tail / (r["eta_class"] / cfg.E))
        d_frac = normalized_deficit(r["eps_frac"], cfg.sigma0, cfg.E)[tail_mask]

        # A reference guide of the form t^{-alpha} is included to visually
        # compare the estimated fractional tail with the expected power-law trend.
        guide = d_ref[-1] * (t_tail / t_tail[-1]) ** (-cfg.alpha_true)

        ax.loglog(t_tail, d_ref, label="Fractional reference", linewidth=2.1)
        ax.loglog(t_tail, d_class, "--", label="Best classical fit", linewidth=2.1)
        ax.loglog(t_tail, d_frac, ":", label="Best fractional refit", linewidth=2.4)
        ax.loglog(t_tail, guide, linestyle=(0, (1, 2)), linewidth=1.4, label=r"Tail guide $t^{-\alpha}$")

        ymin = max(min(d_ref.min(), d_frac.min()) / 8.0, 1e-6)
        ymax = min(max(d_ref.max(), d_frac.max()) * 1.25, 1.2)

        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{code}: α={cfg.alpha_true:.2f}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Normalized deficit $1-E\,\varepsilon/\sigma_0$")
        ax.grid(True, which="both", alpha=0.28)

    handles, labels = axes[-1].get_legend_handles_labels()

    # Deduplicate legend entries because repeated labels occur across subplots.
    seen = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            dedup_h.append(h)
            dedup_l.append(l)

    fig.legend(dedup_h, dedup_l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    _save(fig, outpath)


def plot_local_slope(results: Dict[str, Dict[str, object]], outpath: Path) -> None:
    """
    Plot the positive magnitude of the local log-log slope:

        q(t) = - d ln Delta / d ln t

    INTERPRETATION
    --------------
    This diagnostic reveals whether the tail approaches a stable exponent.
    For the fractional case, q(t) should remain near alpha in the tail.
    For the classical exponential case, q(t) = t/tau grows linearly with time.

    PARAMETERS
    ----------
    results : dict
        Scenario results dictionary.
    outpath : Path
        Destination path for the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))

    for ax, (code, r) in zip(axes, results.items()):
        cfg = r["cfg"]

        t = r["t"].copy()
        t[0] = max(t[1] * 0.5, 1e-6)

        tail_mask = t >= 0.1 * cfg.t_max
        t_tail = t[tail_mask]

        d_ref = normalized_deficit(r["eps_ref"], cfg.sigma0, cfg.E)[tail_mask]
        d_frac = normalized_deficit(r["eps_frac"], cfg.sigma0, cfg.E)[tail_mask]

        tau = r["eta_class"] / cfg.E

        # Numerical differentiation in log-log coordinates.
        s_ref = -np.gradient(np.log(d_ref), np.log(t_tail))
        s_frac = -np.gradient(np.log(d_frac), np.log(t_tail))

        # For the classical exponential deficit, the analytical slope magnitude is t/tau.
        s_class = t_tail / tau

        ax.loglog(t_tail, s_ref, label="Fractional reference", linewidth=2.1)
        ax.loglog(t_tail, s_class, "--", label="Best classical fit", linewidth=2.1)
        ax.loglog(t_tail, s_frac, ":", label="Best fractional refit", linewidth=2.4)
        ax.axhline(cfg.alpha_true, linestyle=(0, (1, 2)), linewidth=1.4, label=r"Target level $\alpha$")

        ax.set_title(f"{code}: α={cfg.alpha_true:.2f}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Slope magnitude $-d\ln \Delta / d\ln t$")
        ax.grid(True, which="both", alpha=0.28)

    handles, labels = axes[-1].get_legend_handles_labels()

    seen = {}
    dedup_h, dedup_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            dedup_h.append(h)
            dedup_l.append(l)

    fig.legend(dedup_h, dedup_l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    _save(fig, outpath)


def plot_parameter_drift(results: Dict[str, Dict[str, object]], outpath: Path) -> None:
    """
    Plot fitted parameter drift as the observation horizon increases.

    INTERPRETATION
    --------------
    A structurally adequate model should exhibit parameter stability when the
    observation horizon grows. Large drift is evidence of structural mismatch.

    PARAMETERS
    ----------
    results : dict
        Scenario results dictionary.
    outpath : Path
        Destination path for the figure.
    """
    tmax = [r["cfg"].t_max for r in results.values()]
    eta_class = [r["eta_class"] for r in results.values()]
    eta_frac = [r["eta_frac"] for r in results.values()]

    fig = plt.figure(figsize=(6.2, 4.0))

    plt.plot(tmax, eta_class, marker="o", linewidth=2.1, label="Best classical viscosity η*")
    plt.plot(tmax, eta_frac, marker="s", linewidth=2.1, label=r"Best fractional viscosity $\eta_\alpha^*$")

    plt.xscale("log")
    plt.xlabel(r"Observation horizon $t_{max}$ (s)")
    plt.ylabel("Best-fit parameter")
    plt.grid(True, which="both", alpha=0.28)
    plt.legend(frameon=False)

    _save(fig, outpath)


def plot_alpha_family(outpath: Path, sigma0: float = 1.0, E: float = 1000.0, eta_alpha: float = 200.0) -> None:
    """
    Plot a family of normalized deficit curves for several alpha values.

    INTERPRETATION
    --------------
    This auxiliary figure isolates the role of alpha in shaping the tail:
    smaller alpha produces heavier, slower-decaying tails.

    PARAMETERS
    ----------
    outpath : Path
        Destination path for the figure.
    sigma0 : float, default=1.0
        Applied stress.
    E : float, default=1000.0
        Elastic modulus.
    eta_alpha : float, default=200.0
        Fractional viscosity.
    """
    t = np.logspace(-2, math.log10(2000.0), 1200)
    alphas = [0.25, 0.40, 0.55, 0.70, 0.85]

    fig = plt.figure(figsize=(6.2, 4.0))

    for alpha in alphas:
        eps = creep_fractional_kelvin_voigt(t, sigma0, E, eta_alpha, alpha, ml_stable_hybrid_negative_real)
        deficit = normalized_deficit(eps, sigma0, E)
        plt.loglog(t, deficit, linewidth=2.0, label=fr"$\alpha={alpha:.2f}$")

    plt.xlabel("Time (s)")
    plt.ylabel(r"Normalized deficit $1-E\,\varepsilon/\sigma_0$")
    plt.grid(True, which="both", alpha=0.28)
    plt.legend(frameon=False, ncol=2)

    _save(fig, outpath)


# ==============================================================================
# SECTION 7. MAIN EXECUTION PIPELINE
# ==============================================================================
#
# This final block orchestrates the complete workflow:
# 1. Create output folders.
# 2. Load default scenarios.
# 3. Fit all models in every scenario.
# 4. Build summary tables.
# 5. Generate all figures.
# 6. Save a manifest with the produced artifacts.
#
# This function is the computational backbone of the article.
# ==============================================================================

def main(outdir: str = "/mnt/data/article2_outputs") -> None:
    """
    Execute the full experiment and write all outputs to disk.

    PARAMETERS
    ----------
    outdir : str, default="/mnt/data/article2_outputs"
        Root output directory where figures, tables, and the manifest
        will be stored.

    RETURNS
    -------
    None
    """
    out = Path(outdir)
    figdir = out / "figures"
    tabdir = out / "tables"

    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    scenarios = default_scenarios()
    results: Dict[str, Dict[str, object]] = {}

    # These lists accumulate row dictionaries that will later become CSV tables.
    global_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    tail_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []

    # --------------------------------------------------------------------------
    # LOOP OVER THE MAIN SCENARIOS
    # --------------------------------------------------------------------------
    for cfg in scenarios:
        # Run the model generation and fitting cycle for one scenario.
        r = fit_models(cfg)
        r["cfg"] = cfg
        results[cfg.code] = r

        # ----------------------------------------------------------------------
        # GLOBAL METRICS TABLE
        # ----------------------------------------------------------------------
        mse_class, mae_class, maxe_class = compute_metrics(r["eps_ref"], r["eps_class"])
        mse_frac, mae_frac, maxe_frac = compute_metrics(r["eps_ref"], r["eps_frac"])

        global_rows.append({
            "Scenario": cfg.code,
            "Description": cfg.title,
            "alpha": cfg.alpha_true,
            "t_max_s": cfg.t_max,
            "eta_class_best": r["eta_class"],
            "eta_alpha_best": r["eta_frac"],
            "MSE_class": mse_class,
            "MAE_class": mae_class,
            "MAXE_class": maxe_class,
            "MSE_frac": mse_frac,
            "MAE_frac": mae_frac,
            "MAXE_frac": maxe_frac,
            "MAE_improvement": mae_class / mae_frac,
        })

        # ----------------------------------------------------------------------
        # WINDOWED METRICS TABLE
        # ----------------------------------------------------------------------
        windows = relative_windows(cfg.t_max)

        for label, a, b in windows:
            mask = (r["t"] >= a) & (r["t"] <= b)

            _, mae_c, _ = compute_metrics(r["eps_ref"][mask], r["eps_class"][mask])
            _, mae_f, _ = compute_metrics(r["eps_ref"][mask], r["eps_frac"][mask])

            window_rows.append({
                "Scenario": cfg.code,
                "alpha": cfg.alpha_true,
                "t_max_s": cfg.t_max,
                "Window": label,
                "Interval_s": f"{a:g}-{b:g}",
                "MAE_class": mae_c,
                "MAE_frac": mae_f,
                "Improvement_factor": mae_c / mae_f,
            })

        # ----------------------------------------------------------------------
        # TAIL DIAGNOSTICS TABLE
        # ----------------------------------------------------------------------
        t_pos = r["t"].copy()

        # Ensure positivity for logarithms.
        t_pos[0] = max(t_pos[1] * 0.5, 1e-6)

        t_start = 0.1 * cfg.t_max

        ref_tail = tail_regression_loglog(
            t_pos[1:],
            normalized_deficit(r["eps_ref"], cfg.sigma0, cfg.E)[1:],
            t_start,
        )
        frac_tail = tail_regression_loglog(
            t_pos[1:],
            normalized_deficit(r["eps_frac"], cfg.sigma0, cfg.E)[1:],
            t_start,
        )

        tau = r["eta_class"] / cfg.E
        class_tail = classical_logdef_regression(t_pos[1:], tau, t_start)

        tail_rows.append({
            "Scenario": cfg.code,
            "alpha": cfg.alpha_true,
            "t_max_s": cfg.t_max,
            "beta_ref": ref_tail["slope"],
            "R2_ref": ref_tail["R2"],
            "beta_frac": frac_tail["slope"],
            "R2_frac": frac_tail["R2"],
            "beta_class_reg": class_tail["slope"],
            "R2_class_reg": class_tail["R2"],
            "class_local_slope_at_0.1_tmax": -(0.1 * cfg.t_max) / tau,
            "class_local_slope_at_tmax": -(cfg.t_max) / tau,
        })

    # --------------------------------------------------------------------------
    # AUXILIARY ALPHA-FAMILY TABLE
    # --------------------------------------------------------------------------
    t = np.logspace(-2, math.log10(2000.0), 1200)

    for alpha in [0.25, 0.40, 0.55, 0.70, 0.85]:
        eps = creep_fractional_kelvin_voigt(t, 1.0, 1000.0, 200.0, alpha, ml_stable_hybrid_negative_real)
        tail = tail_regression_loglog(t, normalized_deficit(eps, 1.0, 1000.0), 200.0)

        alpha_rows.append({
            "alpha": alpha,
            "beta_hat": tail["slope"],
            "R2": tail["R2"],
        })

    # --------------------------------------------------------------------------
    # CONVERT ROWS INTO DATAFRAMES
    # --------------------------------------------------------------------------
    df_global = pd.DataFrame(global_rows)
    df_window = pd.DataFrame(window_rows)
    df_tail = pd.DataFrame(tail_rows)
    df_alpha = pd.DataFrame(alpha_rows)

    # --------------------------------------------------------------------------
    # WRITE CSV TABLES
    # --------------------------------------------------------------------------
    df_global.to_csv(tabdir / "Table_1_global_metrics.csv", index=False)
    df_window.to_csv(tabdir / "Table_2_windowed_metrics.csv", index=False)
    df_tail.to_csv(tabdir / "Table_3_tail_diagnostics.csv", index=False)
    df_alpha.to_csv(tabdir / "Table_4_alpha_family.csv", index=False)

    # --------------------------------------------------------------------------
    # GENERATE FIGURES
    # --------------------------------------------------------------------------
    plot_progressive_linear(results, figdir / "Fig_1_progressive_linear.png")
    plot_progressive_error(results, figdir / "Fig_2_progressive_error.png")
    plot_tail_loglog(results, figdir / "Fig_3_tail_loglog.png")
    plot_local_slope(results, figdir / "Fig_4_local_slope.png")
    plot_parameter_drift(results, figdir / "Fig_S1_parameter_drift.png")
    plot_alpha_family(figdir / "Fig_5_alpha_family_loglog.png")

    # --------------------------------------------------------------------------
    # WRITE MANIFEST FILE
    # --------------------------------------------------------------------------
    manifest = {
        "scenarios": [asdict(s) for s in scenarios],
        "tables": [p.name for p in sorted(tabdir.glob("*.csv"))],
        "figures": [p.name for p in sorted(figdir.glob("*.png"))],
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
#
# This condition makes the script executable as a standalone program.
# When run directly, it launches the full article pipeline.
# ==============================================================================

if __name__ == "__main__":
    main()