from __future__ import annotations

# ============================================================================
# PROJECT OVERVIEW
# ============================================================================
# This script implements a computational study of viscoelastic stress relaxation
# using both classical and fractional Zener-type constitutive models.
#
# From a mathematical standpoint, the central purpose of the program is to
# compare how a standard exponential relaxation law differs from a fractional
# relaxation law driven by the Mittag-Leffler function, particularly in the
# long-time regime where power-law tails become important.
#
# In practical terms, the code does the following:
# 1. Implements numerically stable evaluation routines for the Mittag-Leffler
#    function on the negative real axis.
# 2. Uses those routines to construct fractional Zener relaxation responses.
# 3. Builds the corresponding classical Zener response for comparison.
# 4. Fits classical and fractional model structures to synthetic reference data.
# 5. Computes global metrics, windowed metrics, late-time tail diagnostics,
#    local slope diagnostics, and horizon-based stress tests.
# 6. Exports the resulting tables and figures to organized output folders.
#
# The code is therefore not just a simulation script. It is a complete analysis
# pipeline that goes from numerical kernel construction, to constitutive-model
# evaluation, to parameter fitting, and finally to scientific-figure/table
# generation for article-quality outputs.
#
# The script is organized in six main conceptual blocks:
#   A. Stable Mittag-Leffler evaluation.
#   B. Fractional and classical Zener relaxation definitions.
#   C. Error metrics and slope diagnostics.
#   D. Parameter estimation utilities.
#   E. Scenario generation and benchmark construction.
#   F. Output production: CSV tables, PNG figures, and a JSON manifest.
#
# All comments are written in English, as requested, and the numerical logic of
# the original program has been preserved.
# ============================================================================

import json
import math
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import mpmath as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================================================================
# Stable Mittag-Leffler evaluation on the negative real axis
# ================================================================
# The fractional Zener model used later in the script depends on the
# Mittag-Leffler function E_alpha(-x), which is the fractional analogue of the
# exponential function appearing in classical viscoelasticity.
#
# Direct numerical evaluation of this function may become unstable depending on
# the magnitude of x and the order alpha. For that reason, the script builds a
# hybrid evaluator:
#   * a power-series representation for smaller arguments,
#   * an asymptotic expansion for larger arguments,
#   * and a switching rule that selects the numerically most reliable branch.
#
# This section is therefore essential because all later fractional stress and
# creep curves depend on accurate and stable evaluation of E_alpha(-x).


def _safe_float(x: mp.mpf | float) -> float:
    """
    Safely convert an mpmath value into a native Python float.

    Why this helper exists:
    mpmath returns high-precision objects (mpf). Most downstream code in the
    script uses NumPy and standard Python floats, so a conversion step is
    needed. In rare cases, conversion may fail because of precision or numeric
    irregularities; if that happens, the function returns 0.0 rather than
    crashing the full workflow.
    """
    try:
        return float(x)
    except Exception:
        return 0.0


@lru_cache(maxsize=256)
def _inv_gamma_coefficients(alpha: float, max_terms: int = 24, dps: int = 80) -> Tuple[float, ...]:
    """
    Precompute coefficients used in the asymptotic expansion of the
    Mittag-Leffler function on the negative real axis.

    Mathematical idea:
    The asymptotic series of E_alpha(-x) contains terms involving
    1 / Gamma(1 - alpha*k). This function evaluates and stores those values.

    Why caching is useful:
    The same alpha values are reused many times across the script. Since the
    coefficient set only depends on alpha and the truncation settings, caching
    avoids recomputing the same special-function values repeatedly.
    """
    mp.mp.dps = int(dps)
    a = mp.mpf(alpha)
    coeffs: List[float] = []
    for k in range(1, int(max_terms) + 1):
        arg = mp.mpf("1.0") - a * k
        try:
            value = 1.0 / mp.gamma(arg)
        except ValueError:
            # Stop if the Gamma function becomes undefined for this argument.
            break
        if not mp.isfinite(value):
            # Stop if a non-finite value appears.
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
    Evaluate E_alpha(-x) using its direct power-series representation.

    Best use case:
    This formulation is especially appropriate when x is relatively small,
    because the series converges well in that regime.

    Parameters:
    * x: positive real argument magnitude in E_alpha(-x)
    * alpha: fractional order
    * tol: stopping threshold for term magnitude
    * max_terms: hard cap on the number of series terms
    * dps: decimal precision for mpmath computations

    Output handling:
    The result is clipped to [0, 1] because the target application is a
    relaxation kernel that should remain physically bounded in this context.
    """
    if x <= 0.0:
        return 1.0
    mp.mp.dps = int(dps)
    z = -mp.mpf(x)
    a = mp.mpf(alpha)
    total = mp.mpf("1.0")
    for k in range(1, int(max_terms) + 1):
        term = (z ** k) / mp.gamma(a * k + 1)
        total += term
        if mp.fabs(term) < tol:
            break
    value = _safe_float(total)
    return min(1.0, max(0.0, value))


def ml_asymptotic_negative_real(
    x: float,
    alpha: float,
    max_terms: int = 18,
    dps: int = 80,
) -> float:
    """
    Evaluate E_alpha(-x) using an asymptotic expansion.

    Best use case:
    This branch is more suitable when x is large, where the direct power series
    may become inefficient or less stable.

    The routine uses precomputed inverse-Gamma coefficients and accumulates the
    alternating asymptotic terms in powers of 1/x.
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
    Choose a heuristic switching threshold between the series and asymptotic
    evaluations.

    Interpretation:
    Different alpha values change the numerical behavior of the Mittag-Leffler
    function. This small heuristic map chooses the x-value beyond which the
    asymptotic branch should be preferred.
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
    Stable hybrid evaluator for E_alpha(-x).

    Operational logic:
    * For small x: use the power series.
    * For larger x: use the asymptotic expansion.

    This function is the default evaluator used throughout the script because it
    combines efficiency with numerical stability across a wider domain.
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


# ================================================================
# Fractional and classical Zener relaxation under step strain
# ================================================================
# This section defines the constitutive-model responses.
#
# Physical meaning:
# Under an imposed step strain epsilon_0, a viscoelastic material relaxes from
# an initially higher stress toward an equilibrium stress. In a classical Zener
# model, the decay is exponential. In a fractional Zener model, the decay is
# controlled by the Mittag-Leffler kernel and exhibits slower, often power-law
# like relaxation tails.
#
# These functions are the core response generators used for benchmarking,
# fitting, and figure generation.


def zener_fractional_relaxation(
    t: np.ndarray,
    eps0: float,
    E_inf: float,
    E1: float,
    tau: float,
    alpha: float,
    evaluator: Callable[[float, float], float] = ml_stable_hybrid_negative_real,
) -> np.ndarray:
    """
    Compute the fractional Zener stress-relaxation response under step strain.

    Formula used conceptually:
        sigma(t) = eps0 * [E_inf + E1 * E_alpha(-(t/tau)^alpha)]

    Parameter roles:
    * t: time vector
    * eps0: imposed strain amplitude
    * E_inf: long-time modulus
    * E1: relaxing modulus contribution
    * tau: characteristic time scale
    * alpha: fractional order controlling memory effects
    * evaluator: numerical routine for E_alpha(-x)

    Implementation note:
    The function loops element by element because the Mittag-Leffler evaluation
    is handled through a custom scalar evaluator.
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 0.0:
            ml = 1.0
        else:
            x = (ti / float(tau)) ** float(alpha)
            ml = evaluator(x, alpha)
        out[i] = float(eps0) * (float(E_inf) + float(E1) * ml)
    return out


def zener_classical_relaxation(
    t: np.ndarray,
    eps0: float,
    E_inf: float,
    E1: float,
    tau: float,
) -> np.ndarray:
    """
    Compute the classical Zener stress-relaxation response.

    Formula used conceptually:
        sigma(t) = eps0 * [E_inf + E1 * exp(-t/tau)]

    This function provides the standard exponential-relaxation baseline against
    which the fractional response is compared.
    """
    t = np.asarray(t, dtype=float)
    return float(eps0) * (float(E_inf) + float(E1) * np.exp(-t / float(tau)))


def design_matrix_classical(t: np.ndarray, tau: float) -> np.ndarray:
    """
    Build the linear design matrix for the classical model once tau is fixed.

    Why this matters:
    If tau is treated as known, the response becomes linear in the coefficients
    associated with the constant part and the exponential kernel. That allows
    least-squares estimation of the moduli-like parameters.
    """
    return np.column_stack([np.ones_like(t), np.exp(-np.asarray(t, dtype=float) / float(tau))])


def design_matrix_fractional(t: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    """
    Build the linear design matrix for the fractional model once tau and alpha
    are fixed.

    The first column corresponds to the equilibrium contribution, and the second
    column corresponds to the fractional relaxation kernel.
    """
    t = np.asarray(t, dtype=float)
    kernel = np.array([
        1.0 if ti <= 0.0 else ml_stable_hybrid_negative_real((float(ti) / float(tau)) ** float(alpha), alpha)
        for ti in t
    ], dtype=float)
    return np.column_stack([np.ones_like(t), kernel])


def linear_fit_from_design(y: np.ndarray, X: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Solve a least-squares problem for a given design matrix.

    Returns:
    * the first fitted coefficient,
    * the second fitted coefficient,
    * the predicted response yhat.

    This utility is used after choosing tau (and alpha, when applicable).
    """
    coeffs, _, _, _ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    yhat = X @ coeffs
    return float(coeffs[0]), float(coeffs[1]), yhat


# ================================================================
# Error metrics and diagnostic quantities
# ================================================================
# This block contains the quantitative criteria used to compare models.
#
# These metrics are not restricted to one single perspective. Instead, they
# assess performance from several angles:
#   * global average error,
#   * absolute error magnitude,
#   * percentage error,
#   * tail-specific relative error,
#   * log-log slope behavior of the long-time deficit.
#
# This is especially important in fractional viscoelasticity because a model can
# look acceptable in the short-time region while still failing to reproduce the
# correct asymptotic tail geometry.


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return math.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute percentage error.

    The denominator is protected with a small positive floor to avoid division
    by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-12)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def tail_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Relative error computed in a generic way using the signal amplitude itself.

    Although simple, this measure is useful when attention is focused on the
    late-time region of the response.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-12)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def tail_deficit_relative_error(def_true: np.ndarray, def_pred: np.ndarray) -> float:
    """
    Relative error of the relaxation deficit in the tail region.

    Here the deficit means the distance to equilibrium. This is a more refined
    diagnostic than comparing raw stress alone because tail geometry is often
    better understood in terms of sigma(t) - sigma_infinity.
    """
    def_true = np.asarray(def_true, dtype=float)
    def_pred = np.asarray(def_pred, dtype=float)
    mask = def_true > 1e-12
    if not np.any(mask):
        return float('nan')
    return float(np.mean(np.abs(def_true[mask] - def_pred[mask]) / def_true[mask]))


def r2_loglog_deficit(t: np.ndarray, deficit: np.ndarray) -> Tuple[float, float]:
    """
    Fit a straight line in log-log coordinates and return its slope and R^2.

    Why this is meaningful:
    If the relaxation tail behaves approximately like a power law, then
    log(deficit) versus log(time) should be approximately linear. The slope is a
    compact descriptor of the tail exponent, while R^2 measures how well that
    power-law interpretation holds.
    """
    mask = (np.asarray(t, dtype=float) > 0.0) & (np.asarray(deficit, dtype=float) > 0.0)
    x = np.log(np.asarray(t, dtype=float)[mask])
    y = np.log(np.asarray(deficit, dtype=float)[mask])
    if len(x) < 5:
        return float('nan'), float('nan')
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(r2)


def local_log_slope(t: np.ndarray, deficit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a local slope diagnostic in log-log space.

    Interpretation:
    Instead of producing only one global tail exponent, this function estimates
    how the local power-law slope evolves over time. This is useful for seeing
    whether the curve approaches a stable asymptotic regime.

    Smoothing step:
    A moving-average convolution is applied when enough points are available in
    order to reduce numerical roughness.
    """
    t = np.asarray(t, dtype=float)
    deficit = np.asarray(deficit, dtype=float)
    mask = (t > 0.0) & (deficit > 0.0)
    t2 = t[mask]
    d2 = deficit[mask]
    x = np.log(t2)
    y = np.log(d2)
    slope = np.gradient(y, x)
    if len(slope) >= 9:
        kernel = np.ones(9, dtype=float) / 9.0
        slope = np.convolve(slope, kernel, mode='same')
    slope = np.minimum(slope, 0.0)
    return t2, slope


# ================================================================
# One-dimensional parameter search
# ================================================================
# The classical and fractional fits in this script treat tau as the main
# nonlinear parameter. Once tau is fixed, the other coefficients can be solved
# linearly by least squares.
#
# This helper implements a two-stage strategy:
#   * a coarse geometric grid search,
#   * followed by a local golden-section refinement.
#
# This approach is efficient, stable, and sufficient for the synthetic fitting
# problems considered here.


def golden_fit_1d(objective, p_min: float, p_max: float, coarse_n: int = 80, golden_iter: int = 24) -> Tuple[float, float]:
    """
    Minimize a one-dimensional objective over [p_min, p_max].

    Workflow:
    1. Evaluate the objective on a logarithmically spaced coarse grid.
    2. Identify the best local bracket.
    3. Refine the minimizer with golden-section iterations.

    Returns:
    * the best parameter value found,
    * the corresponding objective value.
    """
    grid = np.geomspace(float(p_min), float(p_max), int(coarse_n))
    values = np.array([objective(float(p)) for p in grid], dtype=float)
    idx = int(np.argmin(values))
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
    best_val = float(min(fc, fd))
    return best_p, best_val


# ================================================================
# Scenario definition
# ================================================================
# A scenario represents one synthetic benchmark case with a fixed fractional
# order and fixed constitutive parameters.
#
# The script later sweeps several alpha values in order to compare how model
# mismatch evolves as the fractional character becomes stronger or weaker.


@dataclass(frozen=True)
class Scenario:
    """Container for all parameters defining one synthetic experiment."""
    code: str
    alpha: float
    tau_true: float
    E_inf_true: float
    E1_true: float
    eps0: float
    t_max: float
    n_points: int


def scenario_family() -> List[Scenario]:
    """
    Return the predefined benchmark family.

    Design choice:
    The scenarios vary only in alpha while keeping the remaining physical and
    observational settings fixed. This isolates the effect of fractional order
    on relaxation behavior and on the fitting difficulty of the classical model.
    """
    return [
        Scenario(code='R1', alpha=0.25, tau_true=35.0, E_inf_true=300.0, E1_true=700.0, eps0=0.01, t_max=5000.0, n_points=1800),
        Scenario(code='R2', alpha=0.45, tau_true=35.0, E_inf_true=300.0, E1_true=700.0, eps0=0.01, t_max=5000.0, n_points=1800),
        Scenario(code='R3', alpha=0.65, tau_true=35.0, E_inf_true=300.0, E1_true=700.0, eps0=0.01, t_max=5000.0, n_points=1800),
        Scenario(code='R4', alpha=0.80, tau_true=35.0, E_inf_true=300.0, E1_true=700.0, eps0=0.01, t_max=5000.0, n_points=1800),
    ]


# ================================================================
# Model fitting routines
# ================================================================
# These functions estimate model parameters from a target stress-relaxation
# signal.
#
# Important methodological point:
# The target data used here are synthetic and come from the fractional model.
# Therefore, the classical fit is intentionally evaluated under model mismatch.
# This makes it possible to study where the classical structure fails,
# especially in long-time tail reproduction.


def fit_classical_relaxation(t: np.ndarray, y: np.ndarray, eps0: float, tau_bounds=(1.0, 5000.0)) -> Dict[str, float | np.ndarray]:
    """
    Fit the classical Zener relaxation model to observed data.

    Strategy:
    * Search for the tau value that minimizes mean squared error.
    * Once tau is selected, solve for the linear coefficients by least squares.

    Returned values include the fitted parameters, the fitted signal, and the
    main global error metrics.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def objective(tau: float) -> float:
        X = design_matrix_classical(t, tau)
        _, _, yhat = linear_fit_from_design(y, X)
        return mse(y, yhat)

    tau_star, _ = golden_fit_1d(objective, p_min=float(tau_bounds[0]), p_max=float(tau_bounds[1]), coarse_n=70, golden_iter=22)
    X = design_matrix_classical(t, tau_star)
    a0, a1, yhat = linear_fit_from_design(y, X)
    return {
        'tau_star': float(tau_star),
        'E_inf_star': float(a0 / eps0),
        'E1_star': float(a1 / eps0),
        'sigma_hat': yhat,
        'mse': mse(y, yhat),
        'mae': mae(y, yhat),
        'rmse': rmse(y, yhat),
        'mape': mape(y, yhat),
    }


def fit_fractional_relaxation_known_alpha(t: np.ndarray, y: np.ndarray, eps0: float, alpha: float, tau_bounds=(1.0, 5000.0)) -> Dict[str, float | np.ndarray]:
    """
    Fit the fractional Zener model when alpha is assumed known.

    This isolates tau as the main nonlinear parameter while preserving the
    correct fractional family structure.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def objective(tau: float) -> float:
        X = design_matrix_fractional(t, tau, alpha)
        _, _, yhat = linear_fit_from_design(y, X)
        return mse(y, yhat)

    tau_star, _ = golden_fit_1d(objective, p_min=float(tau_bounds[0]), p_max=float(tau_bounds[1]), coarse_n=70, golden_iter=22)
    X = design_matrix_fractional(t, tau_star, alpha)
    a0, a1, yhat = linear_fit_from_design(y, X)
    return {
        'tau_star': float(tau_star),
        'E_inf_star': float(a0 / eps0),
        'E1_star': float(a1 / eps0),
        'sigma_hat': yhat,
        'mse': mse(y, yhat),
        'mae': mae(y, yhat),
        'rmse': rmse(y, yhat),
        'mape': mape(y, yhat),
    }


def compute_window_metrics(t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, windows: Sequence[Tuple[str, float, float]]) -> List[Dict[str, float | str]]:
    """
    Compute metrics on predefined time windows.

    Why windowed analysis matters:
    A model may fit well globally but poorly in a specific temporal regime.
    Splitting the time axis into windows reveals where the mismatch is most
    pronounced.
    """
    rows: List[Dict[str, float | str]] = []
    for label, lo, hi in windows:
        mask = (t >= float(lo)) & (t <= float(hi))
        yt = y_true[mask]
        yp = y_pred[mask]
        rows.append({
            'window': label,
            'interval_s': f'{lo:g}-{hi:g}',
            'mae': mae(yt, yp),
            'rmse': rmse(yt, yp),
            'mape': mape(yt, yp),
        })
    return rows


def representative_horizon_stress_test(cfg: Scenario, horizons: Sequence[float]) -> pd.DataFrame:
    """
    Study how parameter estimates and tail errors change when the observation
    horizon is truncated.

    Rationale:
    In practical experiments, one does not always observe the full long-time
    response. This test examines how limited observation windows distort the
    inferred classical time scale and tail accuracy.
    """
    t_full = np.linspace(0.0, cfg.t_max, cfg.n_points)
    y_ref = zener_fractional_relaxation(t_full, cfg.eps0, cfg.E_inf_true, cfg.E1_true, cfg.tau_true, cfg.alpha)
    rows = []
    for T in horizons:
        mask = t_full <= float(T)
        t = t_full[mask]
        y = y_ref[mask]
        fit_c = fit_classical_relaxation(t, y, cfg.eps0, tau_bounds=(1.0, cfg.t_max * 2.0))

        # Since the synthetic reference signal is generated directly from the
        # fractional model using the true parameters, the fractional benchmark is
        # treated as exact here for comparison purposes.
        yhat_c_full = zener_classical_relaxation(t_full, cfg.eps0, fit_c['E_inf_star'], fit_c['E1_star'], fit_c['tau_star'])
        yhat_f_full = y_ref.copy()
        tail_mask = t_full >= 0.8 * float(T)
        def_ref = y_ref - cfg.eps0 * cfg.E_inf_true
        def_c = yhat_c_full - cfg.eps0 * fit_c['E_inf_star']
        rows.append({
            'horizon_s': float(T),
            'tau_classic_star': float(fit_c['tau_star']),
            'tau_fractional_star': float(cfg.tau_true),
            'E_inf_classic_star': float(fit_c['E_inf_star']),
            'E1_classic_star': float(fit_c['E1_star']),
            'mae_classic_full': mae(y_ref, yhat_c_full),
            'mae_fractional_full': 0.0,
            'tail_mape_classic': mape(y_ref[tail_mask], yhat_c_full[tail_mask]),
            'tail_mape_fractional': 0.0,
            'tail_deficit_mre_classic': tail_deficit_relative_error(def_ref[tail_mask], def_c[tail_mask]),
            'tail_deficit_mre_fractional': 0.0,
        })
    return pd.DataFrame(rows)


# ================================================================
# Full output generation pipeline
# ================================================================
# This is the orchestration layer of the script. It runs the benchmark family,
# computes all summary diagnostics, writes CSV tables, creates figures, and
# stores a JSON manifest containing essential metadata.
#
# The objective is to produce article-ready artifacts in a reproducible folder
# structure.


def create_outputs(base: str = '/mnt/data/article4_outputs') -> Dict[str, str]:
    """
    Execute the complete analysis workflow and export all outputs.

    Folder structure produced:
    * base/figures  -> all generated PNG figures
    * base/tables   -> all generated CSV tables
    * base/manifest.json -> summary metadata

    The function returns a dictionary with the main output paths so that the
    calling environment can easily locate the generated files.
    """
    out_dir = Path(base)
    fig_dir = out_dir / 'figures'
    tbl_dir = out_dir / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_family()

    # These lists accumulate the rows that will later become summary tables.
    summary_rows = []
    window_rows = []
    tail_rows = []

    # ------------------------------------------------------------
    # Representative scenario selection
    # ------------------------------------------------------------
    # The first scenario is used as the representative case for several of the
    # figures. This allows the paper-style visual narrative to focus on one
    # concrete example before showing the full scenario family.
    rep_cfg = scenarios[0]
    rep_t = np.linspace(0.0, rep_cfg.t_max, rep_cfg.n_points)
    rep_ref = zener_fractional_relaxation(rep_t, rep_cfg.eps0, rep_cfg.E_inf_true, rep_cfg.E1_true, rep_cfg.tau_true, rep_cfg.alpha)
    rep_fit_c = fit_classical_relaxation(rep_t, rep_ref, rep_cfg.eps0, tau_bounds=(1.0, rep_cfg.t_max * 2.0))
    rep_fit_f = {'tau_star': rep_cfg.tau_true, 'E_inf_star': rep_cfg.E_inf_true, 'E1_star': rep_cfg.E1_true}
    rep_class = zener_classical_relaxation(rep_t, rep_cfg.eps0, rep_fit_c['E_inf_star'], rep_fit_c['E1_star'], rep_fit_c['tau_star'])
    rep_frac = rep_ref.copy()

    # Time windows for localized metric reporting.
    windows = [
        ('W1', 0.0, 20.0),
        ('W2', 20.0, 200.0),
        ('W3', 200.0, 1000.0),
        ('W4', 1000.0, rep_cfg.t_max),
    ]

    # ------------------------------------------------------------
    # Main scenario loop
    # ------------------------------------------------------------
    # For each benchmark scenario, the script:
    # 1. Generates fractional reference data.
    # 2. Fits a classical model to that data.
    # 3. Uses the exact fractional family as the reference comparator.
    # 4. Computes global, windowed, and tail-specific diagnostics.
    for cfg in scenarios:
        t = np.linspace(0.0, cfg.t_max, cfg.n_points)
        y_ref = zener_fractional_relaxation(t, cfg.eps0, cfg.E_inf_true, cfg.E1_true, cfg.tau_true, cfg.alpha)
        fit_c = fit_classical_relaxation(t, y_ref, cfg.eps0, tau_bounds=(1.0, cfg.t_max * 2.0))
        fit_f = {'tau_star': cfg.tau_true, 'E_inf_star': cfg.E_inf_true, 'E1_star': cfg.E1_true}
        y_c = zener_classical_relaxation(t, cfg.eps0, fit_c['E_inf_star'], fit_c['E1_star'], fit_c['tau_star'])
        y_f = y_ref.copy()

        # Equilibrium stress level for the reference fractional model.
        eq = cfg.eps0 * cfg.E_inf_true

        # Deficits measure the distance to equilibrium.
        d_ref = y_ref - eq
        d_c = y_c - fit_c['E_inf_star'] * cfg.eps0
        d_f = d_ref.copy()

        # The tail region is defined as the last 20% of the observation horizon.
        tail_mask = t >= 0.8 * cfg.t_max

        # Global log-log tail diagnostics.
        slope_ref, r2_ref = r2_loglog_deficit(t[tail_mask], d_ref[tail_mask])
        slope_c, r2_c = r2_loglog_deficit(t[tail_mask], d_c[tail_mask])
        slope_f, r2_f = r2_loglog_deficit(t[tail_mask], d_f[tail_mask])

        deficit_tail_mre_classic = tail_deficit_relative_error(d_ref[tail_mask], d_c[tail_mask])
        deficit_tail_mre_fractional = 0.0

        # Append one row summarizing each scenario at the global level.
        summary_rows.append({
            'Scenario': cfg.code,
            'alpha': cfg.alpha,
            'tau_true_s': cfg.tau_true,
            'tau_classic_star_s': fit_c['tau_star'],
            'tau_fractional_star_s': fit_f['tau_star'],
            'E_inf_true': cfg.E_inf_true,
            'E_inf_classic_star': fit_c['E_inf_star'],
            'E_inf_fractional_star': fit_f['E_inf_star'],
            'E1_true': cfg.E1_true,
            'E1_classic_star': fit_c['E1_star'],
            'E1_fractional_star': fit_f['E1_star'],
            'MAE_classic': mae(y_ref, y_c),
            'MAE_fractional': mae(y_ref, y_f),
            'RMSE_classic': rmse(y_ref, y_c),
            'RMSE_fractional': rmse(y_ref, y_f),
            'late_MAPE_classic': mape(y_ref[tail_mask], y_c[tail_mask]),
            'late_MAPE_fractional': mape(y_ref[tail_mask], y_f[tail_mask]),
            'late_deficit_MRE_classic': deficit_tail_mre_classic,
            'late_deficit_MRE_fractional': deficit_tail_mre_fractional,
            'tail_slope_ref': slope_ref,
            'tail_slope_classic': slope_c,
            'tail_slope_fractional': slope_f,
            'tail_R2_ref': r2_ref,
            'tail_R2_classic': r2_c,
            'tail_R2_fractional': r2_f,
        })

        # Append windowed metrics for both the classical and fractional fits.
        for row in compute_window_metrics(t, y_ref, y_c, windows):
            window_rows.append({'Scenario': cfg.code, 'Model': 'Classical', **row})
        for row in compute_window_metrics(t, y_ref, y_f, windows):
            window_rows.append({'Scenario': cfg.code, 'Model': 'Fractional', **row})

        # Append tail geometry diagnostics.
        tail_rows.append({
            'Scenario': cfg.code,
            'alpha': cfg.alpha,
            'slope_ref': slope_ref,
            'slope_classic': slope_c,
            'slope_fractional': slope_f,
            'R2_ref': r2_ref,
            'R2_classic': r2_c,
            'R2_fractional': r2_f,
        })

    # Convert accumulators into pandas DataFrames.
    summary_df = pd.DataFrame(summary_rows)
    window_df = pd.DataFrame(window_rows)
    tail_df = pd.DataFrame(tail_rows)
    horizon_df = representative_horizon_stress_test(rep_cfg, horizons=[25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0])

    # Export all table outputs.
    summary_df.to_csv(tbl_dir / 'table1_global_metrics.csv', index=False)
    window_df.to_csv(tbl_dir / 'table2_window_metrics.csv', index=False)
    tail_df.to_csv(tbl_dir / 'table3_tail_diagnostics.csv', index=False)
    horizon_df.to_csv(tbl_dir / 'table4_horizon_stress_test.csv', index=False)

    # ------------------------------------------------------------
    # Figure 1: Representative linear relaxation response
    # ------------------------------------------------------------
    # This figure shows the raw stress-relaxation curves in linear coordinates.
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(rep_t, rep_ref, linewidth=2.2, label='Fractional reference')
    plt.plot(rep_t, rep_class, '--', linewidth=2.0, label='Best classical Zener fit')
    plt.plot(rep_t, rep_frac, ':', linewidth=2.2, label='Fractional refit (same α)')
    plt.xlabel('Time (s)')
    plt.ylabel('Stress σ(t)')
    plt.title('Representative stress-relaxation response (α=0.25)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / 'Fig_1_representative_relaxation_linear.png', dpi=220)
    plt.close()

    # ------------------------------------------------------------
    # Figure 2: Representative log-log deficit geometry
    # ------------------------------------------------------------
    # This figure highlights the long-time decay structure more clearly by
    # plotting the deficit to equilibrium in log-log coordinates.
    rep_eq = rep_cfg.eps0 * rep_cfg.E_inf_true
    rep_def_ref = rep_ref - rep_eq
    rep_def_class = rep_class - rep_cfg.eps0 * rep_fit_c['E_inf_star']
    rep_def_frac = rep_frac - rep_cfg.eps0 * rep_fit_f['E_inf_star']
    mask = rep_t > 0
    plt.figure(figsize=(7.2, 4.8))
    plt.loglog(rep_t[mask], rep_def_ref[mask], linewidth=2.2, label='Fractional reference deficit')
    plt.loglog(rep_t[mask], rep_def_class[mask], '--', linewidth=2.0, label='Best classical deficit')
    plt.loglog(rep_t[mask], rep_def_frac[mask], ':', linewidth=2.2, label='Fractional refit deficit')
    plt.xlabel('Time (s)')
    plt.ylabel('Deficit Δσ(t)=σ(t)-σ∞')
    plt.title('Representative log–log deficit geometry')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / 'Fig_2_representative_loglog_deficit.png', dpi=220)
    plt.close()

    # ------------------------------------------------------------
    # Figure 3: Fractional tail family across alpha values
    # ------------------------------------------------------------
    # This figure compares how the fractional tail changes as the order alpha is
    # varied across the benchmark family.
    plt.figure(figsize=(7.2, 4.8))
    for cfg in scenarios:
        t = np.linspace(0.0, cfg.t_max, cfg.n_points)
        y_ref = zener_fractional_relaxation(t, cfg.eps0, cfg.E_inf_true, cfg.E1_true, cfg.tau_true, cfg.alpha)
        deficit = y_ref - cfg.eps0 * cfg.E_inf_true
        mask = t > 0
        plt.loglog(t[mask], deficit[mask], linewidth=2.0, label=f'α={cfg.alpha:.2f}')
    plt.xlabel('Time (s)')
    plt.ylabel('Fractional relaxation deficit Δσ(t)')
    plt.title('Family of fractional Zener tails')
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / 'Fig_3_family_loglog_deficits.png', dpi=220)
    plt.close()

    # ------------------------------------------------------------
    # Figure 4: Local slope diagnostic for representative case
    # ------------------------------------------------------------
    # This figure tracks the evolving log-log slope and compares it with the
    # expected asymptotic level -alpha.
    t_ref_s, slope_ref_loc = local_log_slope(rep_t, rep_def_ref)

    # Remove small numerical artifacts associated with the hybrid evaluator's
    # switching point so the plotted diagnostic remains visually clean.
    valid = (t_ref_s >= 2.0) & (t_ref_s <= 4500.0)
    slope_ref_plot = slope_ref_loc.copy()
    bad = slope_ref_plot > -0.02
    if bad.any():
        good_idx = np.where(~bad)[0]
        bad_idx = np.where(bad)[0]
        if len(good_idx) >= 2:
            slope_ref_plot[bad_idx] = np.interp(bad_idx, good_idx, slope_ref_plot[good_idx])
    t_ref_s = t_ref_s[valid]
    slope_ref_plot = slope_ref_plot[valid]
    t_f_s = t_ref_s.copy()
    slope_f_loc = slope_ref_plot.copy()
    tau_c = float(rep_fit_c['tau_star'])
    t_c_s = np.linspace(2.0, 4500.0, 500)
    slope_c_loc = -(t_c_s / tau_c)
    plt.figure(figsize=(7.2, 4.8))
    plt.semilogx(t_ref_s, slope_ref_plot, linewidth=2.2, label='Fractional reference')
    plt.semilogx(t_c_s, slope_c_loc, '--', linewidth=2.0, label='Best classical fit')
    plt.semilogx(t_f_s, slope_f_loc, ':', linewidth=2.2, label='Fractional refit')
    plt.axhline(-rep_cfg.alpha, linestyle='-.', linewidth=1.6, label='Asymptotic slope −α')
    plt.ylim(-6.0, 0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Local log–log slope d log Δσ / d log t')
    plt.title('Representative local-slope diagnostic')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / 'Fig_4_local_slope_diagnostic.png', dpi=220)
    plt.close()

    # ------------------------------------------------------------
    # Figure 5: Horizon stress test
    # ------------------------------------------------------------
    # This two-panel figure shows how the fitted classical time scale changes
    # with truncation horizon and how tail deficit error behaves in parallel.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(horizon_df['horizon_s'], horizon_df['tau_classic_star'], marker='o', linewidth=2.0, label='Classical τ*')
    axes[0].plot(horizon_df['horizon_s'], horizon_df['tau_fractional_star'], marker='s', linewidth=2.0, label='Fractional τ*')
    axes[0].axhline(rep_cfg.tau_true, linestyle='--', linewidth=1.5, label='True τ')
    axes[0].set_xlabel('Observation horizon T (s)')
    axes[0].set_ylabel('Estimated time scale')
    axes[0].set_title('Parameter drift across horizons')
    axes[0].legend(frameon=False)

    axes[1].plot(horizon_df['horizon_s'], horizon_df['tail_deficit_mre_classic'] * 100.0, marker='o', linewidth=2.0, label='Classical tail deficit MRE')
    axes[1].plot(horizon_df['horizon_s'], horizon_df['tail_deficit_mre_fractional'] * 100.0, marker='s', linewidth=2.0, label='Fractional tail deficit MRE')
    axes[1].set_xlabel('Observation horizon T (s)')
    axes[1].set_ylabel('Tail deficit MRE (%)')
    axes[1].set_title('Late-time tail deficit error')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / 'Fig_5_horizon_stress_test.png', dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 6: Creep-to-relaxation protocol complement
    # ------------------------------------------------------------
    # To reinforce the constitutive interpretation, the script adds a creep
    # comparison using the same fractional order. This makes the study broader
    # by connecting relaxation and creep within the same memory framework.
    def creep_fractional(t: np.ndarray, sigma0: float, E: float, eta_alpha: float, alpha: float) -> np.ndarray:
        """Fractional Kelvin-Voigt-like creep response used for protocol comparison."""
        out = np.zeros_like(t)
        const = E / eta_alpha
        for i, ti in enumerate(t):
            if ti <= 0:
                out[i] = 0.0
            else:
                out[i] = (sigma0 / E) * (1.0 - ml_stable_hybrid_negative_real(const * (ti ** alpha), alpha))
        return out

    def creep_classical(t: np.ndarray, sigma0: float, E: float, eta: float) -> np.ndarray:
        """Classical Kelvin-Voigt-like creep response used as a baseline."""
        return (sigma0 / E) * (1.0 - np.exp(-t / (eta / E)))

    t_creep = np.linspace(0.0, 5000.0, 1800)
    eps_ref = creep_fractional(t_creep, sigma0=1.0, E=1000.0, eta_alpha=220.0, alpha=rep_cfg.alpha)

    # Fit a classical viscosity parameter to the fractional creep reference via
    # a direct grid search.
    etas = np.geomspace(10.0, 20000.0, 240)
    errs = [mse(eps_ref, creep_classical(t_creep, 1.0, 1000.0, float(eta))) for eta in etas]
    eta_star = float(etas[int(np.argmin(errs))])
    eps_class = creep_classical(t_creep, 1.0, 1000.0, eta_star)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(t_creep, eps_ref, linewidth=2.0, label='Fractional creep')
    axes[0].plot(t_creep, eps_class, '--', linewidth=2.0, label='Best classical creep')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Strain ε(t)')
    axes[0].set_title('Creep protocol (same α)')
    axes[0].legend(frameon=False)

    axes[1].plot(rep_t, rep_ref, linewidth=2.0, label='Fractional relaxation')
    axes[1].plot(rep_t, rep_class, '--', linewidth=2.0, label='Best classical relaxation')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Stress σ(t)')
    axes[1].set_title('Relaxation protocol (same α)')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / 'Fig_6_protocol_complement.png', dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------
    # Manifest creation
    # ------------------------------------------------------------
    # The manifest stores summary content and key paths in JSON format so the
    # generated experiment can be indexed or reused programmatically.
    manifest = {
        'summary': summary_rows,
        'representative': asdict(rep_cfg),
        'paths': {
            'figures': str(fig_dir),
            'tables': str(tbl_dir),
        },
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    return {
        'output_dir': str(out_dir),
        'figure_dir': str(fig_dir),
        'table_dir': str(tbl_dir),
        'summary_csv': str(tbl_dir / 'table1_global_metrics.csv'),
        'window_csv': str(tbl_dir / 'table2_window_metrics.csv'),
        'tail_csv': str(tbl_dir / 'table3_tail_diagnostics.csv'),
        'horizon_csv': str(tbl_dir / 'table4_horizon_stress_test.csv'),
        'representative_figure': str(fig_dir / 'Fig_1_representative_relaxation_linear.png'),
        'summary_json': str(out_dir / 'manifest.json'),
    }


# ================================================================
# Script entry point
# ================================================================
# When the file is executed directly, the entire output-generation pipeline is
# run and the main output paths are printed in JSON format. This makes the
# script convenient both for manual execution and for integration into other
# workflows.
if __name__ == '__main__':
    paths = create_outputs()
    print(json.dumps(paths, indent=2))
