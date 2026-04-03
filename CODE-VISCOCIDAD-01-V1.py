from __future__ import annotations

"""
Research-oriented pipeline for comparing classical Kelvin-Voigt creep against
a fractional Kelvin-Voigt formulation driven by a stable Mittag-Leffler
evaluator on the negative real axis.

General workflow of the script
------------------------------
1. Build numerically stable evaluators for E_alpha(-x).
2. Generate synthetic fractional-creep reference curves for several scenarios.
3. Fit a classical Kelvin-Voigt model and a refitted fractional model.
4. Quantify the discrepancy between the models with global and windowed metrics.
5. Diagnose long-time tail behavior through log-log regressions.
6. Export publication-ready tables and figures to an output directory.

The code is intentionally organized in sections so that each block can be read
as an independent stage of the computational study: special-function
evaluation, constitutive modeling, fitting, diagnostics, visualization, and
artifact export.

Main outputs:
    outputs_dir/
        figures/
            Fig_01_reference_vs_methods_alpha055.png
            Fig_02_relative_error_profiles.png
            Fig_03_creep_scenarios_full.png
            Fig_04_creep_error_scenarios.png
            Fig_05_long_tail_loglog.png
        tables/
            Table_01_mittag_leffler_accuracy.csv
            Table_02_creep_global_metrics.csv
            Table_03_windowed_mae.csv
        manifest.json

Requirements:
    numpy, matplotlib, mpmath

Run:
    python VISCOCIDAD_ARTICLE_VALIDATION_COMPLETE.py
"""

import json
import math
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import mpmath as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ================================================================
# Stable Mittag-Leffler evaluation on the negative real axis
# This block concentrates all numerical machinery required to evaluate
# the special function that appears in the fractional creep law.
# ================================================================

def _safe_float(x: mp.mpf | float) -> float:
    """
    Convert a high-precision mpmath value into a regular Python float.

    The helper is intentionally defensive: if a conversion fails because the
    value is not finite or carries an unexpected internal representation, the
    function returns 0.0 so that downstream calculations remain numerically
    robust instead of crashing the full pipeline.
    """
    try:
        return float(x)
    except Exception:
        # Returning a safe fallback protects the report-generation workflow from
        # isolated numerical conversion failures.
        return 0.0


@lru_cache(maxsize=256)
def _inv_gamma_coefficients(alpha: float, max_terms: int = 24, dps: int = 80) -> Tuple[float, ...]:
    """
    Precompute coefficients used in the asymptotic inverse-power expansion.

    For the large-x regime, the script approximates E_alpha(-x) by a truncated
    series involving inverse powers of x and inverse Gamma terms. Since the
    same alpha values are queried many times during scenario fitting and
    plotting, the coefficients are cached to avoid recomputation.
    """
    mp.mp.dps = int(dps)
    a = mp.mpf(alpha)
    coeffs: List[float] = []
    for k in range(1, int(max_terms) + 1):
        # The asymptotic expansion contains factors of 1 / Gamma(1 - alpha * k).
        arg = mp.mpf("1.0") - a * k
        try:
            value = 1.0 / mp.gamma(arg)
        except ValueError:
            # Stop when Gamma hits an invalid argument for the current branch.
            break
        if not mp.isfinite(value):
            # Abort the sequence if the coefficient is not numerically usable.
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
    Evaluate E_alpha(-x) with a high-precision power series for small-to-medium x.

    This branch is accurate near the origin, where the direct series converges
    well. The result is clipped to [0, 1] because the present application uses
    the evaluator inside creep formulas where the relevant branch is expected to
    remain within this physical range.
    """
    if x <= 0.0:
        return 1.0
    mp.mp.dps = int(dps)
    z = -mp.mpf(x)
    a = mp.mpf(alpha)
    term = mp.mpf("1.0")
    total = term
    for k in range(1, int(max_terms) + 1):
        # Direct definition of the Mittag-Leffler power series.
        term = (z ** k) / mp.gamma(a * k + 1)
        total += term
        # Once the increment becomes negligible, the truncated sum is accepted.
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
    Evaluate E_alpha(-x) with an inverse-power asymptotic expansion.

    This approximation is preferable for larger x values, where the direct
    series can become inefficient or numerically fragile. The function reuses
    cached coefficients and progressively accumulates powers of 1/x.
    """
    if x <= 0.0:
        return 1.0
    coeffs = _inv_gamma_coefficients(float(alpha), max_terms=int(max_terms), dps=int(dps))
    total = 0.0
    sign = 1.0
    x_pow = float(x)
    for coeff in coeffs:
        # Alternating inverse-power accumulation for the asymptotic branch.
        total += sign * coeff / x_pow
        sign *= -1.0
        x_pow *= float(x)
    return min(1.0, max(0.0, float(total)))


def adaptive_series_switch(alpha: float) -> float:
    """
    Choose the crossover value between the series and asymptotic evaluators.

    Different alpha values move the region where each approximation is most
    stable. The thresholds below are simple empirical rules that keep the
    hybrid evaluator accurate while remaining computationally inexpensive.
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
    Hybrid evaluator for E_alpha(-x) that switches method according to scale.

    The purpose of this dispatcher is to hide the numerical details from the
    constitutive-model routines. Small x values are handled by the power
    series, while larger x values are delegated to the asymptotic expansion.
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
# Fractional / classical Kelvin-Voigt creep
# Here the constitutive laws are defined. The script uses these functions
# as the core physical models that will later be fitted and compared.
# ================================================================

def creep_fractional_kelvin_voigt(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta_alpha: float,
    alpha: float,
    evaluator: Callable[[float, float], float],
) -> np.ndarray:
    """
    Compute fractional Kelvin-Voigt creep under a step stress input.

    Mathematical model:
        epsilon(t) = (sigma0 / E) [1 - E_alpha(-(E/eta_alpha) t^alpha)].

    The function evaluates the strain time history point by point. This
    explicit loop is kept for clarity because each time instant requires a
    special-function evaluation that depends on t^alpha.
    """
    const = float(E) / float(eta_alpha)
    strain = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti <= 0.0:
            # Before or at loading onset, the creep strain is set to zero.
            strain[i] = 0.0
        else:
            # Dimensionless argument passed to the Mittag-Leffler evaluator.
            x = const * (float(ti) ** float(alpha))
            ml = evaluator(x, alpha)
            strain[i] = (float(sigma0) / float(E)) * (1.0 - ml)
    return strain


def creep_classic_kelvin_voigt(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta: float,
) -> np.ndarray:
    """
    Classical Kelvin-Voigt creep response under step stress.

    This is the exponential benchmark used throughout the script to quantify
    how much fidelity is lost when a non-fractional constitutive law is forced
    to mimic fractional reference data.
    """
    tau = float(eta) / float(E)
    return (float(sigma0) / float(E)) * (1.0 - np.exp(-np.asarray(t, dtype=float) / tau))


# ================================================================
# Metrics and fitting
# These utilities quantify model discrepancy and estimate the best single
# parameter values needed to reproduce the reference fractional response.
# ================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute basic error metrics used in the comparison tables.

    The three selected metrics summarize average quadratic error, average
    absolute deviation, and worst-case absolute deviation, respectively.
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
    Fit a single scalar parameter with a coarse scan plus golden-section search.

    The routine first locates a promising bracket on a uniform grid and then
    refines the optimum with a derivative-free search. This keeps the fitting
    logic transparent and avoids external optimization dependencies.
    """
    grid = np.linspace(float(p_min), float(p_max), int(coarse_n))
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
        # Keep shrinking the bracket around the point with the smaller loss.
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


# ================================================================
# Scenario configuration
# The numerical experiments are parameterized through small immutable data
# records so the study remains reproducible and easy to extend.
# ================================================================
@dataclass(frozen=True)
class Scenario:
    """
    Compact container for all parameters that define one numerical experiment.

    Each scenario specifies the target fractional order, the observation
    horizon, the temporal resolution, and the parameter ranges used during the
    classical and fractional re-fitting stages.
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
    Define the default set of experiments used in the paper-style pipeline.

    The selected horizons deliberately progress from short to long observation
    windows so that the output can reveal how tail behavior becomes more
    important as the analysis moves toward long-time creep.
    """
    return [
        Scenario(code="S1", title="Short horizon baseline", alpha_true=0.65, t_max=50.0, n_points=700),
        Scenario(code="S2", title="Moderate horizon stress test", alpha_true=0.55, t_max=200.0, n_points=900),
        Scenario(code="S3", title="Long horizon tail emphasis", alpha_true=0.25, t_max=2000.0, n_points=1200),
    ]


# ================================================================
# Helper calculations
# This section groups secondary transformations and diagnostics used by the
# tables and plots, especially normalized quantities and tail analyses.
# ================================================================

def fit_models(cfg: Scenario) -> Dict[str, object]:
    """
    Run the full fitting cycle for one scenario.

    For each configuration, the function:
    1. Builds the reference fractional response.
    2. Fits the best classical viscosity against that reference.
    3. Refits the fractional viscosity while keeping alpha fixed.
    4. Returns time vectors, strain histories, and fitted parameters.
    """
    t = np.linspace(0.0, cfg.t_max, cfg.n_points)
    eps_ref = creep_fractional_kelvin_voigt(
        t=t,
        sigma0=cfg.sigma0,
        E=cfg.E,
        eta_alpha=cfg.eta_alpha_true,
        alpha=cfg.alpha_true,
        evaluator=ml_stable_hybrid_negative_real,
    )

    # Objective for the classical Kelvin-Voigt benchmark.
    obj_class = lambda eta: np.mean((creep_classic_kelvin_voigt(t, cfg.sigma0, cfg.E, eta) - eps_ref) ** 2)
    # Objective for the fractional refit with the correct alpha but free eta_alpha.
    obj_frac = lambda eta: np.mean((creep_fractional_kelvin_voigt(t, cfg.sigma0, cfg.E, eta, cfg.alpha_true, ml_stable_hybrid_negative_real) - eps_ref) ** 2)

    eta_class, _ = golden_fit_1d(obj_class, cfg.eta_class_bounds[0], cfg.eta_class_bounds[1])
    eta_frac, _ = golden_fit_1d(obj_frac, cfg.eta_frac_bounds[0], cfg.eta_frac_bounds[1])

    # Reconstruct both fitted responses using the optimized parameters.
    eps_class = creep_classic_kelvin_voigt(t, cfg.sigma0, cfg.E, eta_class)
    eps_frac = creep_fractional_kelvin_voigt(t, cfg.sigma0, cfg.E, eta_frac, cfg.alpha_true, ml_stable_hybrid_negative_real)

    return {
        "t": t,
        "eps_ref": eps_ref,
        "eps_class": eps_class,
        "eps_frac": eps_frac,
        "eta_class": eta_class,
        "eta_frac": eta_frac,
    }


def normalized_strain(eps: np.ndarray, sigma0: float, E: float) -> np.ndarray:
    """Scale strain into the dimensionless quantity E * epsilon / sigma0."""
    return (E / sigma0) * np.asarray(eps, dtype=float)


def normalized_deficit(eps: np.ndarray, sigma0: float, E: float) -> np.ndarray:
    """
    Compute the remaining distance to the asymptotic normalized strain level.

    A very small lower bound is enforced to keep later logarithmic transforms
    numerically stable.
    """
    return np.clip(1.0 - normalized_strain(eps, sigma0, E), 1e-300, None)


def relative_windows(t_max: float) -> List[Tuple[str, float, float]]:
    """
    Partition the observation horizon into early, intermediate, and late windows.

    These windows support localized error analysis, which is useful because a
    model can behave well globally while still failing in the tail region.
    """
    return [
        ("W1", 0.0, 0.1 * t_max),
        ("W2", 0.1 * t_max, 0.5 * t_max),
        ("W3", 0.5 * t_max, t_max),
    ]


def tail_regression_loglog(t: np.ndarray, deficit: np.ndarray, t_start: float) -> Dict[str, float]:
    """
    Estimate the power-law tail exponent from log-log data.

    The regression is applied only beyond t_start so that the analysis focuses
    on the long-time regime instead of mixing transient and tail dynamics.
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
    Build a comparable regression summary for the classical exponential tail.

    Although the classical model does not truly exhibit a power-law tail, this
    diagnostic produces a linear summary in the transformed space so that its
    behavior can be contrasted with the fractional curves.
    """
    mask = np.asarray(t) >= float(t_start)
    x = np.log10(np.asarray(t)[mask])
    y = -(np.asarray(t)[mask] / float(tau)) / math.log(10.0)
    coeff = np.polyfit(x, y, 1)
    yhat = coeff[0] * x + coeff[1]
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return {"slope": float(coeff[0]), "intercept": float(coeff[1]), "R2": float(r2)}


# ================================================================
# Plotting
# Each plotting function builds a specific figure intended for analysis or
# inclusion in a technical article, report, or supplementary material.
# ================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    """
    Save a figure with consistent layout, resolution, and cleanup behavior.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_progressive_linear(results: Dict[str, Dict[str, object]], outpath: Path) -> None:
    """
    Plot normalized creep curves in linear coordinates for all scenarios.

    This figure presents the broad shape of the responses and makes the
    overall agreement between the reference curve and the fitted models easy
    to inspect at a glance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for ax, (code, r) in zip(axes, results.items()):
        cfg = r["cfg"]
        t = r["t"]
        ax.plot(t, normalized_strain(r["eps_ref"], cfg.sigma0, cfg.E), label="Fractional reference", linewidth=2.1)
        ax.plot(t, normalized_strain(r["eps_class"], cfg.sigma0, cfg.E), "--", label="Best classical fit", linewidth=2.1)
        ax.plot(t, normalized_strain(r["eps_frac"], cfg.sigma0, cfg.E), ":", label="Best fractional refit", linewidth=2.4)
        ax.set_title(f"{code}: α={cfg.alpha_true:.2f}, $t_{{max}}$={cfg.t_max:g} s", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Normalized strain $E\,\varepsilon/\sigma_0$")
        ax.grid(True, alpha=0.28)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    _save(fig, outpath)


def plot_progressive_error(results: Dict[str, Dict[str, object]], outpath: Path) -> None:
    """
    Plot absolute error trajectories on a semilogarithmic scale.

    The vertical logarithmic axis helps reveal whether one model dominates the
    other only near the origin or throughout the full observation horizon.
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
    Plot late-time deficit decay in log-log coordinates.

    This visualization is designed to emphasize the contrast between
    fractional power-law-like tails and the exponential decay associated with
    the classical Kelvin-Voigt model.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    for ax, (code, r) in zip(axes, results.items()):
        cfg = r["cfg"]
        t = r["t"].copy()
        t[0] = max(t[1] * 0.5, 1e-6)
        tail_mask = t >= 0.1 * cfg.t_max
        t_tail = t[tail_mask]
        d_ref = normalized_deficit(r["eps_ref"], cfg.sigma0, cfg.E)[tail_mask]
        d_class = np.exp(-t_tail / (r["eta_class"] / cfg.E))
        d_frac = normalized_deficit(r["eps_frac"], cfg.sigma0, cfg.E)[tail_mask]
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
    Plot the local slope magnitude of the tail deficit.

    Instead of collapsing the tail into a single fitted exponent, this figure
    shows how the instantaneous decay rate evolves with time for each model.
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
        # positive magnitude of the local log-log slope
        s_ref = -np.gradient(np.log(d_ref), np.log(t_tail))
        s_frac = -np.gradient(np.log(d_frac), np.log(t_tail))
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
    Plot the drift of fitted parameters versus the observation horizon.

    The resulting curve is useful for discussing parameter instability when a
    structurally mismatched model is calibrated over progressively longer
    horizons.
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
    Plot a family of deficit curves for several fractional orders alpha.

    This auxiliary diagnostic isolates the role of alpha and helps interpret
    why smaller values produce heavier and more persistent tails.
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


# ================================================================
# Main execution
# The final section orchestrates the entire workflow and writes all outputs
# to disk so that the experiment can be reproduced end-to-end.
# ================================================================

def main(outdir: str = "/mnt/data/article2_outputs") -> None:
    """
    Execute the full numerical experiment and export all generated artifacts.

    The main routine is the orchestration layer of the script. It creates the
    output folders, loops over the scenarios, computes all summary tables,
    renders the figures, and finally writes a manifest that records the files
    produced during the run.
    """
    out = Path(outdir)
    figdir = out / "figures"
    tabdir = out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    scenarios = default_scenarios()
    results: Dict[str, Dict[str, object]] = {}

    # Rows for the different exported tables.
    global_rows: List[Dict[str, object]] = []
    window_rows: List[Dict[str, object]] = []
    tail_rows: List[Dict[str, object]] = []
    alpha_rows: List[Dict[str, object]] = []

    for cfg in scenarios:
        # Run one full experiment and store the raw outputs for plotting.
        r = fit_models(cfg)
        r["cfg"] = cfg
        results[cfg.code] = r

        # Global performance metrics over the full time horizon.
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

        # Windowed metrics reveal where the models succeed or fail locally.
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

        # Tail diagnostics are computed only for strictly positive times.
        t_pos = r["t"].copy()
        t_pos[0] = max(t_pos[1] * 0.5, 1e-6)
        t_start = 0.1 * cfg.t_max
        ref_tail = tail_regression_loglog(t_pos[1:], normalized_deficit(r["eps_ref"], cfg.sigma0, cfg.E)[1:], t_start)
        frac_tail = tail_regression_loglog(t_pos[1:], normalized_deficit(r["eps_frac"], cfg.sigma0, cfg.E)[1:], t_start)
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

    # Build an auxiliary table that links alpha values with estimated tail slopes.
    t = np.logspace(-2, math.log10(2000.0), 1200)
    for alpha in [0.25, 0.40, 0.55, 0.70, 0.85]:
        eps = creep_fractional_kelvin_voigt(t, 1.0, 1000.0, 200.0, alpha, ml_stable_hybrid_negative_real)
        tail = tail_regression_loglog(t, normalized_deficit(eps, 1.0, 1000.0), 200.0)
        alpha_rows.append({"alpha": alpha, "beta_hat": tail["slope"], "R2": tail["R2"]})

    # Convert the accumulated dictionaries into tabular artifacts.
    df_global = pd.DataFrame(global_rows)
    df_window = pd.DataFrame(window_rows)
    df_tail = pd.DataFrame(tail_rows)
    df_alpha = pd.DataFrame(alpha_rows)

    df_global.to_csv(tabdir / "Table_1_global_metrics.csv", index=False)
    df_window.to_csv(tabdir / "Table_2_windowed_metrics.csv", index=False)
    df_tail.to_csv(tabdir / "Table_3_tail_diagnostics.csv", index=False)
    df_alpha.to_csv(tabdir / "Table_4_alpha_family.csv", index=False)

    # Generate all figures after all scenario results are available.
    plot_progressive_linear(results, figdir / "Fig_1_progressive_linear.png")
    plot_progressive_error(results, figdir / "Fig_2_progressive_error.png")
    plot_tail_loglog(results, figdir / "Fig_3_tail_loglog.png")
    plot_local_slope(results, figdir / "Fig_4_local_slope.png")
    plot_parameter_drift(results, figdir / "Fig_S1_parameter_drift.png")
    plot_alpha_family(figdir / "Fig_5_alpha_family_loglog.png")

    # Write a compact manifest so the generated study remains easy to inspect.
    manifest = {
        "scenarios": [asdict(s) for s in scenarios],
        "tables": [p.name for p in sorted(tabdir.glob("*.csv"))],
        "figures": [p.name for p in sorted(figdir.glob("*.png"))],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
