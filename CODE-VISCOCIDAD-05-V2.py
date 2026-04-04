# -*- coding: utf-8 -*-
"""
article5_pipeline_redelivery.py

Re-delivered Python pipeline for Article 5:
"A Reproducible Benchmark Pipeline for Fractional Viscoelasticity:
 Scenario Generation, Windowed Metrics, and Caputo L1 Identification"

Purpose
-------
This script provides an accessible, reproducible redelivery of the computational
pipeline associated with Article 5. It contains:

1) the benchmark scenario ladder reported in the manuscript,
2) exact article tables embedded as reference data,
3) numerical kernels for classical and fractional Kelvin–Voigt creep,
4) a Caputo L1 implementation for arbitrary stress histories,
5) figure generators consistent with the article narrative, and
6) an export routine that writes publication-ready tables, figures, a README,
   and a manifest.

Important note
--------------
The original article assets existed in internal paths that were not directly
downloadable. This redelivery reconstructs a standalone, accessible pipeline.
Whenever the original figure PNG files are available locally, the script copies
them into the export folder so that the result package remains visually aligned
with the manuscript. If those files are not available, the script generates
numerically consistent fallback figures from the embedded benchmark tables and
the implemented constitutive models.

Dependencies
------------
    numpy
    pandas
    matplotlib
    mpmath

Run
---
    python article5_pipeline_redelivery.py
    python article5_pipeline_redelivery.py --outdir article5_results_redelivery
"""

from __future__ import annotations

# =============================================================================
# GENERAL OVERVIEW OF THE SCRIPT
# =============================================================================
# This program is a complete scientific redelivery pipeline.
#
# In practical terms, it performs four major tasks:
#
# 1. It stores the article's benchmark tables directly inside the code so that
#    the study can be reproduced even if the original external data package is
#    unavailable.
# 2. It implements the main numerical ingredients required to evaluate
#    fractional Kelvin–Voigt creep and its classical counterpart, including a
#    stable negative-real-axis Mittag–Leffler evaluator and a Caputo L1 scheme
#    for arbitrary loading histories.
# 3. It prepares article-style figures and tables, either by copying original
#    image files when they exist or by generating scientifically consistent
#    fallback visualizations from the reconstructed data and constitutive laws.
# 4. It exports all deliverables into a clean folder structure composed of
#    figures, tables, a README summary, and a JSON manifest.
#
# The code is intentionally organized into six blocks:
#   1) Exact reference tables.
#   2) Numerical kernels.
#   3) Helper functions for reproducible exports.
#   4) Figure generators.
#   5) Export routines.
#   6) Main entry point.
#
# All comments added below are written in English and are designed to explain
# both the global scientific intention of the script and the specific purpose of
# each section and function, while preserving the original computational logic.
# =============================================================================

import argparse
import json
import math
import shutil
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd


# =============================================================================
# 1) Exact reference tables reported in the final article
# =============================================================================
# This first block embeds the article's benchmark data directly into the script.
# Doing so has two advantages:
#
# * It makes the pipeline self-contained and reproducible, because it no longer
#   depends on separate spreadsheets or inaccessible manuscript folders.
# * It allows the script to regenerate the article outputs deterministically,
#   which is especially valuable when the original data package cannot be
#   downloaded or when the workflow must be re-executed on a new system.
#
# The objects defined here are not merely auxiliary constants. They represent
# the scientific reference layer of the paper: scenario definitions, parameter
# identification summaries, window-based improvement factors, and validation
# results for the Caputo L1 discretization.
# =============================================================================

ARTICLE_TITLE = (
    "A Reproducible Benchmark Pipeline for Fractional Viscoelasticity: "
    "Scenario Generation, Windowed Metrics, and Caputo L1 Identification"
)

# Scenario ladder used throughout the article.
# Each row describes one benchmark case, including the true fractional order,
# the true fractional viscous parameter, the experiment horizon, the noise level,
# and the ratio between fitted and true elastic modulus.
SCENARIO_LADDER = pd.DataFrame(
    [
        {
            "Scenario": "S1",
            "Purpose": "Ideal short horizon",
            "alpha_true": 0.6700,
            "eta_alpha_true": 215.0,
            "Horizon_s": 60.00,
            "Noise_percent": 0.00e00,
            "Efit_over_Etrue": 1.00,
        },
        {
            "Scenario": "S2",
            "Purpose": "Noisy moderate horizon",
            "alpha_true": 0.5900,
            "eta_alpha_true": 282.0,
            "Horizon_s": 200.0,
            "Noise_percent": 0.5000,
            "Efit_over_Etrue": 1.00,
        },
        {
            "Scenario": "S3",
            "Purpose": "Very-hard long memory",
            "alpha_true": 0.3600,
            "eta_alpha_true": 372.0,
            "Horizon_s": 5000.0,
            "Noise_percent": 0.3000,
            "Efit_over_Etrue": 1.00,
        },
        {
            "Scenario": "S4",
            "Purpose": "Deliberate modulus mismatch",
            "alpha_true": 0.4000,
            "eta_alpha_true": 650.0,
            "Horizon_s": 5000.0,
            "Noise_percent": 0.3000,
            "Efit_over_Etrue": 0.8000,
        },
    ]
)

# Summary of identification outcomes reported in the manuscript.
# This table compares classical fitting against fractional fitting and stores
# the recovered parameters and global absolute-error measures.
IDENTIFICATION_SUMMARY = pd.DataFrame(
    [
        {
            "Scenario": "S1",
            "eta_classical_star": 197.3,
            "eta_alpha_1D_star": 216.0,
            "alpha_2D_star": 0.6812,
            "eta_alpha_2D_star": 231.0,
            "MAE_classical": 1.32e-05,
            "MAE_frac_1D": 7.33e-08,
            "MAE_frac_2D": 3.92e-07,
            "Gain_2D": 33.73,
        },
        {
            "Scenario": "S2",
            "eta_classical_star": 316.0,
            "eta_alpha_1D_star": 279.1,
            "alpha_2D_star": 0.5938,
            "eta_alpha_2D_star": 273.8,
            "MAE_classical": 1.29e-05,
            "MAE_frac_1D": 1.39e-07,
            "MAE_frac_2D": 6.60e-07,
            "Gain_2D": 19.47,
        },
        {
            "Scenario": "S3",
            "eta_classical_star": 5650.9,
            "eta_alpha_1D_star": 380.0,
            "alpha_2D_star": 0.3750,
            "eta_alpha_2D_star": 414.4,
            "MAE_classical": 1.88e-05,
            "MAE_frac_1D": 3.99e-07,
            "MAE_frac_2D": 4.59e-07,
            "Gain_2D": 41.03,
        },
        {
            "Scenario": "S4",
            "eta_classical_star": 12000.0,
            "eta_alpha_1D_star": 1200.0,
            "alpha_2D_star": 0.2000,
            "eta_alpha_2D_star": 1061.8,
            "MAE_classical": 2.72e-04,
            "MAE_frac_1D": 2.07e-04,
            "MAE_frac_2D": 2.99e-05,
            "Gain_2D": 9.08,
        },
    ]
)

# Window-based improvement factors.
# The article emphasizes that model quality should not be judged only globally,
# but also within early, middle, and late observation windows.
WINDOW_IMPROVEMENT = pd.DataFrame(
    [
        {
            "Scenario": "S1",
            "Early_gain_1D": 152.7,
            "Early_gain_2D": 14.69,
            "Middle_gain_1D": 200.1,
            "Middle_gain_2D": 116.9,
            "Late_gain_1D": 202.7,
            "Late_gain_2D": 173.6,
        },
        {
            "Scenario": "S2",
            "Early_gain_1D": 85.43,
            "Early_gain_2D": 21.47,
            "Middle_gain_1D": 96.76,
            "Middle_gain_2D": 19.16,
            "Late_gain_1D": 97.15,
            "Late_gain_2D": 17.84,
        },
        {
            "Scenario": "S3",
            "Early_gain_1D": 48.14,
            "Early_gain_2D": 60.54,
            "Middle_gain_1D": 47.07,
            "Middle_gain_2D": 51.54,
            "Late_gain_1D": 46.87,
            "Late_gain_2D": 28.71,
        },
        {
            "Scenario": "S4",
            "Early_gain_1D": 1.89,
            "Early_gain_2D": 4.92,
            "Middle_gain_1D": 1.34,
            "Middle_gain_2D": 20.21,
            "Late_gain_1D": 1.21,
            "Late_gain_2D": 7.13,
        },
    ]
)

# Validation table for the Caputo L1 discretization.
# The numerical method is assessed under mesh refinement, which allows the code
# to report convergence-related error quantities.
CAPUTO_VALIDATION = pd.DataFrame(
    [
        {
            "dt_s": 0.5000,
            "Grid_points": 161.0,
            "MAE": 1.38e-06,
            "RMSE": 1.00e-05,
            "Max_abs_error": 1.19e-04,
            "Final_relative_error": 1.63e-05,
        },
        {
            "dt_s": 0.2500,
            "Grid_points": 321.0,
            "MAE": 8.69e-07,
            "RMSE": 7.80e-06,
            "Max_abs_error": 1.26e-04,
            "Final_relative_error": 8.15e-06,
        },
        {
            "dt_s": 0.1250,
            "Grid_points": 641.0,
            "MAE": 5.19e-07,
            "RMSE": 5.56e-06,
            "Max_abs_error": 1.20e-04,
            "Final_relative_error": 4.07e-06,
        },
        {
            "dt_s": 0.0625,
            "Grid_points": 1281.0,
            "MAE": 2.95e-07,
            "RMSE": 3.67e-06,
            "Max_abs_error": 1.03e-04,
            "Final_relative_error": 2.03e-06,
        },
    ]
)

# Compact dictionary of headline outcomes used later in the README and manifest.
KEY_RESULTS = {
    "article_title": ARTICLE_TITLE,
    "mean_global_improvement_factor": 25.83,
    "mean_late_window_improvement_factor": 56.82,
    "maximum_late_window_improvement_factor": 173.62,
    "mismatch_case_S4_classical_MAE": 2.72e-04,
    "mismatch_case_S4_fractional_2D_MAE": 2.99e-05,
    "mismatch_case_true_alpha": 0.40,
    "mismatch_case_identified_alpha": 0.20,
    "caputo_finest_dt": 0.0625,
    "caputo_finest_final_relative_error": 2.03e-06,
}


# =============================================================================
# 2) Numerical kernels
# =============================================================================
# This section contains the mathematical core of the script.
#
# It provides:
# * defensive conversion from high-precision mpmath values to standard floats,
# * coefficients for the asymptotic Mittag–Leffler expansion,
# * a power-series evaluator and an asymptotic evaluator on the negative real
#   axis,
# * a higher-accuracy reference evaluator,
# * classical and fractional Kelvin–Voigt creep responses,
# * a Caputo L1 discretization for arbitrary stress histories, and
# * basic error metrics.
#
# In other words, this is the block where the constitutive mathematics and the
# numerical analysis layer meet.
# =============================================================================

def _safe_float(value: mp.mpf | float) -> float:
    """
    Safely convert a high-precision mpmath number into a standard float.

    The conversion is intentionally defensive because special-function
    evaluations can sometimes return values that are difficult to cast cleanly.
    Returning NaN rather than crashing keeps the broader export pipeline alive
    and makes failures easier to diagnose downstream.
    """
    try:
        return float(value)
    except Exception:
        return math.nan


@lru_cache(maxsize=256)
def _inv_gamma_coefficients(alpha: float, max_terms: int, dps: int) -> Tuple[float, ...]:
    """
    Precompute inverse-Gamma coefficients for the asymptotic Mittag–Leffler branch.

    For large arguments x, E_alpha(-x) can be approximated by an inverse-power
    expansion whose coefficients depend on 1 / Gamma(1 - alpha*k). Because the
    same alpha values are typically reused many times during plotting and model
    reconstruction, the result is cached to avoid unnecessary recomputation.
    """
    mp.mp.dps = int(dps)
    a = mp.mpf(alpha)
    coeffs: List[float] = []
    for k in range(1, int(max_terms) + 1):
        arg = mp.mpf("1.0") - a * k
        try:
            arg_float = float(arg)
            # Stop if the Gamma argument lands on a non-positive integer,
            # because Gamma has poles there.
            if arg_float <= 0.0 and abs(arg_float - round(arg_float)) < 1e-12:
                break
        except Exception:
            pass
        try:
            coeffs.append(float(1 / mp.gamma(arg)))
        except Exception:
            break
    return tuple(coeffs)


def ml_power_series_negative_real(
    x: float,
    alpha: float,
    tol: float = 1e-15,
    max_terms: int = 12000,
    dps: int = 80,
) -> float:
    """
    Evaluate E_alpha(-x) through its power-series expansion.

    This branch is appropriate for small or moderate x, where the series is
    numerically stable and converges well. The output is clamped to [0, 1]
    because, for the viscoelastic setting considered here, the physically
    meaningful range of the kernel lies in that interval.
    """
    if x <= 0.0:
        return 1.0
    mp.mp.dps = int(dps)
    x_mp = mp.mpf(x)
    a_mp = mp.mpf(alpha)
    s = mp.mpf("0.0")
    for k in range(int(max_terms)):
        term = (-x_mp) ** k / mp.gamma(a_mp * k + 1)
        s += term
        # Stop once the new term is smaller than the requested tolerance.
        if mp.fabs(term) < tol:
            break
    value = _safe_float(s)
    return max(0.0, min(1.0, value))


def ml_asymptotic_negative_real(
    x: float,
    alpha: float,
    max_terms: int = 18,
    dps: int = 80,
) -> float:
    """
    Evaluate E_alpha(-x) using an asymptotic inverse-power expansion.

    This branch is more appropriate for larger x, where the direct series may
    become inefficient or less stable. The expansion is built from the cached
    inverse-Gamma coefficients computed above.
    """
    if x <= 0.0:
        return 1.0
    coeffs = _inv_gamma_coefficients(float(alpha), int(max_terms), int(dps))
    if not coeffs:
        return 0.0
    inv_x = 1.0 / float(x)
    power = inv_x
    s = 0.0
    sign = 1.0
    for coeff in coeffs:
        s += sign * power * coeff
        sign *= -1.0
        power *= inv_x
    return max(0.0, min(1.0, s))


def ml_reference_negative_real(
    x: float,
    alpha: float,
    series_threshold: float = 0.8,
    dps: int = 60,
) -> float:
    """
    High-quality reference evaluator for E_alpha(-x).

    The function combines two strategies:
    * a very accurate power-series branch for small x, and
    * an integral representation for larger x.

    This routine is more expensive than the practical stable evaluator, but it
    is useful as a trusted reference when a more faithful value is needed.
    """
    if x <= 0.0:
        return 1.0
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    if x < float(series_threshold):
        return ml_power_series_negative_real(x=x, alpha=alpha, tol=1e-25, max_terms=16000, dps=max(80, dps))

    mp.mp.dps = int(dps)
    a = mp.mpf(alpha)
    x_mp = mp.mpf(x)
    xa = x_mp ** (1 / a)
    sin_term = mp.sin(mp.pi * a)
    cos_term = mp.cos(mp.pi * a)

    # Integral representation specialized for the negative real axis.
    def integrand(r: mp.mpf) -> mp.mpf:
        numerator = mp.e ** (-xa * r) * (r ** (a - 1)) * sin_term
        denominator = (r ** (2 * a)) + (2 * (r ** a) * cos_term) + 1
        return numerator / denominator

    value = (1 / mp.pi) * mp.quad(integrand, [0, 1, mp.inf])
    value_f = _safe_float(value)
    return max(0.0, min(1.0, value_f))


def ml_stable_negative_real(x: float, alpha: float) -> float:
    """
    Practical stable evaluator for E_alpha(-x).

    The code switches between the power-series branch and the asymptotic branch
    using a simple threshold. This function is the default evaluator used by the
    fractional Kelvin–Voigt creep routine because it balances stability and
    computational cost.
    """
    if x < 1.5:
        return ml_power_series_negative_real(x=x, alpha=alpha, tol=1e-16, max_terms=12000, dps=80)
    return ml_asymptotic_negative_real(x=x, alpha=alpha, max_terms=18, dps=80)


def creep_fractional_kelvin_voigt(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta_alpha: float,
    alpha: float,
    evaluator=ml_stable_negative_real,
) -> np.ndarray:
    """
    Compute the fractional Kelvin–Voigt creep response under constant stress.

    For each time value t_i, the routine evaluates the fractional viscoelastic
    kernel through the chosen Mittag–Leffler evaluator and converts it into the
    corresponding strain response.
    """
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti <= 0.0:
            out[i] = 0.0
        else:
            x = (E / eta_alpha) * (ti ** alpha)
            out[i] = (sigma0 / E) * (1.0 - evaluator(x, alpha))
    return out


def creep_classic_kelvin_voigt(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta: float,
) -> np.ndarray:
    """
    Compute the classical Kelvin–Voigt creep response.

    This is the exponential-memory baseline used throughout the article for
    comparison against the fractional long-memory model.
    """
    t = np.asarray(t, dtype=float)
    tau = eta / E
    return (sigma0 / E) * (1.0 - np.exp(-t / tau))


def caputo_l1_creep_response(
    t: np.ndarray,
    sigma: np.ndarray,
    E: float,
    eta_alpha: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute the fractional creep response for an arbitrary stress history using
    the Caputo L1 time-discretization scheme.

    This function is especially relevant because the analytical closed form used
    for constant stress is not directly available for general loading histories.
    The L1 scheme provides a discrete approximation to the Caputo derivative on
    a uniform time grid and allows the script to reconstruct responses beyond
    the simple step-creep case.
    """
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if t.ndim != 1 or sigma.ndim != 1 or len(t) != len(sigma):
        raise ValueError("t and sigma must be one-dimensional arrays with equal length.")
    if len(t) < 2:
        raise ValueError("At least two time nodes are required.")

    dt = float(t[1] - t[0])
    if np.max(np.abs(np.diff(t) - dt)) > 1e-12:
        raise ValueError("The Caputo L1 implementation requires a uniform time step.")

    # Precompute the fractional L1 weights.
    c = 1.0 / (math.gamma(2.0 - alpha) * (dt ** alpha))
    b = np.array([(k + 1) ** (1.0 - alpha) - (k ** (1.0 - alpha)) for k in range(len(t))], dtype=float)

    eps = np.zeros_like(t, dtype=float)
    # Initial strain comes from the instantaneous elastic balance.
    eps[0] = sigma[0] / E

    for n in range(1, len(t)):
        history_sum = 0.0
        # Accumulate the memory contribution of all previous strain increments.
        for k in range(1, n):
            history_sum += b[k] * (eps[n - k] - eps[n - k - 1])

        numerator = sigma[n] + eta_alpha * c * eps[n - 1] - eta_alpha * c * history_sum
        denominator = E + eta_alpha * c
        eps[n] = numerator / denominator
    return eps


def compute_metrics(y_ref: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard pointwise error metrics between a reference curve and a prediction.

    The returned dictionary is intentionally compact and publication-friendly,
    containing mean squared error, root mean squared error, mean absolute error,
    and maximum absolute error.
    """
    err = np.asarray(y_pred) - np.asarray(y_ref)
    return {
        "MSE": float(np.mean(err ** 2)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MaxAbsError": float(np.max(np.abs(err))),
    }


# =============================================================================
# 3) Helper functions for reproducible exports
# =============================================================================
# The next block converts the embedded tables and scenario rows into objects that
# are easier to reuse across the figure-generation and export layers. It also
# centralizes path lookup for figure assets and constructs a composite stress
# history used in the arbitrary-loading experiment.
# =============================================================================

@dataclass
class ScenarioConfig:
    """
    Lightweight structured container for one benchmark scenario.

    Using a dataclass here improves readability and makes scenario access more
    explicit than repeatedly indexing raw DataFrame rows.
    """
    name: str
    purpose: str
    alpha_true: float
    eta_alpha_true: float
    horizon_s: float
    noise_percent: float
    efit_over_etrue: float



def scenario_configs() -> List[ScenarioConfig]:
    """
    Convert the scenario DataFrame into a list of ScenarioConfig objects.

    This makes downstream plotting code cleaner and more expressive.
    """
    cfgs: List[ScenarioConfig] = []
    for row in SCENARIO_LADDER.to_dict(orient="records"):
        cfgs.append(
            ScenarioConfig(
                name=row["Scenario"],
                purpose=row["Purpose"],
                alpha_true=float(row["alpha_true"]),
                eta_alpha_true=float(row["eta_alpha_true"]),
                horizon_s=float(row["Horizon_s"]),
                noise_percent=float(row["Noise_percent"]),
                efit_over_etrue=float(row["Efit_over_Etrue"]),
            )
        )
    return cfgs



def article_table_exports() -> Dict[str, pd.DataFrame]:
    """
    Return the article tables with the filenames that should be used on export.

    The dictionary format is convenient because it binds each DataFrame directly
    to its intended CSV name.
    """
    return {
        "Table_1_scenario_ladder.csv": SCENARIO_LADDER.copy(),
        "Table_2_identification_summary.csv": IDENTIFICATION_SUMMARY.copy(),
        "Table_3_window_improvement.csv": WINDOW_IMPROVEMENT.copy(),
        "Table_4_caputo_validation.csv": CAPUTO_VALIDATION.copy(),
    }



def candidate_figure_sources() -> Dict[str, List[Path]]:
    """
    Provide a list of possible source paths for each article figure.

    The script first attempts to recover original PNG assets from these
    locations. If none of them exist, it will later generate a fallback figure
    numerically.
    """
    return {
        "Fig_1_scenario_overview.png": [
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/a7ef1b60c1a5460ebc05992262f7d652/mnt/data/article5_outputs/figures/Fig_1_scenario_overview.png"),
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/67c3efa767354dbb9a5d91bed804cc0d/mnt/data/article5_outputs/figures/Fig_1_scenario_overview.png"),
        ],
        "Fig_2_global_and_late_error_bars.png": [
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/a7ef1b60c1a5460ebc05992262f7d652/mnt/data/article5_outputs/figures/Fig_2_global_and_late_error_bars.png"),
        ],
        "Fig_3_parameter_recovery.png": [
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/a7ef1b60c1a5460ebc05992262f7d652/mnt/data/article5_outputs/figures/Fig_3_parameter_recovery.png"),
        ],
        "Fig_4_model_mismatch_deep_dive.png": [
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/172c9251e4ef425b98d9778060992c06/mnt/data/article5_outputs/figures/Fig_4_model_mismatch_deep_dive.png"),
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/a7ef1b60c1a5460ebc05992262f7d652/mnt/data/article5_outputs/figures/Fig_4_model_mismatch_deep_dive.png"),
        ],
        "Fig_5_caputo_l1_validation.png": [
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/172c9251e4ef425b98d9778060992c06/mnt/data/article5_outputs/figures/Fig_5_caputo_l1_validation.png"),
        ],
        "Fig_6_arbitrary_loading_history.png": [
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/172c9251e4ef425b98d9778060992c06/mnt/data/article5_outputs/figures/Fig_6_arbitrary_loading_history.png"),
        ],
        "Fig_7_window_improvement_heatmap.png": [
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/172c9251e4ef425b98d9778060992c06/mnt/data/article5_outputs/figures/Fig_7_window_improvement_heatmap.png"),
            Path("/mnt/data/user-y4714XtLS7FJPlLd6reosI8y/67c3efa767354dbb9a5d91bed804cc0d/mnt/data/article5_outputs/figures/Fig_7_window_improvement_heatmap.png"),
        ],
    }



def composite_stress_history(t: np.ndarray) -> np.ndarray:
    """
    Define a piecewise-constant stress history for the arbitrary-loading example.

    The resulting signal contains several loading plateaus of different
    amplitudes so that the Caputo L1 solver can be tested beyond the single
    step-load case.
    """
    t = np.asarray(t, dtype=float)
    sigma = np.zeros_like(t)
    sigma[(t >= 5.0) & (t < 25.0)] = 1.0
    sigma[(t >= 25.0) & (t < 45.0)] = 0.45
    sigma[(t >= 45.0) & (t < 70.0)] = 1.35
    sigma[(t >= 70.0) & (t < 95.0)] = 0.75
    sigma[t >= 95.0] = 0.20
    return sigma


# =============================================================================
# 4) Figure generators (used when original PNG assets are unavailable)
# =============================================================================
# These functions recreate the visual layer of the article directly from the
# embedded tables and constitutive-model routines. Each function writes one PNG
# file to disk. If the original images cannot be copied from the candidate
# locations, these generators ensure that the manuscript package is still
# complete and scientifically interpretable.
# =============================================================================

def plot_fig1_scenario_overview(outpath: Path) -> None:
    """
    Generate the multi-panel scenario overview figure.

    Each panel compares the trusted fractional reference with the best classical
    fit and the best two-dimensional fractional fit.
    """
    E_true = 1200.0
    sigma0 = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, cfg in zip(axes, scenario_configs()):
        t = np.linspace(0.0, cfg.horizon_s, 1200)
        ref = creep_fractional_kelvin_voigt(
            t=t,
            sigma0=sigma0,
            E=E_true,
            eta_alpha=cfg.eta_alpha_true,
            alpha=cfg.alpha_true,
        )

        fit_row = IDENTIFICATION_SUMMARY.loc[IDENTIFICATION_SUMMARY["Scenario"] == cfg.name].iloc[0]
        E_fit = cfg.efit_over_etrue * E_true

        classical = creep_classic_kelvin_voigt(t=t, sigma0=sigma0, E=E_fit, eta=float(fit_row["eta_classical_star"]))
        frac2d = creep_fractional_kelvin_voigt(
            t=t,
            sigma0=sigma0,
            E=E_fit,
            eta_alpha=float(fit_row["eta_alpha_2D_star"]),
            alpha=float(fit_row["alpha_2D_star"]),
        )

        ax.plot(t, ref, linewidth=2.0, label="Fractional reference")
        ax.plot(t, classical, "--", linewidth=1.8, label="Best classical fit")
        ax.plot(t, frac2d, ":", linewidth=2.0, label="Best fractional 2D fit")
        ax.set_title(f"{cfg.name}: {cfg.purpose}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Scenario Overview: Fractional Reference vs. Best Fits", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_fig2_error_bars(outpath: Path) -> None:
    """
    Generate the bar chart of global and late-window improvement factors.

    This figure summarizes how much the two-dimensional fractional fit improves
    over the classical fit across scenarios.
    """
    scenarios = IDENTIFICATION_SUMMARY["Scenario"].tolist()
    global_gain = IDENTIFICATION_SUMMARY["Gain_2D"].to_numpy(dtype=float)
    late_gain = WINDOW_IMPROVEMENT["Late_gain_2D"].to_numpy(dtype=float)

    x = np.arange(len(scenarios))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, global_gain, width=width, label="Global gain (2D fractional vs classical)")
    ax.bar(x + width / 2, late_gain, width=width, label="Late-window gain (2D fractional vs classical)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Improvement factor")
    ax.set_xlabel("Scenario")
    ax.set_title("Global and Late-Window Improvement Factors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_fig3_parameter_recovery(outpath: Path) -> None:
    """
    Generate the parameter-recovery figure.

    The left panel compares true and recovered fractional orders, while the
    right panel compares true and recovered fractional viscous parameters.
    """
    merged = SCENARIO_LADDER.merge(IDENTIFICATION_SUMMARY, on="Scenario", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    axes[0].plot(merged["alpha_true"], merged["alpha_true"], linewidth=1.5, label="Ideal recovery")
    axes[0].scatter(merged["alpha_true"], merged["alpha_2D_star"], s=55, marker="o", label="Recovered α*")
    for _, row in merged.iterrows():
        axes[0].annotate(row["Scenario"], (row["alpha_true"], row["alpha_2D_star"]), xytext=(5, 4), textcoords="offset points")
    axes[0].set_xlabel("True fractional order α")
    axes[0].set_ylabel("Recovered fractional order α*")
    axes[0].set_title("Fractional-order recovery")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(merged["eta_alpha_true"], merged["eta_alpha_true"], linewidth=1.5, label="Ideal recovery")
    axes[1].scatter(merged["eta_alpha_true"], merged["eta_alpha_2D_star"], s=55, marker="o", label="Recovered ηα*")
    for _, row in merged.iterrows():
        axes[1].annotate(row["Scenario"], (row["eta_alpha_true"], row["eta_alpha_2D_star"]), xytext=(5, 4), textcoords="offset points")
    axes[1].set_xlabel("True ηα")
    axes[1].set_ylabel("Recovered ηα*")
    axes[1].set_title("Viscous-order parameter recovery")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_fig4_model_mismatch_deep_dive(outpath: Path) -> None:
    """
    Generate the detailed mismatch analysis for scenario S4.

    This is the deliberately challenging case where the fitted elastic modulus
    differs from the true modulus, making it possible to visualize how the
    classical model fails relative to the fractional alternative.
    """
    cfg_row = SCENARIO_LADDER.loc[SCENARIO_LADDER["Scenario"] == "S4"].iloc[0]
    fit_row = IDENTIFICATION_SUMMARY.loc[IDENTIFICATION_SUMMARY["Scenario"] == "S4"].iloc[0]
    E_true = 1200.0
    E_fit = float(cfg_row["Efit_over_Etrue"]) * E_true
    sigma0 = 1.0
    t = np.linspace(0.0, float(cfg_row["Horizon_s"]), 1800)

    ref = creep_fractional_kelvin_voigt(
        t=t,
        sigma0=sigma0,
        E=E_true,
        eta_alpha=float(cfg_row["eta_alpha_true"]),
        alpha=float(cfg_row["alpha_true"]),
    )
    classical = creep_classic_kelvin_voigt(t=t, sigma0=sigma0, E=E_fit, eta=float(fit_row["eta_classical_star"]))
    frac2d = creep_fractional_kelvin_voigt(
        t=t,
        sigma0=sigma0,
        E=E_fit,
        eta_alpha=float(fit_row["eta_alpha_2D_star"]),
        alpha=float(fit_row["alpha_2D_star"]),
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)

    axes[0].plot(t, ref, linewidth=2.0, label="Trusted fractional reference")
    axes[0].plot(t, classical, "--", linewidth=1.8, label="Best classical fit")
    axes[0].plot(t, frac2d, ":", linewidth=2.0, label="Best blind fractional 2D fit")
    axes[0].set_ylabel("Strain")
    axes[0].set_title("S4 Deliberate Modulus-Mismatch Scenario")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(t, classical - ref, "--", linewidth=1.8, label="Classical residual")
    axes[1].plot(t, frac2d - ref, ":", linewidth=2.0, label="Fractional 2D residual")
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_fig5_caputo_validation(outpath: Path) -> None:
    """
    Generate the Caputo L1 validation figure.

    The left panel shows how error decreases with time-step refinement.
    The right panel compares the numerical L1 response against the analytical
    fractional step-creep reference.
    """
    E = 1200.0
    eta_alpha = 372.0
    alpha = 0.36
    sigma0 = 1.0
    horizon = 80.0

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # Left panel: convergence metrics from the exact article table.
    dt = CAPUTO_VALIDATION["dt_s"].to_numpy(dtype=float)
    final_rel = CAPUTO_VALIDATION["Final_relative_error"].to_numpy(dtype=float)
    max_abs = CAPUTO_VALIDATION["Max_abs_error"].to_numpy(dtype=float)
    axes[0].loglog(dt, final_rel, marker="o", linewidth=2.0, label="Final relative error")
    axes[0].loglog(dt, max_abs, marker="s", linewidth=2.0, label="Max absolute error")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Time step Δt (s)")
    axes[0].set_ylabel("Error")
    axes[0].set_title("Caputo L1 refinement trend")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    # Right panel: analytical step-creep reference vs numerical approximations.
    t_ref = np.linspace(0.0, horizon, 1601)
    ref = creep_fractional_kelvin_voigt(t_ref, sigma0=sigma0, E=E, eta_alpha=eta_alpha, alpha=alpha)
    axes[1].plot(t_ref, ref, linewidth=2.0, label="Analytical fractional reference")

    for dt_i in CAPUTO_VALIDATION["dt_s"].to_numpy(dtype=float):
        t_num = np.arange(0.0, horizon + dt_i, dt_i)
        sigma = np.ones_like(t_num) * sigma0
        sigma[0] = 0.0
        eps_num = caputo_l1_creep_response(t_num, sigma, E=E, eta_alpha=eta_alpha, alpha=alpha)
        axes[1].plot(t_num, eps_num, linewidth=1.4, label=f"L1, Δt={dt_i:g}")

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Strain")
    axes[1].set_title("L1 approximation vs analytical step creep")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_fig6_arbitrary_loading(outpath: Path) -> None:
    """
    Generate the arbitrary-loading-history figure.

    The first panel shows the imposed piecewise stress signal and the second
    panel shows the resulting fractional Kelvin–Voigt strain response computed
    with the Caputo L1 scheme.
    """
    E = 1200.0
    eta_alpha = 372.0
    alpha = 0.36
    t = np.arange(0.0, 120.0 + 0.125, 0.125)
    sigma = composite_stress_history(t)
    eps = caputo_l1_creep_response(t, sigma, E=E, eta_alpha=eta_alpha, alpha=alpha)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
    axes[0].plot(t, sigma, linewidth=2.0)
    axes[0].set_ylabel("Stress")
    axes[0].set_title("Composite loading history")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, eps, linewidth=2.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Strain")
    axes[1].set_title("Fractional Kelvin–Voigt response (Caputo L1)")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_fig7_heatmap(outpath: Path) -> None:
    """
    Generate a heatmap of 2D fractional improvement factors across time windows.

    This figure provides a compact visual summary of where the fractional model
    gains the most relative to the classical baseline.
    """
    data = WINDOW_IMPROVEMENT.set_index("Scenario")[["Early_gain_2D", "Middle_gain_2D", "Late_gain_2D"]].to_numpy(dtype=float)
    scenarios = WINDOW_IMPROVEMENT["Scenario"].tolist()
    windows = ["Early", "Middle", "Late"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title("2D Fractional Improvement Factors by Time Window")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Improvement factor")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# Mapping between figure names and their fallback generator functions.
FALLBACK_GENERATORS = {
    "Fig_1_scenario_overview.png": plot_fig1_scenario_overview,
    "Fig_2_global_and_late_error_bars.png": plot_fig2_error_bars,
    "Fig_3_parameter_recovery.png": plot_fig3_parameter_recovery,
    "Fig_4_model_mismatch_deep_dive.png": plot_fig4_model_mismatch_deep_dive,
    "Fig_5_caputo_l1_validation.png": plot_fig5_caputo_validation,
    "Fig_6_arbitrary_loading_history.png": plot_fig6_arbitrary_loading,
    "Fig_7_window_improvement_heatmap.png": plot_fig7_heatmap,
}


# =============================================================================
# 5) Export routines
# =============================================================================
# This block is responsible for creating the final deliverable package. It
# writes the tables, copies or generates the figures, creates a README file for
# human readers, and also creates a JSON manifest for machine-readable tracking.
# =============================================================================

def copy_or_generate_figure(name: str, destination: Path) -> str:
    """
    Copy an existing figure if available; otherwise generate it from the code.

    The returned string records whether the image was copied from an original
    location or generated by the redelivery pipeline.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    for src in candidate_figure_sources().get(name, []):
        if src.exists():
            shutil.copy2(src, destination)
            return f"copied from {src}"
    generator = FALLBACK_GENERATORS.get(name)
    if generator is None:
        raise ValueError(f"No source or generator available for {name}")
    generator(destination)
    return "generated from redelivery pipeline"



def write_tables(tables_dir: Path) -> List[str]:
    """
    Export all article tables to CSV files.

    The function returns the list of written file paths so they can be recorded
    in the final package summary.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for filename, df in article_table_exports().items():
        outpath = tables_dir / filename
        df.to_csv(outpath, index=False)
        written.append(str(outpath))
    return written



def write_readme(outdir: Path) -> Path:
    """
    Write a human-readable README summarizing the contents of the output package.

    Besides listing what is included, the README also reports the manuscript's
    main headline numerical results.
    """
    readme = outdir / "README.txt"
    text = f"""Article 5 results redelivery
====================================

Title:
{ARTICLE_TITLE}

This folder contains an accessible redelivery of the computational outputs for
Article 5. The package includes:

1) figures/
   - the seven benchmark figures used in the manuscript,
2) tables/
   - CSV exports of the four article tables,
3) manifest.json
   - a machine-readable summary of the package.

Reference headline results
--------------------------
Average global fractional improvement factor: {KEY_RESULTS['mean_global_improvement_factor']:.2f}
Average late-window fractional improvement factor: {KEY_RESULTS['mean_late_window_improvement_factor']:.2f}
Maximum late-window improvement factor: {KEY_RESULTS['maximum_late_window_improvement_factor']:.2f}
S4 classical MAE: {KEY_RESULTS['mismatch_case_S4_classical_MAE']:.2e}
S4 fractional 2D MAE: {KEY_RESULTS['mismatch_case_S4_fractional_2D_MAE']:.2e}
S4 alpha shift: {KEY_RESULTS['mismatch_case_true_alpha']:.2f} -> {KEY_RESULTS['mismatch_case_identified_alpha']:.2f}
Finest L1 final relative error: {KEY_RESULTS['caputo_finest_final_relative_error']:.2e} at Δt={KEY_RESULTS['caputo_finest_dt']}

The Python script article5_pipeline_redelivery.py can regenerate this package.
"""
    readme.write_text(text, encoding="utf-8")
    return readme



def write_manifest(outdir: Path, figure_logs: Dict[str, str]) -> Path:
    """
    Write a machine-readable JSON manifest describing the exported package.

    The manifest includes the figure provenance, exported table names, and key
    numerical results that define the scientific summary of the article.
    """
    manifest = {
        "article_title": ARTICLE_TITLE,
        "tables": list(article_table_exports().keys()),
        "figures": figure_logs,
        "key_results": KEY_RESULTS,
    }
    outpath = outdir / "manifest.json"
    outpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outpath



def export_package(outdir: Path) -> Dict[str, object]:
    """
    Build the complete output package in the requested directory.

    This function orchestrates the full export workflow: tables, figures,
    README, and manifest. It returns a structured summary of all written
    artifacts.
    """
    outdir = Path(outdir)
    figs_dir = outdir / "figures"
    tables_dir = outdir / "tables"

    written_tables = write_tables(tables_dir)
    figure_logs: Dict[str, str] = {}

    for fig_name in FALLBACK_GENERATORS.keys():
        destination = figs_dir / fig_name
        figure_logs[fig_name] = copy_or_generate_figure(fig_name, destination)

    readme = write_readme(outdir)
    manifest = write_manifest(outdir, figure_logs)

    return {
        "outdir": str(outdir),
        "tables": written_tables,
        "figures": {k: str(figs_dir / k) for k in FALLBACK_GENERATORS.keys()},
        "readme": str(readme),
        "manifest": str(manifest),
    }


# =============================================================================
# 6) Main
# =============================================================================
# The main entry point parses the output directory, launches the export process,
# and prints a JSON summary so the execution result can be inspected quickly.
# =============================================================================

def main() -> None:
    """
    Parse command-line arguments and execute the full redelivery export.
    """
    parser = argparse.ArgumentParser(description="Redelivery pipeline for Article 5.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="/mnt/data/article5_results_redelivery",
        help="Output directory for figures, tables, and manifest.",
    )
    args = parser.parse_args()
    result = export_package(Path(args.outdir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
