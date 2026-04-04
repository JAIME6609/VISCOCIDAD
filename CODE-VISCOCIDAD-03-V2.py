#!/usr/bin/env python3
"""
Reconstructed delivery pipeline for Article 3:
"Fractional Creep Under Measurement Noise: Robust Comparison Against Classical Viscoelastic Fits"

This script rebuilds an accessible and publication-consistent output package for the article.
The program does not attempt to recover the original unavailable public pipeline line by line.
Instead, it reproduces the reported scenario definitions and identification summaries from the
article manuscript, and it generates a representative set of deterministic figures and tables
that are consistent with the equations and fitted values reported in the study.

High-level workflow
-------------------
1. Create output directories where figures, tables, and auxiliary files will be stored.
2. Reconstruct the published scenario matrix and the article tables as pandas DataFrames.
3. Compute a derived diagnostic table related to late-time bias under noise.
4. Generate a representative noisy time series for scenario N3 at 2% noise.
5. Rebuild publication-style figures comparing classical and fractional viscoelastic fits.
6. Export CSV tables, PNG figures, a README file, and a JSON manifest.

Scientific context
------------------
The script contrasts two rheological descriptions of creep behavior:
- a classical Kelvin-Voigt model,
- a fractional Kelvin-Voigt model.

The central purpose is to show how the fractional formulation remains more robust in the
presence of measurement noise, especially in strong-memory regimes and in the late-time tail.

Notes on reproducibility
------------------------
- The original public path of the article-3 pipeline was no longer available.
- Therefore, this script is a reconstructed delivery pipeline.
- The aggregate tables reproduced here are the values reported in the article itself.
- The representative time-series figures are generated from the article equations and the
  reported fitted parameters.
- The script is deterministic for the representative noisy realization because it uses a
  fixed random-number seed.

Main outputs
------------
The script writes its outputs to:
    /mnt/data/article3_redelivery_outputs

Inside that directory, it creates:
- figures/   -> regenerated publication-style figures in PNG format,
- tables/    -> CSV versions of the article tables and a derived diagnostic table,
- README.txt -> a concise explanation of the package contents,
- manifest.json -> a machine-readable inventory of the generated outputs,
- representative_case_N3_noise2pct.csv -> the representative time-domain dataset used in
  the main comparison figures.
"""
from __future__ import annotations

# Standard-library imports
# ------------------------
# json: used to write a small machine-readable manifest describing generated files.
# math: kept available for possible numerical support; although not heavily used here,
#       it remains part of the reconstructed environment.
# Path: preferred filesystem abstraction for creating directories and writing files.
import json
import math
from pathlib import Path

# Third-party scientific stack
# ----------------------------
# matplotlib: used for non-interactive figure generation.
# numpy: used for vectorized numerical computation.
# pandas: used to build and export tables.
# mpmath: used for numerical inverse Laplace transform evaluation of the Mittag-Leffler term.
import matplotlib

# The script is intended to run in a headless environment, so it explicitly selects the
# non-interactive Agg backend before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpmath as mp


# ---------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not already exist and return the same Path object.

    Parameters
    ----------
    path : Path
        Directory path that should exist before downstream output is written.

    Returns
    -------
    Path
        The same path object, which allows compact and readable directory-creation code.

    Why this helper exists
    ----------------------
    The script writes several groups of outputs (root folder, figures folder, tables folder).
    Repeating the same directory-creation logic in multiple places would be noisy and less
    maintainable, so this helper centralizes that operation.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path



def savefig(fig: plt.Figure, path: Path) -> None:
    """
    Apply consistent layout handling, save a figure to disk, and then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object that has already been populated with axes and graphical content.
    path : Path
        Output path where the PNG file will be written.

    Why this helper exists
    ----------------------
    Figure-saving boilerplate is the same for every plot in this pipeline:
    - apply tight layout to reduce overlaps,
    - save at a reproducible resolution,
    - use a tight bounding box,
    - close the figure to free memory.

    Centralizing that behavior ensures that all exported figures share the same style and
    prevents memory accumulation when several figures are generated in sequence.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Article data reproduced from the manuscript tables
# ---------------------------------------------------------------------
def build_published_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct the main article tables as pandas DataFrames.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing:
        1. scenario_df : scenario definitions used in the manuscript,
        2. table2_df   : aggregate identification metrics under noise,
        3. table3_df   : parameter summary at 2% noise.

    Interpretation
    --------------
    These tables are not estimated from raw experimental files inside this script. They are
    entered directly from the article manuscript so that the redelivery package remains faithful
    to the reported publication values even though the original public pipeline is unavailable.
    """

    # scenario_df stores the scenario matrix defining the three synthetic test cases.
    # Each row represents one viscoelastic-memory regime studied in the article.
    scenario_df = pd.DataFrame(
        [
            ["N1", "Moderate memory regime", 0.65, 120, 420, 220, "0, 1, 2, 3", 60],
            ["N2", "Strong memory regime", 0.45, 500, 560, 220, "0, 1, 2, 3", 60],
            ["N3", "Very strong memory regime", 0.25, 2000, 720, 220, "0, 1, 2, 3", 60],
        ],
        columns=[
            "Scenario",
            "Description",
            "alpha",
            "t_max_s",
            "n_points",
            "eta_alpha_true",
            "noise_percent_eps_eq",
            "replicates",
        ],
    )

    # table2_df reproduces the aggregate fit-accuracy metrics for classical and fractional
    # models across scenarios and noise levels. Values are reported in micro-strain units
    # (scaled by 1e6 in the table headings).
    table2_df = pd.DataFrame(
        [
            ["N1", 1, 9.60, 0.28, 4.32, 0.15, 34.40, 29.08],
            ["N1", 2, 9.60, 0.87, 4.32, 0.54, 10.97, 8.04],
            ["N1", 3, 9.60, 1.00, 4.32, 0.53, 9.63, 8.10],
            ["N2", 1, 14.40, 0.34, 8.94, 0.25, 42.64, 36.22],
            ["N2", 2, 14.40, 0.81, 8.94, 0.63, 17.71, 14.12],
            ["N2", 3, 14.42, 1.17, 8.94, 0.91, 12.36, 9.78],
            ["N3", 1, 34.29, 0.19, 27.35, 0.16, 184.98, 173.31],
            ["N3", 2, 34.29, 0.58, 27.35, 0.53, 59.44, 51.27],
            ["N3", 3, 34.30, 1.22, 27.35, 0.96, 28.11, 28.37],
        ],
        columns=[
            "Scenario",
            "Noise_percent",
            "MAE_class_x1e6",
            "MAE_frac_x1e6",
            "Late_MAE_class_x1e6",
            "Late_MAE_frac_x1e6",
            "Global_factor",
            "Late_factor",
        ],
    )

    # table3_df reproduces summary statistics for parameter recovery at 2% noise. It includes
    # the apparent classical viscosity, the recovered fractional order, the recovered fractional
    # viscosity, and the reported likelihood-style summary terms P_L_class and P_L_frac.
    table3_df = pd.DataFrame(
        [
            ["N1", "229.8 ± 16.9", 0.645, 0.027, 217.8, 16.9, 1.000, 0.350],
            ["N2", "496.6 ± 39.8", 0.447, 0.028, 215.7, 29.0, 1.000, 0.517],
            ["N3", "1513.8 ± 110.2", 0.248, 0.014, 219.4, 21.1, 1.000, 0.200],
        ],
        columns=[
            "Scenario",
            "eta_class_mean_sd_text",
            "alpha_hat_mean",
            "alpha_hat_sd",
            "eta_alpha_hat_mean",
            "eta_alpha_hat_sd",
            "P_L_class",
            "P_L_frac",
        ],
    )

    return scenario_df, table2_df, table3_df


# ---------------------------------------------------------------------
# Fractional Kelvin-Voigt benchmark for representative figures
# ---------------------------------------------------------------------
def ml_negative_real(t: float, alpha: float, lam: float) -> float:
    """
    Evaluate the Mittag-Leffler term E_alpha(-lam * t^alpha) on the real axis.

    Mathematical background
    -----------------------
    For the fractional Kelvin-Voigt creep response, the relevant special function is
    the Mittag-Leffler function evaluated at a negative real argument:

        E_alpha(-lam * t^alpha)

    In this reconstruction, the quantity is computed through numerical inverse Laplace
    transformation using the identity:

        E_alpha(-lam * t^alpha) = L^{-1}[ s^(alpha-1) / (s^alpha + lam) ](t)

    Parameters
    ----------
    t : float
        Time at which the function is evaluated.
    alpha : float
        Fractional order parameter, typically between 0 and 1 for the cases studied here.
    lam : float
        Characteristic coefficient E / eta_alpha appearing in the fractional model.

    Returns
    -------
    float
        Real-valued approximation of the Mittag-Leffler response term.

    Implementation note
    -------------------
    This approach is computationally more expensive than using a dedicated closed-form
    special-function routine, but it is sufficiently stable and accurate for the relatively
    small number of representative curves generated in this script.
    """

    # At t = 0, the Mittag-Leffler function satisfies E_alpha(0) = 1 exactly.
    # Returning the exact value avoids unnecessary numerical inversion.
    if t <= 0.0:
        return 1.0

    # Convert inputs to mpmath high-precision objects before performing the inverse
    # Laplace transform. This helps numerical stability in special-function evaluation.
    alpha_mp = mp.mpf(alpha)
    lam_mp = mp.mpf(lam)

    # Define the Laplace-domain function whose inverse transform gives the required
    # Mittag-Leffler value in the time domain.
    f = lambda s: (s ** (alpha_mp - 1)) / (s ** alpha_mp + lam_mp)

    # The Talbot method is a standard contour-based algorithm for numerical inverse
    # Laplace transforms. The result may carry a tiny imaginary part from numerical
    # error, so only the real part is kept.
    value = mp.invertlaplace(f, mp.mpf(t), method="talbot")
    return float(mp.re(value))



def fractional_kelvin_voigt_creep(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta_alpha: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute the creep response of the fractional Kelvin-Voigt model.

    Parameters
    ----------
    t : np.ndarray
        Array of time points.
    sigma0 : float
        Applied stress magnitude.
    E : float
        Elastic modulus.
    eta_alpha : float
        Fractional viscosity-like parameter.
    alpha : float
        Fractional order.

    Returns
    -------
    np.ndarray
        Strain response evaluated at the supplied time points.

    Model form
    ----------
    The equilibrium strain is:
        eps_eq = sigma0 / E

    The creep evolution is written as:
        eps(t) = eps_eq * [1 - E_alpha(-lam * t^alpha)]
    with:
        lam = E / eta_alpha

    This function evaluates the Mittag-Leffler term point by point and assembles the
    full strain curve.
    """

    # lam is the effective coefficient controlling the fractional relaxation/creep rate.
    lam = E / eta_alpha

    # eps_eq is the asymptotic equilibrium strain under constant applied stress.
    eps_eq = sigma0 / E

    # Evaluate the Mittag-Leffler term at each time sample. A Python list comprehension is
    # used because the underlying inverse Laplace call is scalar and not natively vectorized.
    ml_vals = np.array([ml_negative_real(float(tt), alpha, lam) for tt in t], dtype=float)

    # Assemble the final fractional creep response.
    return eps_eq * (1.0 - ml_vals)



def classical_kelvin_voigt_creep(
    t: np.ndarray,
    sigma0: float,
    E: float,
    eta: float,
) -> np.ndarray:
    """
    Compute the classical Kelvin-Voigt creep response.

    Parameters
    ----------
    t : np.ndarray
        Array of time points.
    sigma0 : float
        Applied stress magnitude.
    E : float
        Elastic modulus.
    eta : float
        Classical viscosity parameter.

    Returns
    -------
    np.ndarray
        Classical strain response evaluated at the supplied time points.

    Model form
    ----------
    For the standard Kelvin-Voigt model under a step stress:
        eps_eq = sigma0 / E
        tau = eta / E
        eps(t) = eps_eq * [1 - exp(-t / tau)]

    This model has a single exponential memory kernel and is contrasted against the
    fractional model throughout the article.
    """

    # Equilibrium strain under constant stress.
    eps_eq = sigma0 / E

    # Characteristic time constant of the classical Kelvin-Voigt model.
    tau = eta / E

    # Standard exponential creep response.
    return eps_eq * (1.0 - np.exp(-t / tau))



def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    """
    Compute a simple centered moving average using convolution.

    Parameters
    ----------
    y : np.ndarray
        Input signal.
    win : int
        Window length. When win <= 1, the original signal is returned unchanged.

    Returns
    -------
    np.ndarray
        Smoothed signal with the same nominal length as the input.

    Use in this script
    ------------------
    The helper is used to estimate a smooth residual trend and a residual envelope in the
    late-time comparison figure. It is intentionally simple and transparent.
    """

    # A window of length 1 or less implies no smoothing is requested.
    if win <= 1:
        return y.copy()

    # Build a normalized boxcar kernel so that the smoothed result preserves the scale of y.
    kernel = np.ones(win) / win

    # Use 'same' mode to keep the output aligned with the original vector length.
    return np.convolve(y, kernel, mode="same")


# ---------------------------------------------------------------------
# Derived diagnostics used for the accessible redelivery package
# ---------------------------------------------------------------------
def build_derived_bias_table(table2_df: pd.DataFrame, eps_eq: float = 1e-3) -> pd.DataFrame:
    """
    Construct a derived table with late-bias diagnostics inferred from the article values.

    Parameters
    ----------
    table2_df : pd.DataFrame
        Aggregate identification metrics reproduced from the article.
    eps_eq : float, default=1e-3
        Equilibrium strain used to normalize late-time bias measures.

    Returns
    -------
    pd.DataFrame
        Copy of the input table with additional columns describing:
        - classical late bias as a percentage of equilibrium strain,
        - classical bias-to-noise ratio,
        - fractional absolute bias upper bound as a percentage of equilibrium strain,
        - fractional upper-bound-to-noise ratio.

    Rationale
    ---------
    The manuscript reports that the classical late bias is positive and nearly equal to the
    late MAE. Therefore, this reconstructed diagnostic uses Late_MAE_class_x1e6 as a practical
    proxy for classical late bias magnitude.

    For the fractional model, the abstract reports an upper bound on the mean late bias rather
    than scenario-specific values. To avoid inventing unsupported numbers, the script stores
    that reported bound directly.
    """

    # Work on a copy so the original article table remains unchanged and can still be exported
    # exactly as reconstructed from the manuscript.
    df = table2_df.copy()

    # Convert the classical late MAE from the table's micro-scaled units into strain units and
    # then express it as a percentage of the equilibrium strain.
    #
    # The logic is:
    #   Late_MAE_class_x1e6 * 1e-6  -> late MAE in strain units
    #   divide by eps_eq            -> normalize by equilibrium strain
    #   multiply by 100             -> express in percent
    df["Classical_late_bias_percent_eps_eq"] = 100.0 * (df["Late_MAE_class_x1e6"] * 1e-6) / eps_eq

    # Relate the normalized classical late bias to the imposed measurement-noise percentage.
    # Larger values indicate that the tail bias is large relative to the nominal noise level.
    df["Classical_bias_to_noise_ratio"] = df["Classical_late_bias_percent_eps_eq"] / df["Noise_percent"]

    # According to the abstract, the fractional fit keeps the mean late bias within ±0.031%
    # of the equilibrium strain. Because no scenario-by-scenario decomposition is reported,
    # the script records the same bound for all rows rather than creating artificial estimates.
    df["Fractional_abs_bias_upper_bound_percent_eps_eq"] = 0.031

    # Normalize that published bound by the noise percentage to obtain an interpretable
    # upper-bound-to-noise ratio.
    df["Fractional_upper_bound_to_noise_ratio"] = (
        df["Fractional_abs_bias_upper_bound_percent_eps_eq"] / df["Noise_percent"]
    )

    return df


# ---------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------
def figure_1_representative_noisy_fit(outdir: Path) -> pd.DataFrame:
    """
    Generate Figure 1 and export the representative N3 noisy dataset.

    Parameters
    ----------
    outdir : Path
        Root output directory where the figure and CSV file will be written.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the representative clean signal, noisy realization, fitted
        classical response, fitted fractional response, and residuals to the clean reference.

    Purpose of this figure
    ----------------------
    This figure provides an intuitive time-domain comparison between:
    - the clean fractional benchmark,
    - noisy synthetic measurements,
    - the best classical fit,
    - the best fractional fit.

    It also serves as the data source for the later residual-envelope figure.
    """

    # Define the representative physical and fitting parameters for scenario N3 at 2% noise.
    sigma0 = 1.0
    E = 1000.0
    eps_eq = sigma0 / E
    alpha_true = 0.25
    eta_alpha_true = 220.0
    alpha_fit = 0.248
    eta_alpha_fit = 219.4
    eta_class = 1513.8
    noise_frac = 0.02
    t_max = 2000.0
    n = 360

    # Create a uniformly sampled time vector over the full horizon.
    t = np.linspace(0.0, t_max, n)

    # Generate the clean reference from the true fractional model.
    clean = fractional_kelvin_voigt_creep(t, sigma0, E, eta_alpha_true, alpha_true)

    # Generate the fitted fractional and classical responses using the reported recovered
    # parameters from the article summary.
    frac_fit = fractional_kelvin_voigt_creep(t, sigma0, E, eta_alpha_fit, alpha_fit)
    class_fit = classical_kelvin_voigt_creep(t, sigma0, E, eta_class)

    # Use a fixed random seed so that the noisy realization is reproducible across runs.
    rng = np.random.default_rng(20260316)

    # Add Gaussian measurement noise with standard deviation equal to the prescribed fraction
    # of the equilibrium strain.
    noisy = clean + rng.normal(0.0, noise_frac * eps_eq, size=t.size)

    # Store the full representative dataset. Keeping residuals in the exported CSV makes it easy
    # to regenerate downstream analyses without recomputing the full model responses.
    df = pd.DataFrame(
        {
            "time_s": t,
            "clean_fractional_strain": clean,
            "noisy_measurement": noisy,
            "classical_fit": class_fit,
            "fractional_fit": frac_fit,
            "classical_residual_to_clean": class_fit - clean,
            "fractional_residual_to_clean": frac_fit - clean,
        }
    )
    df.to_csv(outdir / "representative_case_N3_noise2pct.csv", index=False)

    # Define the late-time window as the final 30% of the observation horizon.
    late_mask = t >= 0.70 * t_max

    # Build a two-panel figure:
    # left  -> full time horizon,
    # right -> late-time zoom.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    # -----------------------------
    # Panel 1: full-horizon view
    # -----------------------------
    ax = axes[0]

    # Plot only every few noisy points to keep the scatter readable.
    step = 4
    ax.plot(t, clean, linewidth=2.0, label="Clean fractional reference")
    ax.scatter(t[::step], noisy[::step], s=12, alpha=0.45, label="Noisy measurements (2%)")
    ax.plot(t, class_fit, linewidth=2.0, label="Best classical fit")
    ax.plot(t, frac_fit, linewidth=2.0, label="Best fractional fit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    ax.set_title("Representative N3 realization (full horizon)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # -----------------------------
    # Panel 2: late-time zoom
    # -----------------------------
    ax = axes[1]
    ax.plot(t[late_mask], clean[late_mask], linewidth=2.0, label="Clean fractional reference")
    ax.scatter(t[late_mask][::step], noisy[late_mask][::step], s=12, alpha=0.45, label="Noisy measurements")
    ax.plot(t[late_mask], class_fit[late_mask], linewidth=2.0, label="Best classical fit")
    ax.plot(t[late_mask], frac_fit[late_mask], linewidth=2.0, label="Best fractional fit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    ax.set_title("Late-time zoom")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # Save the final figure in the figures subdirectory.
    savefig(fig, outdir / "figures" / "Fig_1_representative_noisy_fit.png")
    return df



def figure_2_parameter_recovery_summary(table3_df: pd.DataFrame, outdir: Path) -> None:
    """
    Generate Figure 2 summarizing parameter recovery across the three scenarios.

    Parameters
    ----------
    table3_df : pd.DataFrame
        Parameter-summary table reconstructed from the manuscript.
    outdir : Path
        Root output directory where the figure will be saved.

    Figure structure
    ----------------
    The figure contains three panels showing:
    1. apparent classical viscosity,
    2. recovered fractional order,
    3. recovered fractional viscosity.

    Error bars represent the reported standard deviations.
    """

    # Scenario labels define the x-axis categories.
    scenarios = table3_df["Scenario"].tolist()

    # The classical viscosity column is stored in the manuscript as formatted text containing
    # mean ± standard deviation. The numeric mean and standard deviation are reproduced here
    # explicitly so the plot can use them directly.
    eta_class_mean = [229.8, 496.6, 1513.8]
    eta_class_sd = [16.9, 39.8, 110.2]

    # Extract the numerical fractional summaries directly from the DataFrame.
    alpha_mean = table3_df["alpha_hat_mean"].to_numpy()
    alpha_sd = table3_df["alpha_hat_sd"].to_numpy()
    eta_alpha_mean = table3_df["eta_alpha_hat_mean"].to_numpy()
    eta_alpha_sd = table3_df["eta_alpha_hat_sd"].to_numpy()

    # Create a categorical x-axis index.
    x = np.arange(len(scenarios))

    # Build a three-panel figure for side-by-side parameter comparison.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # Panel 1: apparent classical viscosity. This highlights how the classical model absorbs
    # memory effects into an inflated effective viscosity, especially for stronger-memory cases.
    axes[0].errorbar(x, eta_class_mean, yerr=eta_class_sd, fmt="o-", capsize=4)
    axes[0].set_xticks(x, scenarios)
    axes[0].set_title("Apparent classical viscosity")
    axes[0].set_ylabel("η*")
    axes[0].grid(True, alpha=0.25)

    # Panel 2: recovered fractional order, which should track the memory strength across cases.
    axes[1].errorbar(x, alpha_mean, yerr=alpha_sd, fmt="o-", capsize=4)
    axes[1].set_xticks(x, scenarios)
    axes[1].set_title("Recovered fractional order")
    axes[1].set_ylabel("α̂")
    axes[1].grid(True, alpha=0.25)

    # Panel 3: recovered fractional viscosity, which should remain comparatively stable near the
    # true value used in the synthetic scenarios.
    axes[2].errorbar(x, eta_alpha_mean, yerr=eta_alpha_sd, fmt="o-", capsize=4)
    axes[2].set_xticks(x, scenarios)
    axes[2].set_title("Recovered fractional viscosity")
    axes[2].set_ylabel("η̂α")
    axes[2].grid(True, alpha=0.25)

    savefig(fig, outdir / "figures" / "Fig_2_parameter_recovery_summary.png")



def figure_3_accuracy_under_noise(table2_df: pd.DataFrame, outdir: Path) -> None:
    """
    Generate Figure 3 comparing global and late-window MAE under different noise levels.

    Parameters
    ----------
    table2_df : pd.DataFrame
        Aggregate identification metrics reconstructed from the manuscript.
    outdir : Path
        Root output directory where the figure will be written.

    Scientific purpose
    ------------------
    This figure directly visualizes how the fractional model outperforms the classical model
    as noise increases, both globally and in the late-time window where tail fidelity is most
    critical.
    """

    # Create two panels that share the same x-axis:
    # left  -> global MAE,
    # right -> late-window MAE.
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharex=True)

    # Explicit scenario order ensures consistent plotting and legend order.
    scenario_order = ["N1", "N2", "N3"]

    for scen in scenario_order:
        # Select rows for one scenario and sort by increasing noise level.
        part = table2_df[table2_df["Scenario"] == scen].sort_values("Noise_percent")
        x = part["Noise_percent"].to_numpy()

        # Dashed curves are used for the classical model; solid curves for the fractional model.
        axes[0].plot(x, part["MAE_class_x1e6"], "o--", label=f"{scen} classical")
        axes[0].plot(x, part["MAE_frac_x1e6"], "o-", label=f"{scen} fractional")
        axes[1].plot(x, part["Late_MAE_class_x1e6"], "o--", label=f"{scen} classical")
        axes[1].plot(x, part["Late_MAE_frac_x1e6"], "o-", label=f"{scen} fractional")

    # Configure the global-MAE panel.
    axes[0].set_title("Global MAE versus noise")
    axes[0].set_xlabel("Noise level (% of equilibrium strain)")
    axes[0].set_ylabel(r"MAE ($\times 10^{-6}$)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    # Configure the late-window-MAE panel.
    axes[1].set_title("Late-window MAE versus noise")
    axes[1].set_xlabel("Noise level (% of equilibrium strain)")
    axes[1].set_ylabel(r"Late MAE ($\times 10^{-6}$)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)

    savefig(fig, outdir / "figures" / "Fig_3_accuracy_under_noise.png")



def figure_4_tail_bias_to_noise_ratio(derived_bias_df: pd.DataFrame, outdir: Path) -> None:
    """
    Generate Figure 4 showing late-time bias diagnostics relative to noise magnitude.

    Parameters
    ----------
    derived_bias_df : pd.DataFrame
        Derived diagnostic table created by build_derived_bias_table().
    outdir : Path
        Root output directory where the figure will be saved.

    Figure interpretation
    ---------------------
    The left panel shows the normalized classical late bias together with the published
    fractional bias bound. The right panel shows how large the bias is relative to the
    measurement-noise percentage.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharex=True)
    scenario_order = ["N1", "N2", "N3"]

    for scen in scenario_order:
        # Select and sort the rows for the current scenario.
        part = derived_bias_df[derived_bias_df["Scenario"] == scen].sort_values("Noise_percent")
        x = part["Noise_percent"].to_numpy()

        # Left panel: normalized classical late bias.
        axes[0].plot(x, part["Classical_late_bias_percent_eps_eq"], "o-", label=f"{scen} classical bias")

        # Right panel: ratio between classical bias and noise percentage.
        axes[1].plot(x, part["Classical_bias_to_noise_ratio"], "o-", label=f"{scen} classical ratio")

    # Fractional upper-bound band from the article abstract.
    # The left panel uses a shaded region representing ±0.031% of equilibrium strain.
    x = np.array([1, 2, 3], dtype=float)
    frac_bound = 0.031 / x
    axes[0].axhspan(-0.031, 0.031, alpha=0.15, label="Fractional mean bias within ±0.031% εeq")

    # The right panel plots the corresponding upper bound normalized by the noise percentage.
    axes[1].plot(x, frac_bound, "--", linewidth=2.0, label="Fractional bias upper bound / noise")

    axes[0].set_title("Classical late bias and fractional bias bound")
    axes[0].set_xlabel("Noise level (% of equilibrium strain)")
    axes[0].set_ylabel("Late bias (% of equilibrium strain)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Tail-bias-to-noise ratio")
    axes[1].set_xlabel("Noise level (% of equilibrium strain)")
    axes[1].set_ylabel("Bias-to-noise ratio")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)

    savefig(fig, outdir / "figures" / "Fig_4_tail_bias_to_noise_ratio.png")



def figure_5_tail_residual_envelope(rep_df: pd.DataFrame, outdir: Path) -> None:
    """
    Generate Figure 5 showing a smoothed residual mean and residual envelope in the late window.

    Parameters
    ----------
    rep_df : pd.DataFrame
        Representative dataset returned by figure_1_representative_noisy_fit().
    outdir : Path
        Root output directory where the figure will be stored.

    Purpose
    -------
    This figure emphasizes late-window behavior by comparing the classical and fractional
    residual trends against the clean reference. The residual envelope gives an intuitive
    sense of spread around the smoothed residual mean.
    """

    # Extract time and residual columns from the representative dataset.
    t = rep_df["time_s"].to_numpy()
    class_res = rep_df["classical_residual_to_clean"].to_numpy()
    frac_res = rep_df["fractional_residual_to_clean"].to_numpy()

    # Restrict attention to the final 30% of the time domain.
    late_mask = t >= 0.70 * t.max()
    t_late = t[late_mask]
    class_late = class_res[late_mask]
    frac_late = frac_res[late_mask]

    # Choose a smoothing window proportional to the late-window size, with a minimum width so
    # that the residual trend and envelope remain reasonably smooth.
    env_win = max(7, int(0.03 * class_late.size))

    # Estimate the local mean residual for each model.
    class_mean = moving_average(class_late, env_win)
    frac_mean = moving_average(frac_late, env_win)

    # Estimate a simple local envelope by smoothing the absolute deviation from the local mean.
    class_env = moving_average(np.abs(class_late - class_mean), env_win)
    frac_env = moving_average(np.abs(frac_late - frac_mean), env_win)

    # Build the residual-envelope plot.
    fig, ax = plt.subplots(figsize=(12.3, 4.7))

    # Classical residual mean and envelope.
    ax.plot(t_late, class_mean, linewidth=2.0, label="Classical residual mean")
    ax.fill_between(
        t_late,
        class_mean - class_env,
        class_mean + class_env,
        alpha=0.18,
        label="Classical residual envelope",
    )

    # Fractional residual mean and envelope.
    ax.plot(t_late, frac_mean, linewidth=2.0, label="Fractional residual mean")
    ax.fill_between(
        t_late,
        frac_mean - frac_env,
        frac_mean + frac_env,
        alpha=0.18,
        label="Fractional residual envelope",
    )

    # Zero reference line helps interpret the sign and magnitude of residual bias.
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual to clean reference")
    ax.set_title("Late-window residual envelope for the representative N3 case")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    savefig(fig, outdir / "figures" / "Fig_5_tail_residual_envelope.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    """
    Execute the full redelivery pipeline.

    Steps performed
    ---------------
    1. Create the root output directory and the figures/tables subdirectories.
    2. Reconstruct the article tables.
    3. Build the derived tail-bias diagnostic table.
    4. Export all tables as CSV files.
    5. Generate the representative time-series dataset and all figures.
    6. Write a README file describing the package.
    7. Write a JSON manifest summarizing the generated files.
    """

    # Create the directory structure used by the output package.
    root = ensure_dir(Path("/mnt/data/article3_redelivery_outputs"))
    figs = ensure_dir(root / "figures")
    tables = ensure_dir(root / "tables")

    # Reconstruct the article tables from the manuscript values.
    scenario_df, table2_df, table3_df = build_published_tables()

    # Compute the supplementary diagnostic table derived from the published metrics.
    derived_bias_df = build_derived_bias_table(table2_df)

    # Export all tables to CSV so they can be inspected independently from the figures.
    scenario_df.to_csv(tables / "Table_1_scenario_matrix.csv", index=False)
    table2_df.to_csv(tables / "Table_2_aggregate_identification_metrics.csv", index=False)
    table3_df.to_csv(tables / "Table_3_parameter_summary_at_2pct_noise.csv", index=False)
    derived_bias_df.to_csv(tables / "Table_4_derived_tail_bias_diagnostics.csv", index=False)

    # Generate the representative figure and retain the resulting dataset for later use in the
    # residual-envelope figure.
    rep_df = figure_1_representative_noisy_fit(root)

    # Generate the remaining summary figures.
    figure_2_parameter_recovery_summary(table3_df, root)
    figure_3_accuracy_under_noise(table2_df, root)
    figure_4_tail_bias_to_noise_ratio(derived_bias_df, root)
    figure_5_tail_residual_envelope(rep_df, root)

    # Prepare a human-readable README that explains the package contents and the reconstructed
    # nature of the pipeline.
    readme = f"""Article 3 redelivery package

Title:
Fractional Creep Under Measurement Noise: Robust Comparison Against Classical Viscoelastic Fits

Contents:
- figures/: publication-style figures regenerated for accessible delivery.
- tables/: CSV versions of the article tables and a derived tail-bias diagnostic table.
- representative_case_N3_noise2pct.csv: time-series data used in Figures 1 and 5.

Important note:
The original public path of the Python pipeline was no longer available at the time of redelivery.
This package therefore includes a reconstructed, publication-consistent pipeline.
The aggregate identification tables are reproduced from the article manuscript itself.
The representative time-domain curves are generated from the article equations and the reported fitted parameters.
"""
    (root / "README.txt").write_text(readme, encoding="utf-8")

    # Create a compact machine-readable manifest so that downstream workflows can inspect what
    # was produced without scanning the directory manually.
    manifest = {
        "title": "Article 3 redelivery outputs",
        "script": "article3_noise_validation_pipeline_redelivery.py",
        "figures": sorted([p.name for p in figs.glob("*.png")]),
        "tables": sorted([p.name for p in tables.glob("*.csv")]),
        "representative_data": "representative_case_N3_noise2pct.csv",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# Standard Python entry point.
# This ensures the pipeline runs only when the file is executed as a script and not when it is
# imported as a module in another program.
if __name__ == "__main__":
    main()
