#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Pan-genome composition, accumulation curves, and Heaps' law analysis

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description: This script performs comprehensive pan-genome compositional analysis,
gene-family accumulation analysis, and Heaps' law modeling to characterize pangenome 
diversity and evaluate whether the pan-genome exhibits an open or closed structure.

The analysis uses pan-genome summary statistics and a genome-by-gene-family presence/absence matrix 
generated from the upstream pan-genome analysis workflow.

Objectives:
a. Characterize the distribution of gene families across pangenome categories, including core, accessory, and unique gene families.
b. Quantify the proportional contribution of each gene category to the overall pangenome.
c. Evaluate gene-family accumulation patterns as additional genomes are incorporated into the analysis.
d. Estimate pan-genome and core-genome accumulation curves using repeated randomized genome orderings.
e. Quantify the discovery rate of new gene families with increasing genome sampling.
f. Fit Heaps' law to gene-family accumulation data and estimate the Heaps' law exponent (alpha).
g. Evaluate pan-genome openness or closure based on the estimated Heaps' law exponent and the specified classification criterion.
h. Generate a three-panel figure summarizing pan-genome composition,accumulation patterns, and Heaps' law modeling.
i. Export analytical results, figures, and execution metadata.

Input files:
a. summary_statistics.tsv: Pan-genome summary statistics containing gene-family counts
   across different pangenome categories.

b. pan-genome_gene_presence_absence.tsv: Genome-by-gene-family presence/absence matrix.

Input data requirements:
- The first column must contain genome identifiers.
- Remaining columns must represent individual gene families.
- Matrix values must indicate gene presence or absence.
- Genome identifiers must be unique.

Command-line arguments:
--summary: Path to the pangenome summary statistics file.
--matrix: Path to the genome-by-gene-family presence/absence matrix.
--output-dir: Directory for saving analysis results and figures.
--permutations: Number of randomized genome orderings.
--seed: Random seed for reproducibility.

Example execution:
python pan_genome_analysis.py \
    --summary summary_statistics.tsv \
    --matrix pan-genome_gene_presence_absence.tsv \
    --output-dir results/pan_genome \
    --permutations 50 \
    --seed 42

Expected outputs:
- Pan-genome composition statistics.
- Gene-family percentage distributions.
- Pan-genome accumulation curve results.
- Core-genome accumulation curve results.
- Heaps' law model estimates.
- Pan-genome openness classification.
- Three-panel analytical figure.
- Execution metadata.

Reproducibility: Randomized genome ordering is controlled using the specified random seed. 
The number of permutations and analysis parameters should be documented to facilitate reproducibility.

Software requirements:
- Python 3.x
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn

Author(s):
    Nafyad Ibrahim Batu (Email: nafyibro4@gmail.com)
    Jagathala Mahalingam Sasikumar
    Hussein S. Abo El-makarem
    Mostafa Ghanem
    Ketema Tafess Tulu
    Seid Mohammed Ebu
    Tharsius R. William Raja
"""
#Import libraries

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from matplotlib.patches import Patch
from scipy.optimize import curve_fit


#Configuration

RANDOM_STATE = 42
N_PERMUTATIONS = 50

GENE_CATEGORY_ORDER = [
    "Core genes",
    "Soft core genes",
    "Shell genes",
    "Cloud genes",
]

GENE_CATEGORY_COLORS = {
    "Core genes": "#009E73",
    "Soft core genes": "#E69F00",
    "Shell genes": "#0072B2",
    "Cloud genes": "#D55E00",
}

FIGURE_NAME = (
    "Pan-genome composition and Heaps law-based "
    "inference of open genome dynamics"
)


#Plotting style

def configure_plot_style() -> None:
    """Set global matplotlib publication-style parameters."""

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "Times New Roman",
        "DejaVu Serif",
    ]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


#Command-line arguments

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze pan-genome composition, accumulation curves, "
            "and Heaps' law."
        )
    )

    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Path to the summary_statistics.tsv file.",
    )

    parser.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="Path to the pan-genome presence/absence matrix.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where figures and result files will be saved.",
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=f"Number of genome-order permutations (default: {N_PERMUTATIONS}).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed (default: {RANDOM_STATE}).",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure resolution in DPI (default: 600).",
    )

    return parser.parse_args()


#Input validation

def validate_input_files(
    summary_file: Path,
    matrix_file: Path,
) -> None:
    """Check that input files exist."""

    if not summary_file.is_file():
        raise FileNotFoundError(
            f"Summary statistics file not found: {summary_file}"
        )

    if not matrix_file.is_file():
        raise FileNotFoundError(
            f"Presence/absence matrix not found: {matrix_file}"
        )


#Load summary statistics

def load_summary_statistics(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load and validate pan-genome summary statistics.

    Required columns:
    - Gene_Category
    - Gene_Count
    """

    df = pd.read_csv(
        file_path,
        sep="\t",
    )

    required_columns = {
        "Gene_Category",
        "Gene_Count",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Summary statistics file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[
        ["Gene_Category", "Gene_Count"]
    ].copy()

    df = df.dropna()

    df["Gene_Category"] = (
        df["Gene_Category"]
        .astype(str)
        .str.strip()
    )

    df["Gene_Count"] = pd.to_numeric(
        df["Gene_Count"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Gene_Count"]
    )

    df = df[
        df["Gene_Category"].isin(
            GENE_CATEGORY_ORDER
        )
    ]

    if df.empty:
        raise ValueError(
            "No valid gene categories were found in the summary file."
        )

    if (df["Gene_Count"] < 0).any():
        raise ValueError(
            "Gene_Count contains negative values."
        )

    ##Combine duplicate category entries, if present.
    df = (
        df.groupby(
            "Gene_Category",
            as_index=False,
        )["Gene_Count"]
        .sum()
    )

    ##Add missing categories with zero counts.
    df = (
        df.set_index("Gene_Category")
        .reindex(
            GENE_CATEGORY_ORDER,
            fill_value=0,
        )
        .reset_index()
    )

    total_genes = df["Gene_Count"].sum()

    if total_genes <= 0:
        raise ValueError(
            "The total number of gene families must be greater than zero."
        )

    df["Gene_Category"] = pd.Categorical(
        df["Gene_Category"],
        categories=GENE_CATEGORY_ORDER,
        ordered=True,
    )

    df = df.sort_values(
        "Gene_Category"
    ).reset_index(drop=True)

    df["Percent"] = (
        df["Gene_Count"] / total_genes
    ) * 100

    return df


#Load presence/absence matrix

def load_presence_absence_matrix(
    file_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the genome-by-gene presence/absence matrix.

    The first column is interpreted as the genome identifier.
    All remaining columns are interpreted as gene families.

    Values greater than zero are treated as gene presence.
    Values equal to zero are treated as gene absence.
    """

    matrix_df = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False,
    )

    if matrix_df.empty:
        raise ValueError(
            "The presence/absence matrix is empty."
        )

    if matrix_df.shape[1] < 2:
        raise ValueError(
            "The matrix must contain a genome ID column "
            "and at least one gene-family column."
        )

    genome_id_column = matrix_df.columns[0]

    genome_ids = (
        matrix_df[genome_id_column]
        .astype(str)
        .str.strip()
    )

    if genome_ids.isna().any() or (genome_ids == "").any():
        raise ValueError(
            "The matrix contains missing or empty genome identifiers."
        )

    if genome_ids.duplicated().any():
        duplicate_ids = genome_ids[
            genome_ids.duplicated()
        ].unique()

        raise ValueError(
            "Duplicate genome identifiers detected: "
            f"{duplicate_ids[:10].tolist()}"
        )

    gene_df = matrix_df.iloc[:, 1:].copy()

    if gene_df.columns.duplicated().any():
        raise ValueError(
            "The matrix contains duplicated gene-family column names."
        )

    gene_df = gene_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if gene_df.isna().any().any():
        raise ValueError(
            "The presence/absence matrix contains missing or "
            "non-numeric gene-family values."
        )

    values = gene_df.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "The presence/absence matrix contains infinite values."
        )

    if (values < 0).any():
        raise ValueError(
            "Negative values were detected in the presence/absence matrix."
        )

    ##Convert all positive values to 1 and zero values to 0
    matrix = (
        values > 0
    ).astype(np.uint8)

    matrix_df = pd.DataFrame(
        matrix,
        index=genome_ids,
        columns=gene_df.columns,
    )

    return matrix_df, matrix


#Pan-genome accumulation analysis

def calculate_accumulation_curves(
    matrix: np.ndarray,
    n_permutations: int,
    random_state: int,
) -> dict:
    """
    Calculate pan-genome and core-genome accumulation curves.

    Each permutation represents a different random ordering of genomes.

    Pan-genome:
        Number of unique gene families observed up to genome n.

    Core genome:
        Number of gene families present in every genome observed
        up to genome n.
    """

    n_genomes, n_genes = matrix.shape

    if n_genomes < 3:
        raise ValueError(
            "At least three genomes are required for accumulation "
            "and Heaps' law analysis."
        )

    if n_genes < 1:
        raise ValueError(
            "The matrix must contain at least one gene family."
        )

    if n_permutations < 2:
        raise ValueError(
            "At least two permutations are required."
        )

    rng = np.random.default_rng(
        random_state
    )

    pan_curves = np.zeros(
        (n_permutations, n_genomes),
        dtype=np.int64,
    )

    core_curves = np.zeros(
        (n_permutations, n_genomes),
        dtype=np.int64,
    )

    for permutation in range(n_permutations):

        order_idx = rng.permutation(
            n_genomes
        )

        cumulative_sum = np.zeros(
            n_genes,
            dtype=np.uint32,
        )

        for i, genome_idx in enumerate(order_idx):

            cumulative_sum += matrix[genome_idx]

            pan_curves[permutation, i] = np.count_nonzero(
                cumulative_sum > 0
            )

            core_curves[permutation, i] = np.count_nonzero(
                cumulative_sum == (i + 1)
            )

    return {
        "pan_curves": pan_curves,
        "core_curves": core_curves,
        "mean_pan": pan_curves.mean(axis=0),
        "std_pan": pan_curves.std(axis=0),
        "mean_core": core_curves.mean(axis=0),
        "std_core": core_curves.std(axis=0),
        "n_genomes": n_genomes,
        "n_genes": n_genes,
    }


#Heaps' law

def heaps_law(
    n: np.ndarray,
    k: float,
    alpha: float,
) -> np.ndarray:
    """
    Heaps' law model for newly observed gene families.

    G(n) = K * n^(-alpha)

    Here, G(n) represents the number of new gene families
    contributed by the nth genome.
    """

    return k * np.power(
        n,
        -alpha,
    )


def fit_heaps_law(
    mean_pan: np.ndarray,
) -> dict:
    """
    Fit Heaps' law to the number of new gene families contributed
    by each additional genome.

    The first genome is excluded from fitting because it contributes
    the initial pan-genome rather than an increment after an existing
    genome collection.
    """

    n_genomes = len(mean_pan)

    x = np.arange(
        1,
        n_genomes + 1,
    )

    new_genes = np.diff(
        np.concatenate(
            ([0], mean_pan)
        )
    )

    fit_x = x[1:]
    fit_y = new_genes[1:]

    valid = (
        np.isfinite(fit_x)
        & np.isfinite(fit_y)
        & (fit_y > 0)
    )

    fit_x = fit_x[valid]
    fit_y = fit_y[valid]

    if len(fit_x) < 3:
        raise ValueError(
            "Too few positive observations are available to fit "
            "Heaps' law. At least three valid observations are required."
        )

    ##Initial parameter estimates
    initial_k = max(
        float(fit_y[0]),
        1e-6,
    )

    initial_alpha = 0.5

    try:
        parameters, covariance = curve_fit(
            heaps_law,
            fit_x,
            fit_y,
            p0=[
                initial_k,
                initial_alpha,
            ],
            bounds=(
                [0, 0],
                [np.inf, 5],
            ),
            maxfev=100000,
        )

    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            "Heaps' law fitting failed. Check the accumulation "
            "data and the number of genomes."
        ) from error

    k_fit, alpha_fit = parameters

    if covariance.shape != (2, 2):
        raise RuntimeError(
            "The covariance matrix returned by curve_fit "
            "has an unexpected shape."
        )

    standard_errors = np.sqrt(
        np.maximum(
            np.diag(covariance),
            0,
        )
    )

    alpha_se = standard_errors[1]

    alpha_ci_low = (
        alpha_fit - 1.96 * alpha_se
    )

    alpha_ci_high = (
        alpha_fit + 1.96 * alpha_se
    )

    ##Constrain the confidence interval to the fitted parameter bounds
    alpha_ci_low = max(
        0,
        alpha_ci_low,
    )

    alpha_ci_high = min(
        5,
        alpha_ci_high,
    )

    fitted_values = heaps_law(
        fit_x,
        k_fit,
        alpha_fit,
    )

    ss_res = np.sum(
        (fit_y - fitted_values) ** 2
    )

    ss_tot = np.sum(
        (fit_y - np.mean(fit_y)) ** 2
    )

    if ss_tot > 0:
        r_squared = 1 - (
            ss_res / ss_tot
        )
    else:
        r_squared = np.nan

    status = (
        "Open pan-genome"
        if alpha_fit < 1
        else "Closed pan-genome"
    )

    return {
        "fit_x": fit_x,
        "fit_y": fit_y,
        "k": float(k_fit),
        "alpha": float(alpha_fit),
        "alpha_se": float(alpha_se),
        "alpha_ci_low": float(alpha_ci_low),
        "alpha_ci_high": float(alpha_ci_high),
        "r_squared": float(r_squared),
        "status": status,
    }


#Save numerical results

def save_analysis_results(
    output_dir: Path,
    summary_df: pd.DataFrame,
    accumulation: dict,
    heaps_results: dict,
) -> None:
    """Save numerical results as tab-separated files."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ##Gene category composition.
    summary_df.to_csv(
        output_dir / "gene_category_composition.tsv",
        sep="\t",
        index=False,
    )

    ##Accumulation curve summary.
    accumulation_df = pd.DataFrame({
        "Number_of_Genomes": np.arange(
            1,
            accumulation["n_genomes"] + 1,
        ),
        "Mean_Pan_Genome": accumulation["mean_pan"],
        "SD_Pan_Genome": accumulation["std_pan"],
        "Mean_Core_Genome": accumulation["mean_core"],
        "SD_Core_Genome": accumulation["std_core"],
    })

    accumulation_df.to_csv(
        output_dir / "pan_genome_accumulation.tsv",
        sep="\t",
        index=False,
    )

    ##Heaps' law observations & fitted values
    heaps_df = pd.DataFrame({
        "Number_of_Genomes": heaps_results["fit_x"],
        "Observed_New_Gene_Families": heaps_results["fit_y"],
        "Fitted_New_Gene_Families": heaps_law(
            heaps_results["fit_x"],
            heaps_results["k"],
            heaps_results["alpha"],
        ),
    })

    heaps_df.to_csv(
        output_dir / "heaps_law_fit.tsv",
        sep="\t",
        index=False,
    )

    ##Model summary
    model_summary = {
        "model": "Heaps' law",
        "equation": "G(n) = K * n^(-alpha)",
        "K": heaps_results["k"],
        "alpha": heaps_results["alpha"],
        "alpha_standard_error": heaps_results["alpha_se"],
        "alpha_95_percent_CI": [
            heaps_results["alpha_ci_low"],
            heaps_results["alpha_ci_high"],
        ],
        "r_squared": heaps_results["r_squared"],
        "classification": heaps_results["status"],
    }

    with open(
        output_dir / "heaps_law_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_summary,
            file,
            indent=4,
        )


#Print heaps' law results

def print_heaps_results(
    results: dict,
) -> None:
    """Print the fitted Heaps' law parameters."""

    print("\n" + "=" * 60)
    print("HEAPS' LAW ANALYSIS")
    print("=" * 60)

    print("\nModel:")
    print("G(n) = K * n^(-alpha)")

    print(
        f"\nK = {results['k']:.6f}"
    )

    print(
        f"Heaps' alpha = {results['alpha']:.6f}"
    )

    print(
        "95% confidence interval = "
        f"{results['alpha_ci_low']:.6f} - "
        f"{results['alpha_ci_high']:.6f}"
    )

    print(
        f"R-squared = {results['r_squared']:.6f}"
    )

    print(
        f"Classification = {results['status']}"
    )

    print("=" * 60)


#Figure generation

def generate_figure(
    summary_df: pd.DataFrame,
    accumulation: dict,
    heaps_results: dict,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Generate and save the three-panel pan-genome figure."""

    configure_plot_style()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(26, 18),
        gridspec_kw={
            "width_ratios": [
                1.4,
                0.8,
                1,
            ]
        },
    )

    #Panel A: PAN-genome composition

    ax1 = axes[0]

    title_fs_A = 32
    label_fs_A = 30
    tick_fs_A = 28
    legend_fs_A = 28

    left = 0

    for _, row in summary_df.iterrows():

        category = row["Gene_Category"]
        count = row["Gene_Count"]

        ax1.barh(
            y=0,
            width=count,
            left=left,
            color=GENE_CATEGORY_COLORS[category],
            edgecolor="black",
            linewidth=0.9,
        )

        left += count

    total_genes = summary_df["Gene_Count"].sum()

    ax1.margins(
        x=0
    )

    ##Retain a small left margin, as in the original figure
    ax1.set_xlim(
        -0.02 * total_genes,
        total_genes,
    )

    legend_elements = [
        Patch(
            facecolor=GENE_CATEGORY_COLORS[row["Gene_Category"]],
            edgecolor="black",
            label=(
                f"{row['Gene_Category']} "
                f"({row['Gene_Count']:,.0f})"
            ),
        )
        for _, row in summary_df.iterrows()
        if row["Gene_Count"] > 0
    ]

    ax1.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=legend_fs_A,
        columnspacing=1.8,
        handletextpad=0.6,
    )

    ax1.set_title(
        "(a) Pan-genome composition",
        fontsize=title_fs_A,
        pad=12,
    )

    ax1.set_xlabel(
        "Number of gene families",
        fontsize=label_fs_A,
    )

    ax1.set_yticks([])

    ax1.tick_params(
        axis="both",
        labelsize=tick_fs_A,
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1.set_ylim(
        -0.45,
        0.45,
    )

    #Panel B: heaps' law

    ax2 = axes[1]

    title_fs_B = 32
    label_fs_B = 30
    tick_fs_B = 28
    legend_fs_B = 28

    fit_x = heaps_results["fit_x"]
    fit_y = heaps_results["fit_y"]

    ax2.scatter(
        fit_x,
        fit_y,
        s=25,
        color="#0072B2",
        label="Observed",
    )

    smooth_x = np.linspace(
        fit_x.min(),
        fit_x.max(),
        500,
    )

    smooth_y = heaps_law(
        smooth_x,
        heaps_results["k"],
        heaps_results["alpha"],
    )

    ax2.plot(
        smooth_x,
        smooth_y,
        color="#D55E00",
        linewidth=3,
        label="Heaps' fit",
    )

    ax2.set_title(
        "(b) Heaps' law model",
        fontsize=title_fs_B,
    )

    ax2.set_xlabel(
        "Number of genomes",
        fontsize=label_fs_B,
    )

    ax2.set_ylabel(
        "New gene families added",
        fontsize=label_fs_B,
    )

    ax2.text(
        0.05,
        0.92,
        f"α = {heaps_results['alpha']:.3f}",
        transform=ax2.transAxes,
        fontsize=28,
    )

    ax2.text(
        0.05,
        0.85,
        (
            "95% CI: "
            f"{heaps_results['alpha_ci_low']:.3f}–"
            f"{heaps_results['alpha_ci_high']:.3f}"
        ),
        transform=ax2.transAxes,
        fontsize=26,
    )

    ax2.text(
        0.05,
        0.77,
        heaps_results["status"],
        transform=ax2.transAxes,
        fontsize=28,
    )

    ax2.legend(
        frameon=False,
        fontsize=legend_fs_B,
    )

    ax2.tick_params(
        axis="both",
        labelsize=tick_fs_B,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    #Panel C: PAN-genome dynamics

    ax3 = axes[2]

    title_fs_C = 32
    label_fs_C = 30
    tick_fs_C = 28
    legend_fs_C = 28

    x = np.arange(
        1,
        accumulation["n_genomes"] + 1,
    )

    mean_pan = accumulation["mean_pan"]
    std_pan = accumulation["std_pan"]

    mean_core = accumulation["mean_core"]
    std_core = accumulation["std_core"]

    ax3.plot(
        x,
        mean_pan,
        color="#D55E00",
        linewidth=2.5,
        label="Pan-genome",
    )

    ax3.fill_between(
        x,
        mean_pan - std_pan,
        mean_pan + std_pan,
        alpha=0.2,
        color="#D55E00",
    )

    ax3.plot(
        x,
        mean_core,
        color="#0072B2",
        linewidth=2.5,
        label="Core genome",
    )

    ax3.fill_between(
        x,
        mean_core - std_core,
        mean_core + std_core,
        alpha=0.2,
        color="#0072B2",
    )

    ax3.set_title(
        "(c) Pan-genome dynamics",
        fontsize=title_fs_C,
    )

    ax3.set_xlabel(
        "Number of genomes",
        fontsize=label_fs_C,
    )

    ax3.set_ylabel(
        "Gene count",
        fontsize=label_fs_C,
    )

    ax3.legend(
        frameon=False,
        fontsize=legend_fs_C,
    )

    ax3.tick_params(
        axis="both",
        labelsize=tick_fs_C,
    )

    ax3.yaxis.set_major_formatter(
        mticker.StrMethodFormatter("{x:,.0f}")
    )

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ##Final layout

    plt.tight_layout()

    ###Shift Panel A slightly to the right.
    pos = ax1.get_position()

    ax1.set_position([
        pos.x0 + 0.04,
        pos.y0,
        pos.width,
        pos.height,
    ])

    plt.subplots_adjust(
        bottom=0.25
    )

    ##Save figure

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_file = output_dir / f"{FIGURE_NAME}.png"
    pdf_file = output_dir / f"{FIGURE_NAME}.pdf"

    fig.savefig(
        png_file,
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_file, pdf_file


##Run metadata

def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    matrix_shape: tuple[int, int],
) -> None:
    """Save parameters and input information for reproducibility."""

    metadata = {
        "analysis": "Pan-genome composition and Heaps' law",
        "date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "summary_file": str(args.summary.resolve()),
        "matrix_file": str(args.matrix.resolve()),
        "number_of_genomes": int(matrix_shape[0]),
        "number_of_gene_families": int(matrix_shape[1]),
        "number_of_permutations": int(args.permutations),
        "random_seed": int(args.seed),
        "figure_dpi": int(args.dpi),
    }

    with open(
        output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )


#Main workflow

def main() -> None:
    """Execute the complete pan-genome analysis."""

    args = parse_arguments()

    validate_input_files(
        args.summary,
        args.matrix,
    )

    if args.permutations < 2:
        raise ValueError(
            "--permutations must be at least 2."
        )

    if args.dpi < 100:
        raise ValueError(
            "--dpi must be at least 100."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading summary statistics...")

    summary_df = load_summary_statistics(
        args.summary
    )

    print("Loading presence/absence matrix...")

    matrix_df, matrix = load_presence_absence_matrix(
        args.matrix
    )

    n_genomes, n_genes = matrix.shape

    print("\nInput data:")
    print(f"Number of genomes: {n_genomes:,}")
    print(f"Number of gene families: {n_genes:,}")

    print("\nCalculating accumulation curves...")

    accumulation = calculate_accumulation_curves(
        matrix=matrix,
        n_permutations=args.permutations,
        random_state=args.seed,
    )

    print("Fitting Heaps' law...")

    heaps_results = fit_heaps_law(
        accumulation["mean_pan"]
    )

    print_heaps_results(
        heaps_results
    )

    print("\nSaving numerical results...")

    save_analysis_results(
        output_dir=args.output_dir,
        summary_df=summary_df,
        accumulation=accumulation,
        heaps_results=heaps_results,
    )

    print("Generating figure...")

    png_file, pdf_file = generate_figure(
        summary_df=summary_df,
        accumulation=accumulation,
        heaps_results=heaps_results,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    save_run_metadata(
        output_dir=args.output_dir,
        args=args,
        matrix_shape=matrix.shape,
    )

    print("\nAnalysis completed successfully.")

    print("\nSaved files:")
    print(f"Figure PNG: {png_file}")
    print(f"Figure PDF: {pdf_file}")
    print(
        "Gene category composition: "
        f"{args.output_dir / 'gene_category_composition.tsv'}"
    )
    print(
        "Accumulation curves: "
        f"{args.output_dir / 'pan_genome_accumulation.tsv'}"
    )
    print(
        "Heaps' law fit: "
        f"{args.output_dir / 'heaps_law_fit.tsv'}"
    )
    print(
        "Heaps' law summary: "
        f"{args.output_dir / 'heaps_law_summary.json'}"
    )
    print(
        "Run metadata: "
        f"{args.output_dir / 'run_metadata.json'}"
    )


#Entry point

if __name__ == "__main__":
    main()

