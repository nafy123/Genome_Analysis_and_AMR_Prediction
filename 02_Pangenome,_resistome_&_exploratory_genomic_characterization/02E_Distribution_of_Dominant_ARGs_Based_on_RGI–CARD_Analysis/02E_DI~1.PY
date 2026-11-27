#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Distribution of AMR genes & AMR mechanisms

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description: This script characterizes the distribution and frequency of AMR genes identified across the analyzed bacterial 
genomes and summarizes their associated AMR mechanisms.

The analysis integrates an ARG presence/absence matrix with a curated AMR gene and resistance mechanism annotation 
table to generate a comparative visualization of the most frequently detected ARGs and their associated resistance mechanisms.

Objectives:
a. Characterize the distribution of AMR genes across the analyzed bacterial genomes.
b. Calculate the frequency of occurrence of individual ARGs based on their presence/absence profiles.
c. Identify and visualize the top 35 most frequently detected AMR genes.
d. Associate identified ARGs with their corresponding AMR mechanisms using a curated annotation table.
e. Summarize the distribution of ARGs according to their associated resistance mechanisms.
f. Generate a two-panel figure illustrating ARG frequency and resistance mechanism distribution.
g. Export analytical results and publication-oriented figures.

Analytical workflow:
a. Load and validate the ARG presence/absence matrix.
b. Extract genome identifiers and ARG profiles.
c. Calculate the number of genomes carrying each ARG.
d. Rank ARGs according to their detection frequency.
e. Select the 35 most frequently detected ARGs, or the number specified through the command-line argument.
f. Load the curated ARG annotation and resistance mechanism table.
g. Integrate ARG frequency information with resistance mechanism annotations.
h. Summarize ARG distributions across resistance mechanisms.
i. Generate Figure 2 with two analytical panels:
   - Panel A: Distribution of the 35 most frequently detected AMR genes.
   - Panel B: Distribution of ARGs grouped by their associated AMR mechanisms.
j. Save the resulting figure and processed analytical data.

Input files:

a. ARG_presence_absence_matrix.tsv: Genome-by-ARG presence/absence matrix containing the
   distribution of AMR genes across the analyzed genomes.

b. AMR_gene_mechanisms.tsv: Curated AMR gene annotation table containing gene frequencies and 
associated resistance mechanisms.

Input data requirements:

ARG presence/absence matrix:
- First column: Genome ID.
- Remaining columns: Individual AMR gene names.
- Matrix values: Indicate ARG presence or absence.

Curated mechanism annotation table:
- AMR Gene: AMR gene identifier.
- Number of Genomes: Number of genomes carrying the ARG.
- Resistance Mechanism: Associated AMR mechanism.

Command-line arguments:
--arg-matrix: Path to the ARG presence/absence matrix.

--mechanism-file: Path to the curated ARG and resistance mechanism table.

--output-dir: Directory for saving figures and analytical results.

--top-n: Number of most frequently detected ARGs to visualize. 

--dpi: Resolution of the exported figure. Default example: 600 dpi.

Example execution:
python arg_distribution_analysis.py \
    --arg-matrix data/ARG_presence_absence_matrix.tsv \
    --mechanism-file data/AMR_gene_mechanisms.tsv \
    --output-dir results/figure_2 \
    --top-n 35 \
    --dpi 600

Expected outputs:
- ARG frequency distribution results.
- Ranked list of the most frequently detected ARGs.
- ARG-to-resistance-mechanism annotation results.
- Resistance mechanism distribution summary.
- Figure 2 with two analytical panels.
- Exported publication-resolution figure.
- Processed analytical tables, if implemented.

Reproducibility: The analysis uses the supplied ARG presence/absence matrix and curated resistance mechanism annotations. 
The number of ARGs selected for visualization and figure resolution are controlled through command-line arguments.

Software requirements:
- Python 3.11
- NumPy
- Pandas
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
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Configuration

RANDOM_STATE = 42

TOP_N = 35

MECHANISM_ORDER = [
    "antibiotic efflux",
    "antibiotic target alteration",
    "antibiotic target replacement",
    "antibiotic target protection",
    "antibiotic inactivation",
]

MECHANISM_DISPLAY_NAMES = {
    "antibiotic efflux": "Efflux pumps",
    "antibiotic target alteration": "Target alteration",
    "antibiotic target replacement": "Target replacement",
    "antibiotic target protection": "Target protection",
    "antibiotic inactivation": "Antibiotic inactivation",
}

FIGURE_NAME = "Distribution_of_ARGs_and_AMR_Mechanisms"

BAR_COLOR_A = "#D55E00"
BAR_COLOR_B = "#0072B2"


#Global plot settings

def configure_plot_style() -> None:
    """Set publication-style matplotlib parameters."""

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "Times New Roman",
        "DejaVu Serif",
    ]

    mpl.rcParams["font.size"] = 26
    mpl.rcParams["axes.unicode_minus"] = False

    # Embed editable TrueType fonts in PDF output.
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


#Command-line arguments

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a two-panel figure showing ARG distribution "
            "and AMR resistance mechanisms."
        )
    )

    parser.add_argument(
        "--arg-matrix",
        required=True,
        type=Path,
        help="Path to the ARG presence/absence matrix.",
    )

    parser.add_argument(
        "--mechanism-file",
        required=True,
        type=Path,
        help="Path to the curated AMR mechanism table.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for output figures and analysis files.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help=f"Number of top ARGs to display (default: {TOP_N}).",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output figure resolution (default: 600).",
    )

    return parser.parse_args()


#Input validation

def validate_input_files(
    arg_matrix_file: Path,
    mechanism_file: Path,
) -> None:
    """Check whether input files exist."""

    if not arg_matrix_file.is_file():
        raise FileNotFoundError(
            f"ARG matrix file not found: {arg_matrix_file}"
        )

    if not mechanism_file.is_file():
        raise FileNotFoundError(
            f"Mechanism table not found: {mechanism_file}"
        )


#Load ARG presence/absence matrix

def load_arg_matrix(
    file_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the ARG presence/absence matrix.

    The first column is interpreted as Genome ID.
    All remaining columns are interpreted as ARGs.

    Positive values are converted to 1.
    Zero values are converted to 0.
    """

    raw_df = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False,
    )

    if raw_df.empty:
        raise ValueError(
            "The ARG presence/absence matrix is empty."
        )

    if raw_df.shape[1] < 2:
        raise ValueError(
            "The ARG matrix must contain a Genome ID column "
            "and at least one ARG column."
        )

    genome_id_column = raw_df.columns[0]

    genome_ids = (
        raw_df[genome_id_column]
        .astype("string")
        .str.strip()
    )

    if genome_ids.isna().any() or genome_ids.eq("").any():
        raise ValueError(
            "Missing or empty Genome IDs were detected."
        )

    if genome_ids.duplicated().any():
        raise ValueError(
            "Duplicate Genome IDs were detected in the ARG matrix."
        )

    arg_df = raw_df.iloc[:, 1:].copy()

    ##Remove accidental metadata columns
    invalid_columns = arg_df.columns.astype(str).str.contains(
        r"Genome|ID|Unnamed",
        case=False,
        regex=True,
    )

    removed_columns = arg_df.columns[invalid_columns].tolist()

    if removed_columns:
        print(
            "\nWarning: Removing possible non-gene columns:",
            removed_columns,
        )

        arg_df = arg_df.loc[
            :,
            ~invalid_columns,
        ]

    if arg_df.empty:
        raise ValueError(
            "No ARG columns remain after removing metadata columns."
        )

    if arg_df.columns.duplicated().any():
        raise ValueError(
            "Duplicate ARG column names were detected."
        )

    ##Convert all ARG values to numeric
    arg_df = arg_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    #Missing or non-numeric values are not silently interpreted as gene absence
    if arg_df.isna().any().any():
        missing_count = int(
            arg_df.isna().sum().sum()
        )

        raise ValueError(
            f"The ARG matrix contains {missing_count} missing or "
            "non-numeric values. Please inspect and correct the input."
        )

    values = arg_df.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "The ARG matrix contains infinite values."
        )

    if (values < 0).any():
        raise ValueError(
            "Negative values were detected in the ARG matrix."
        )

    ##Binary presence/absence conversion
    arg_df = pd.DataFrame(
        (values > 0).astype(np.uint8),
        index=genome_ids,
        columns=arg_df.columns,
    )

    ##Calculate the number of genomes carrying each ARG
    gene_frequency = arg_df.sum(
        axis=0
    )

    frequency_df = pd.DataFrame({
        "AMR Gene": gene_frequency.index.astype(str),
        "Number of Genomes": gene_frequency.values,
    })

    frequency_df = frequency_df.sort_values(
        by="Number of Genomes",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    return arg_df, frequency_df


#Select top ARGs

def select_top_args(
    frequency_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Select the most frequently detected ARGs."""

    if top_n < 1:
        raise ValueError(
            "The number of top ARGs must be at least 1."
        )

    top_df = (
        frequency_df
        .head(top_n)
        .copy()
        .iloc[::-1]
        .reset_index(drop=True)
    )

    if top_df.empty:
        raise ValueError(
            "No ARGs are available for plotting."
        )

    return top_df


#Load curated AMR mechanism table

def load_mechanism_table(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load and validate the curated AMR mechanism table.

    Required columns:
        AMR Gene
        Number of Genomes
        Resistance Mechanism
    """

    mech_df = pd.read_csv(
        file_path,
        sep="\t",
    )

    required_columns = {
        "AMR Gene",
        "Number of Genomes",
        "Resistance Mechanism",
    }

    missing_columns = required_columns.difference(
        mech_df.columns
    )

    if missing_columns:
        raise ValueError(
            "The mechanism table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    mech_df = mech_df[
        [
            "AMR Gene",
            "Number of Genomes",
            "Resistance Mechanism",
        ]
    ].copy()

    ##Standardize gene names
    mech_df["AMR Gene"] = (
        mech_df["AMR Gene"]
        .astype("string")
        .str.strip()
    )

    ##Standardize mechanism labels
    mech_df["Resistance Mechanism"] = (
        mech_df["Resistance Mechanism"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    mech_df["Number of Genomes"] = pd.to_numeric(
        mech_df["Number of Genomes"],
        errors="coerce",
    )

    mech_df = mech_df.dropna(
        subset=[
            "AMR Gene",
            "Number of Genomes",
            "Resistance Mechanism",
        ]
    )

    if mech_df.empty:
        raise ValueError(
            "No valid rows remain in the mechanism table."
        )

    if mech_df["AMR Gene"].eq("").any():
        raise ValueError(
            "Empty AMR gene names were detected."
        )

    if mech_df["Resistance Mechanism"].eq("").any():
        raise ValueError(
            "Empty resistance mechanism names were detected."
        )

    if (mech_df["Number of Genomes"] < 0).any():
        raise ValueError(
            "Negative genome counts were detected."
        )

    ##Check for unexpected mechanism categories
    unexpected = sorted(
        set(mech_df["Resistance Mechanism"])
        - set(MECHANISM_ORDER)
    )

    if unexpected:
        print(
            "\nWarning: The following resistance mechanisms are not "
            "included in the figure:",
            unexpected,
        )

    ##Keep only the five specified mechanisms
    mech_df = mech_df[
        mech_df["Resistance Mechanism"].isin(
            MECHANISM_ORDER
        )
    ].copy()

    if mech_df.empty:
        raise ValueError(
            "No rows match the five configured resistance mechanisms."
        )

    return mech_df


#Build resistance mechanism groups

def build_mechanism_groups(
    mech_df: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    """
    Organize genes into resistance mechanism groups.

    Mechanism order follows MECHANISM_ORDER.
    Genes within each group are sorted by genome frequency.
    """

    groups = []

    for mechanism in MECHANISM_ORDER:

        subset = mech_df[
            mech_df["Resistance Mechanism"] == mechanism
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(
            by="Number of Genomes",
            ascending=False,
            kind="stable",
        )

        display_name = MECHANISM_DISPLAY_NAMES[
            mechanism
        ]

        groups.append(
            (display_name, subset)
        )

    return groups


#Check gene assignments

def print_mechanism_assignments(
    mech_df: pd.DataFrame,
) -> None:
    """Print gene assignments for each resistance mechanism."""

    print("\nAMR gene assignments by resistance mechanism:")
    print("=" * 60)

    for mechanism in MECHANISM_ORDER:

        subset = mech_df[
            mech_df["Resistance Mechanism"] == mechanism
        ]

        if subset.empty:
            continue

        genes = subset["AMR Gene"].tolist()

        print(
            f"\n{MECHANISM_DISPLAY_NAMES[mechanism]}:"
        )

        print(
            ", ".join(genes)
        )

    print("=" * 60)


#Panel A: ARG frequency

def plot_top_args(
    ax: plt.Axes,
    top_df: pd.DataFrame,
) -> None:
    """Plot the most frequently detected ARGs."""

    ax.barh(
        top_df["AMR Gene"],
        top_df["Number of Genomes"],
        color=BAR_COLOR_A,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Number of Genomes",
        fontsize=26,
    )

    ax.set_ylabel(
        "AMR Genes",
        fontsize=26,
    )

    ax.set_title(
        "(a) Distribution of dominant ARGs",
        fontsize=28,
        pad=12,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.30,
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        labelsize=24,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


#Panel B: resistance mechanism distribution

def plot_mechanism_distribution(
    ax: plt.Axes,
    groups: list[tuple[str, pd.DataFrame]],
) -> None:
    """Plot ARGs grouped by resistance mechanism."""

    y_positions = []
    gene_labels = []
    values = []

    section_positions = []
    separator_positions = []

    current_y = 0

    ##Prepare gene positions and mechanism group positions

    for title, data in groups:

        start_y = current_y

        for _, row in data.iterrows():

            y_positions.append(
                current_y
            )

            gene_labels.append(
                row["AMR Gene"]
            )

            values.append(
                float(row["Number of Genomes"])
            )

            current_y += 1

        end_y = current_y - 1

        section_positions.append(
            (
                title,
                (start_y + end_y) / 2,
            )
        )

        separator_positions.append(
            current_y - 0.5
        )

        ##Add space between mechanisms
        current_y += 1

    if not values:
        raise ValueError(
            "No resistance mechanism genes are available for plotting."
        )

    max_value = max(values)

    ##Plot horizontal bars

    ax.barh(
        y_positions,
        values,
        color=BAR_COLOR_B,
        edgecolor="black",
        linewidth=0.8,
        height=0.7,
    )

    ###Put the first mechanism at the top.
    ax.invert_yaxis()

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        gene_labels,
        fontsize=24,
    )

    ##Axis labels and title

    ax.set_xlabel(
        "Number of Genomes",
        fontsize=26,
    )

    ax.set_ylabel(
        "",
    )

    ax.set_title(
        "(b) Distribution of AMR resistance mechanisms",
        fontsize=28,
        pad=12,
    )

    ##Mechanism section labels

    label_offset = max_value * 0.15

    for title, ypos in section_positions:

        ax.text(
            -label_offset,
            ypos,
            title,
            fontsize=26,
            ha="right",
            va="center",
            clip_on=False,
        )

    ##Separator lines

    for ypos in separator_positions[:-1]:

        ax.axhline(
            ypos,
            color="black",
            linewidth=0.8,
            alpha=0.8,
        )

    #Formatting

    ax.set_xlim(
        0,
        max_value * 1.15 if max_value > 0 else 1,
    )

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.tick_params(
        axis="x",
        labelsize=24,
    )


#Generate figure

def generate_figure(
    top_df: pd.DataFrame,
    groups: list[tuple[str, pd.DataFrame]],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Generate and save the two-panel ARG distribution figure."""

    configure_plot_style()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(26, 22),
        gridspec_kw={
            "width_ratios": [
                1,
                1.5,
            ]
        },
    )

    ##Panel A
    plot_top_args(
        axes[0],
        top_df,
    )

    ##Panel B
    plot_mechanism_distribution(
        axes[1],
        groups,
    )

    ##Layout

    plt.tight_layout()

    ##Save figures

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


#Save analysis tables

def save_analysis_tables(
    output_dir: Path,
    frequency_df: pd.DataFrame,
    top_df: pd.DataFrame,
    mech_df: pd.DataFrame,
) -> None:
    """Save ARG frequencies and mechanism assignments."""

    frequency_df.to_csv(
        output_dir / "ARG_genome_frequency.tsv",
        sep="\t",
        index=False,
    )

    top_df.to_csv(
        output_dir / "Top_ARGs.tsv",
        sep="\t",
        index=False,
    )

    mech_df.to_csv(
        output_dir / "AMR_gene_mechanism_assignments.tsv",
        sep="\t",
        index=False,
    )


#Save run metadata

def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    n_genomes: int,
    n_args: int,
    n_mechanism_genes: int,
) -> None:
    """Save analysis parameters and input information."""

    metadata = {
        "analysis": "ARG distribution and AMR mechanisms",
        "date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "arg_matrix": str(
            args.arg_matrix.resolve()
        ),
        "mechanism_file": str(
            args.mechanism_file.resolve()
        ),
        "number_of_genomes": int(n_genomes),
        "number_of_ARGs": int(n_args),
        "number_of_mechanism_table_rows": int(
            n_mechanism_genes
        ),
        "top_n_ARGs_displayed": int(args.top_n),
        "mechanisms_included": MECHANISM_ORDER,
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
    """Run the complete ARG distribution analysis."""

    args = parse_arguments()

    validate_input_files(
        args.arg_matrix,
        args.mechanism_file,
    )

    if args.top_n < 1:
        raise ValueError(
            "--top-n must be at least 1."
        )

    if args.dpi < 100:
        raise ValueError(
            "--dpi must be at least 100."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ##Load ARG matrix

    print("\nLoading ARG presence/absence matrix...")

    arg_matrix, frequency_df = load_arg_matrix(
        args.arg_matrix
    )

    n_genomes, n_args = arg_matrix.shape

    print(f"Number of genomes: {n_genomes:,}")
    print(f"Number of ARGs: {n_args:,}")

    ##Select top ARGs

    top_df = select_top_args(
        frequency_df,
        args.top_n,
    )

    print(
        f"\nSelected top {len(top_df)} ARGs for Panel A."
    )

    ##Load mechanism table

    print("\nLoading AMR resistance mechanism table...")

    mech_df = load_mechanism_table(
        args.mechanism_file
    )

    groups = build_mechanism_groups(
        mech_df
    )

    print_mechanism_assignments(
        mech_df
    )

    ##Save numerical results

    print("\nSaving analysis tables...")

    save_analysis_tables(
        output_dir=args.output_dir,
        frequency_df=frequency_df,
        top_df=top_df,
        mech_df=mech_df,
    )

    ##Generate figure

    print("\nGenerating Figure 2...")

    png_file, pdf_file = generate_figure(
        top_df=top_df,
        groups=groups,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    ##Save metadata

    save_run_metadata(
        output_dir=args.output_dir,
        args=args,
        n_genomes=n_genomes,
        n_args=n_args,
        n_mechanism_genes=len(mech_df),
    )

    ##Completion message

    print("\nAnalysis completed successfully.")

    print("\nOutput files:")
    print(f"PNG figure: {png_file}")
    print(f"PDF figure: {pdf_file}")

    print(
        "ARG frequency table: "
        f"{args.output_dir / 'ARG_genome_frequency.tsv'}"
    )

    print(
        "Top ARG table: "
        f"{args.output_dir / 'Top_ARGs.tsv'}"
    )

    print(
        "Mechanism assignments: "
        f"{args.output_dir / 'AMR_gene_mechanism_assignments.tsv'}"
    )

    print(
        "Run metadata: "
        f"{args.output_dir / 'run_metadata.json'}"
    )


#Entry point

if __name__ == "__main__":
    main()

