#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Clustered heatmap of AMR gene profiles with MRSA and MSSA phenotype annotation

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description: This script performs hierarchical clustering and visualization of AMR gene profiles across bacterial genomes, 
incorporating methicillin-resistant Staphylococcus aureus (MRSA) and methicillin-susceptible Staphylococcus aureus (MSSA)
phenotype annotations.

The analysis integrates a binary ARG presence/absence matrix with genome-level phenotype information to investigate patterns
of AMR gene distribution, genomic similarity, and clustering relationships among the analyzed isolates.

Objectives:
a. Characterize the distribution of AMR genes across the analyzed bacterial genomes.
b. Integrate ARG presence/absence profiles with corresponding MRSA and MSSA phenotype annotations.
c. Match genome identifiers between the ARG matrix and phenotype metadata.
d. Validate the consistency of genome identifiers, binary ARG profiles, and phenotype classifications.
e. Perform hierarchical clustering of genomes based on their ARG presence/absence profiles.
f. Perform hierarchical clustering of ARGs to identify co-occurrence and distribution patterns.
g. Generate a clustered heatmap illustrating ARG profiles across genomes with MRSA/MSSA phenotype annotations.
h. Export the resulting clustered heatmap and corresponding genome and ARG clustering orders.

Analytical workflow:
a. Load the ARG presence/absence matrix.
b. Load the genome phenotype annotation table.
c. Identify and match genome identifiers shared between the two datasets.
d. Validate the binary ARG matrix and phenotype annotations.
e. Extract the ARG profiles and corresponding phenotype labels.
f. Calculate pairwise distances between genomes and ARGs using the specified distance metric.
g. Perform hierarchical clustering using the selected linkage method.
h. Generate a clustered heatmap with genome-level phenotype annotations distinguishing MRSA and MSSA isolates.
i. Export publication-resolution PDF and PNG figures.
j. Save the clustered genome ordering and ARG ordering for downstream interpretation and reproducibility.

Input files:

a. ARG_presence_absence_matrix.tsv: Genome-by-ARG binary presence/absence matrix containing AMR gene profiles across 
the analyzed bacterial genomes.

b. data_summary_combined.tsv: Genome metadata table containing genome identifiers, organism information, and 
phenotype-related annotations.

Input data requirements:

ARG presence/absence matrix:
- First column: Genome ID.
- Remaining columns: Individual antimicrobial resistance
  gene identifiers.
- Matrix values: Binary presence/absence indicators.
- Genome identifiers must be unique.

Phenotype annotation table:
- Genome ID: Identifier corresponding to the genome in
  the ARG presence/absence matrix.
- Organism Qualifier: Organism or phenotype annotation
  used to identify the relevant MRSA and MSSA categories.

Phenotype classification:
- MRSA: Methicillin-resistant Staphylococcus aureus.
- MSSA: Methicillin-susceptible Staphylococcus aureus.

Command-line arguments:
--amr-file: Path to the ARG presence/absence matrix.

--phenotype-file: Path to the genome phenotype annotation table.

--output-dir: Directory for saving clustered heatmap figures & associated analytical results.

--method: Hierarchical clustering linkage method. E.g: average.

--metric: Distance metric used for hierarchical clustering. E.g: euclidean.

--dpi: Resolution of exported figures. E.g: 300 dpi.

Example execution:
python clustered_arg_heatmap.py \
    --amr-file data/ARG_presence_absence_matrix.tsv \
    --phenotype-file data/data_summary_combined.tsv \
    --output-dir results/clustered_heatmap \
    --method average \
    --metric euclidean \
    --dpi 300

Expected outputs:
- Validated ARG presence/absence matrix.
- Matched genome phenotype annotations.
- Genome hierarchical clustering results.
- ARG hierarchical clustering results.
- Clustered heatmap with MRSA/MSSA phenotype annotations.
- Publication-resolution PDF figure.
- Publication-resolution PNG figure.
- Clustered genome ordering.
- Clustered ARG ordering.

Reproducibility: The hierarchical clustering procedure is controlled by the specified linkage method and 
distance metric. Input datasets, clustering parameters, and phenotype classification criteria should be maintained and 
documented to facilitate reproducibility.

Software requirements:
- Python 3.11
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn (if used for clustering or distance calculations)

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
import seaborn as sns

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


#Configuration

PHENOTYPE_COLUMN = "Organism Qualifier"

PHENOTYPE_COLORS = {
    "MRSA":"#AC3020",
    "MSSA": "#4DBBD5",
}

ABSENCE_COLOR = "#FFFFFF"
PRESENCE_COLOR = "#2C7FB8"

FIGURE_NAME = (
    "Clustered_Heatmap_ARG_Profiles_MRSA_vs_MSSA"
)


#Font & plot settings

def configure_plot_style() -> None:
    """Configure global matplotlib publication-style settings."""

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [
        "Times New Roman",
        "DejaVu Serif",
    ]

    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["axes.unicode_minus"] = False

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    sns.set_theme(
        style="white",
        font="serif",
    )


#Command-line arguments

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a clustered ARG presence/absence heatmap "
            "with MRSA/MSSA phenotype annotation."
        )
    )

    parser.add_argument(
        "--amr-file",
        required=True,
        type=Path,
        help="Path to the ARG presence/absence matrix.",
    )

    parser.add_argument(
        "--phenotype-file",
        required=True,
        type=Path,
        help="Path to the genome phenotype summary table.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for saving figures and result files.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="average",
        choices=[
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        ],
        help="Hierarchical clustering linkage method.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric used for hierarchical clustering.",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=30,
        help="Figure width in inches (default: 30).",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=24,
        help="Figure height in inches (default: 24).",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure resolution (default: 300).",
    )

    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help=(
            "Do not stop when genomes are missing from either input. "
            "Only genomes present in both files will be plotted."
        ),
    )

    return parser.parse_args()


#Input file validation

def validate_input_files(
    amr_file: Path,
    phenotype_file: Path,
) -> None:
    """Check that the input files exist."""

    if not amr_file.is_file():
        raise FileNotFoundError(
            f"ARG matrix not found: {amr_file}"
        )

    if not phenotype_file.is_file():
        raise FileNotFoundError(
            f"Phenotype file not found: {phenotype_file}"
        )


#Load ARG matrix

def load_arg_matrix(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load the ARG presence/absence matrix.

    Genome ID is used as the row index.
    Remaining columns represent ARGs.

    Values greater than zero are treated as presence.
    Values equal to zero are treated as absence.
    """

    df = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            "The ARG presence/absence matrix is empty."
        )

    if "Genome ID" not in df.columns:
        raise ValueError(
            "The ARG matrix must contain a 'Genome ID' column."
        )

    genome_ids = (
        df["Genome ID"]
        .astype("string")
        .str.strip()
    )

    if genome_ids.isna().any() or genome_ids.eq("").any():
        raise ValueError(
            "Missing or empty Genome IDs were detected in the ARG matrix."
        )

    if genome_ids.duplicated().any():
        duplicate_ids = genome_ids[
            genome_ids.duplicated()
        ].unique()

        raise ValueError(
            "Duplicate Genome IDs detected in the ARG matrix: "
            f"{duplicate_ids[:10].tolist()}"
        )

    df = df.drop(
        columns=["Genome ID"]
    )

    if df.empty:
        raise ValueError(
            "The ARG matrix does not contain any gene columns."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate ARG column names were detected."
        )

    df.index = genome_ids
    df.index.name = "Genome ID"

    ##Convert ARG values to numeric
    df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df.isna().any().any():
        missing_count = int(
            df.isna().sum().sum()
        )

        raise ValueError(
            f"The ARG matrix contains {missing_count} missing or "
            "non-numeric values. Please inspect the input."
        )

    values = df.to_numpy(
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

    ##Convert to binary presence/absence
    df = pd.DataFrame(
        (values > 0).astype(np.uint8),
        index=df.index,
        columns=df.columns,
    )

    ##Remove ARGs absent from all genomes
    zero_columns = df.columns[
        df.sum(axis=0) == 0
    ].tolist()

    if zero_columns:
        print(
            f"\nRemoving {len(zero_columns)} ARGs absent "
            "from all genomes."
        )

        df = df.drop(
            columns=zero_columns
        )

    ##Remove genomes without any detected ARGs
    zero_rows = df.index[
        df.sum(axis=1) == 0
    ].tolist()

    if zero_rows:
        print(
            f"Warning: {len(zero_rows)} genomes have no detected ARGs."
        )

        print(
            "These genomes will be retained in the analysis."
        )

    if df.empty or df.shape[1] == 0:
        raise ValueError(
            "No usable ARG data remain after validation."
        )

    return df


#Load phenotype data

def load_phenotype_data(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load genome phenotype information.

    Required columns:
        Genome ID
        Organism Qualifier
    """

    df = pd.read_csv(
        file_path,
        sep="\t",
    )

    required_columns = {
        "Genome ID",
        PHENOTYPE_COLUMN,
    }

    missing_columns = required_columns.difference(
        df.columns
    )

    if missing_columns:
        raise ValueError(
            "Phenotype file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[
        [
            "Genome ID",
            PHENOTYPE_COLUMN,
        ]
    ].copy()

    df["Genome ID"] = (
        df["Genome ID"]
        .astype("string")
        .str.strip()
    )

    df[PHENOTYPE_COLUMN] = (
        df[PHENOTYPE_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if df["Genome ID"].isna().any() or df["Genome ID"].eq("").any():
        raise ValueError(
            "Missing or empty Genome IDs were detected in the phenotype file."
        )

    if df["Genome ID"].duplicated().any():
        duplicate_ids = df.loc[
            df["Genome ID"].duplicated(),
            "Genome ID",
        ].unique()

        raise ValueError(
            "Duplicate Genome IDs detected in the phenotype file: "
            f"{duplicate_ids[:10].tolist()}"
        )

    df = df.set_index(
        "Genome ID"
    )

    return df


#Match genomes between datasets

def match_genomes(
    amr_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    keep_unmatched: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Match genomes between the ARG matrix and phenotype table.

    The final analysis includes only genomes found in both files.
    The original ARG matrix order is preserved.
    """

    amr_ids = set(
        amr_df.index
    )

    phenotype_ids = set(
        phenotype_df.index
    )

    missing_phenotypes = sorted(
        amr_ids - phenotype_ids
    )

    missing_arg_profiles = sorted(
        phenotype_ids - amr_ids
    )

    print("\nGenome matching:")
    print(f"Genomes in ARG matrix: {len(amr_ids):,}")
    print(f"Genomes in phenotype table: {len(phenotype_ids):,}")
    print(
        f"Genomes without phenotype records: "
        f"{len(missing_phenotypes):,}"
    )
    print(
        f"Phenotype records without ARG profiles: "
        f"{len(missing_arg_profiles):,}"
    )

    if not keep_unmatched and (
        missing_phenotypes or missing_arg_profiles
    ):
        raise ValueError(
            "The two input files do not contain identical genome sets. "
            "Review the genome matching report. Use --keep-unmatched "
            "to proceed with the shared genomes only."
        )

    shared_ids = [
        genome_id
        for genome_id in amr_df.index
        if genome_id in phenotype_ids
    ]

    if not shared_ids:
        raise ValueError(
            "No matching genomes were found between the two files."
        )

    amr_matched = amr_df.loc[
        shared_ids
    ].copy()

    phenotype_matched = phenotype_df.loc[
        shared_ids,
        PHENOTYPE_COLUMN,
    ].copy()

    ##Retain only the two explicitly defined phenotypes
    valid_phenotypes = set(
        PHENOTYPE_COLORS
    )

    invalid_phenotypes = sorted(
        set(phenotype_matched.dropna())
        - valid_phenotypes
    )

    if invalid_phenotypes:
        raise ValueError(
            "Unexpected phenotype labels were detected: "
            f"{invalid_phenotypes}. Expected only MRSA and MSSA."
        )

    if phenotype_matched.isna().any():
        missing_count = int(
            phenotype_matched.isna().sum()
        )

        raise ValueError(
            f"{missing_count} matched genomes have missing phenotype "
            "annotations. Please correct the phenotype table."
        )

    if amr_matched.shape[0] < 2:
        raise ValueError(
            "At least two matched genomes are required for clustering."
        )

    if amr_matched.shape[1] < 2:
        raise ValueError(
            "At least two ARGs are required for row and column clustering."
        )

    return amr_matched, phenotype_matched


#Phenotype annotation colors

def create_phenotype_colors(
    phenotype_series: pd.Series,
) -> pd.Series:
    """Map MRSA and MSSA labels to annotation colors."""

    colors = phenotype_series.map(
        PHENOTYPE_COLORS
    )

    colors.name = None

    if colors.isna().any():
        raise ValueError(
            "One or more phenotype labels could not be mapped to colors."
        )

    return colors


#Save clustering orders

def save_clustering_orders(
    cluster_grid,
    amr_df: pd.DataFrame,
    phenotype_series: pd.Series,
    output_dir: Path,
) -> None:
    """
    Export genome and ARG orders generated by hierarchical clustering.

    The heatmap dendrogram determines the reordered row and column indices.
    """

    row_order = cluster_grid.dendrogram_row.reordered_ind
    column_order = cluster_grid.dendrogram_col.reordered_ind

    ordered_genomes = amr_df.index[
        row_order
    ]

    ordered_args = amr_df.columns[
        column_order
    ]

    genome_order_df = pd.DataFrame({
        "Clustering_Order": np.arange(
            1,
            len(ordered_genomes) + 1,
        ),
        "Genome_ID": ordered_genomes.astype(str),
        "Organism_Qualifier": phenotype_series.loc[
            ordered_genomes
        ].values,
    })

    arg_order_df = pd.DataFrame({
        "Clustering_Order": np.arange(
            1,
            len(ordered_args) + 1,
        ),
        "AMR_Gene": ordered_args.astype(str),
    })

    genome_order_df.to_csv(
        output_dir / "clustered_genome_order.tsv",
        sep="\t",
        index=False,
    )

    arg_order_df.to_csv(
        output_dir / "clustered_ARG_order.tsv",
        sep="\t",
        index=False,
    )


#Save data summary

def save_data_summary(
    amr_df: pd.DataFrame,
    phenotype_series: pd.Series,
    output_dir: Path,
) -> None:
    """Save phenotype and ARG frequency summaries."""

    phenotype_summary = (
        phenotype_series
        .value_counts()
        .rename_axis("Organism_Qualifier")
        .reset_index(name="Number_of_Genomes")
    )

    phenotype_summary.to_csv(
        output_dir / "phenotype_distribution.tsv",
        sep="\t",
        index=False,
    )

    arg_frequency = amr_df.sum(
        axis=0
    ).sort_values(
        ascending=False
    )

    arg_frequency_df = pd.DataFrame({
        "AMR_Gene": arg_frequency.index.astype(str),
        "Number_of_Genomes": arg_frequency.values,
        "Percentage_of_Genomes": (
            arg_frequency.values / len(amr_df)
        ) * 100,
    })

    arg_frequency_df.to_csv(
        output_dir / "ARG_frequency_summary.tsv",
        sep="\t",
        index=False,
    )


#Save run metadata

def save_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    amr_df: pd.DataFrame,
    phenotype_series: pd.Series,
) -> None:
    """Save analysis parameters and dataset information."""

    metadata = {
        "analysis": (
            "Clustered ARG presence/absence heatmap "
            "with MRSA/MSSA annotation"
        ),
        "date_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "arg_matrix": str(
            args.amr_file.resolve()
        ),
        "phenotype_file": str(
            args.phenotype_file.resolve()
        ),
        "number_of_genomes": int(
            amr_df.shape[0]
        ),
        "number_of_ARGs": int(
            amr_df.shape[1]
        ),
        "phenotype_distribution": {
            str(key): int(value)
            for key, value in phenotype_series.value_counts().items()
        },
        "clustering_method": args.method,
        "distance_metric": args.metric,
        "row_clustering": True,
        "column_clustering": True,
        "figure_width": args.width,
        "figure_height": args.height,
        "figure_dpi": args.dpi,
        "absence_color": ABSENCE_COLOR,
        "presence_color": PRESENCE_COLOR,
        "phenotype_colors": PHENOTYPE_COLORS,
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


#Generate clustered heatmap

def generate_clustered_heatmap(
    amr_df: pd.DataFrame,
    phenotype_series: pd.Series,
    output_dir: Path,
    method: str,
    metric: str,
    width: float,
    height: float,
    dpi: int,
):
    """Generate and save the publication-quality clustered heatmap."""

    configure_plot_style()

    ##Annotation colors

    organism_qualifier_colors = create_phenotype_colors(
        phenotype_series
    )

    binary_cmap = ListedColormap([
        ABSENCE_COLOR,
        PRESENCE_COLOR,
    ])

    ##Cluster map

    cluster_grid = sns.clustermap(
        amr_df,
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        row_colors=organism_qualifier_colors,
        figsize=(width, height),
        xticklabels=True,
        yticklabels=False,
        metric=metric,
        method=method,
        linewidths=0,
        dendrogram_ratio=(0.10, 0.08),
        colors_ratio=(0.03, 0.03),
        cbar_pos=None,
        row_cluster=True,
        col_cluster=True,
    )

    ##Row annotation label

    cluster_grid.ax_row_colors.set_xlabel(
        "Organism Qualifier",
        fontsize=36,
        fontname="Times New Roman",
        rotation=90,
        labelpad=10,
    )

    ##Dendrogram styling

    for ax in [
        cluster_grid.ax_row_dendrogram,
        cluster_grid.ax_col_dendrogram,
    ]:

        for line in ax.collections:
            line.set_linewidth(1.5)
            line.set_color("#333333")

    ##Axis fonts

    for ax in [
        cluster_grid.ax_heatmap,
        cluster_grid.ax_row_dendrogram,
        cluster_grid.ax_col_dendrogram,
    ]:

        for label in ax.get_xticklabels():
            label.set_fontname("Times New Roman")

        for label in ax.get_yticklabels():
            label.set_fontname("Times New Roman")

    ##Heatmap axis labels

    cluster_grid.ax_heatmap.set_xlabel(
        "AMR Genes",
        fontsize=36,
        fontname="Times New Roman",
        labelpad=8,
    )

    cluster_grid.ax_heatmap.set_ylabel(
        "Genomes",
        fontsize=36,
        fontname="Times New Roman",
        labelpad=10,
    )

    ##X-axis labels

    plt.setp(
        cluster_grid.ax_heatmap.get_xticklabels(),
        rotation=90,
        ha="center",
        fontsize=32,
        fontname="Times New Roman",
    )

    cluster_grid.ax_heatmap.tick_params(
        axis="x",
        length=0,
    )

    ##Clean heatmap border

    for spine in cluster_grid.ax_heatmap.spines.values():
        spine.set_visible(False)

    ##MRSA/MSSA legend

    legend_elements = [
        Patch(
            facecolor=PHENOTYPE_COLORS["MRSA"],
            label="MRSA",
        ),
        Patch(
            facecolor=PHENOTYPE_COLORS["MSSA"],
            label="MSSA",
        ),
    ]

    legend = cluster_grid.fig.legend(
        handles=legend_elements,
        title="Organism Qualifier",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
    )

    legend.get_title().set_fontsize(32)
    legend.get_title().set_fontname("Times New Roman")

    for text in legend.get_texts():
        text.set_fontsize(34)
        text.set_fontname("Times New Roman")

    ##Figure spacing

    cluster_grid.fig.subplots_adjust(
        top=0.92,
        bottom=0.18,
        left=0.06,
        right=0.98,
    )

    ##Save outputs

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_output = output_dir / f"{FIGURE_NAME}.pdf"
    png_output = output_dir / f"{FIGURE_NAME}.png"

    cluster_grid.fig.savefig(
        pdf_output,
        bbox_inches="tight",
        dpi=dpi,
    )

    cluster_grid.fig.savefig(
        png_output,
        bbox_inches="tight",
        dpi=dpi,
    )

    return cluster_grid, pdf_output, png_output


#Main workflow

def main() -> None:
    """Execute the complete clustered ARG heatmap workflow."""

    args = parse_arguments()

    validate_input_files(
        args.amr_file,
        args.phenotype_file,
    )

    if args.width <= 0 or args.height <= 0:
        raise ValueError(
            "Figure width and height must be positive."
        )

    if args.dpi < 100:
        raise ValueError(
            "Figure DPI must be at least 100."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ##Load datasets

    print("\nLoading ARG presence/absence matrix...")

    amr_df = load_arg_matrix(
        args.amr_file
    )

    print("Loading phenotype data...")

    phenotype_df = load_phenotype_data(
        args.phenotype_file
    )

    ##Match genomes

    amr_df, phenotype_series = match_genomes(
        amr_df=amr_df,
        phenotype_df=phenotype_df,
        keep_unmatched=args.keep_unmatched,
    )

    ##Print dataset information

    print("\nFinal dataset:")
    print(f"Number of genomes: {amr_df.shape[0]:,}")
    print(f"Number of ARGs: {amr_df.shape[1]:,}")

    print("\nPhenotype distribution:")
    print(
        phenotype_series.value_counts().to_string()
    )

    ##Save summaries

    save_data_summary(
        amr_df=amr_df,
        phenotype_series=phenotype_series,
        output_dir=args.output_dir,
    )

    ##Generate heatmap

    print("\nGenerating clustered ARG heatmap...")

    cluster_grid, pdf_output, png_output = (
        generate_clustered_heatmap(
            amr_df=amr_df,
            phenotype_series=phenotype_series,
            output_dir=args.output_dir,
            method=args.method,
            metric=args.metric,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
    )

    ##Export clustering orders

    save_clustering_orders(
        cluster_grid=cluster_grid,
        amr_df=amr_df,
        phenotype_series=phenotype_series,
        output_dir=args.output_dir,
    )

    ##Save metadata

    save_run_metadata(
        output_dir=args.output_dir,
        args=args,
        amr_df=amr_df,
        phenotype_series=phenotype_series,
    )

    ##Close figure

    plt.close(
        cluster_grid.fig
    )

    ##Completion message

    print("\nAnalysis completed successfully.")

    print("\nSaved files:")
    print(f"PDF figure: {pdf_output}")
    print(f"PNG figure: {png_output}")

    print(
        "Clustered genome order: "
        f"{args.output_dir / 'clustered_genome_order.tsv'}"
    )

    print(
        "Clustered ARG order: "
        f"{args.output_dir / 'clustered_ARG_order.tsv'}"
    )

    print(
        "ARG frequency summary: "
        f"{args.output_dir / 'ARG_frequency_summary.tsv'}"
    )

    print(
        "Phenotype distribution: "
        f"{args.output_dir / 'phenotype_distribution.tsv'}"
    )

    print(
        "Run metadata: "
        f"{args.output_dir / 'run_metadata.json'}"
    )


#Entry point

if __name__ == "__main__":
    main()

