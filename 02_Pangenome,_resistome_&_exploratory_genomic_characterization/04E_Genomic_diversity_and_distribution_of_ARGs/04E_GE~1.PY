#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Pan-genome & AMR gene landscape analysis

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description: This script performs integrated exploratory genomic analysis of pangenome gene-family profiles and AMR
gene distributions across the analyzed bacterial genomes.

The analysis combines Panaroo-derived gene presence/absence profiles with ARG presence/absence data to investigate
genomic variation, gene-family distribution, AMR gene burden, and the overall resistome landscape.

Principal component analysis (PCA) and t-distributed stochastic neighbor embedding (t-SNE) are used to explore
patterns of genomic similarity and variation in pangenome and ARG profiles.

Objectives:
a. Characterize genomic variation using pangenome gene-family presence/absence profiles.
b. Explore similarities and differences among genomes through PCA and t-SNE dimensionality reduction.
c. Investigate the distribution of AMR gene profiles using PCA and t-SNE.
d. Quantify the AMR gene burden across individual genomes.
e. Determine the frequency and distribution of individual ARGs across the analyzed genome collection.
f. Integrate pangenome and resistome information to provide a comparative overview of genomic and AMR characteristics.
g. Generate a four-panel figure summarizing the pan-genome and ARG landscape.
h. Export dimensionality reduction coordinates, statistical summaries, genome matching information, and 
visualization outputs.

Analytical workflow:

a. Load the Panaroo gene presence/absence matrix.
b. Load the ARG presence/absence matrix.
c. Validate genome identifiers and binary presence/absence profiles.
d. Match genome identifiers between the pangenome and ARG datasets.
e. Extract matched pangenome and ARG feature matrices.
f. Perform PCA on the pangenome gene-family profiles.
g. Apply t-SNE to the pangenome representation.
h. Perform PCA on ARG presence/absence profiles.
i. Apply t-SNE to the ARG representation.
j. Calculate ARG burden for each genome based on the number of detected resistance genes.
k. Calculate the frequency of individual ARGs across the analyzed genomes.
l. Generate a 2 x 2 visualization comprising:
    - Panel A: Pangenome PCA and t-SNE analysis.
    - Panel B: ARG-profile PCA and t-SNE analysis.
    - Panel C: ARG burden distribution across genomes.
    - Panel D: Frequency distribution of ARGs across genomes.
m. Export dimensionality reduction coordinates, PCA explained variance summaries, ARG burden tables,
    ARG frequency tables, and genome matching reports.
n. Save publication-resolution PNG and TIFF figures and the analysis summary.

Input files:

a. Panaroo gene presence/absence matrix: Genome-by-gene-family matrix generated from Panaroo pan-genome analysis.

b. ARG presence/absence matrix: Genome-by-ARG binary matrix generated from the RGI-CARD antimicrobial resistance analysis.

Input data requirements:

Panaroo gene presence/absence matrix:
- Genome identifiers must be available for each row.
- Remaining columns must represent individual gene families.
- Matrix values must indicate gene presence or absence.

ARG presence/absence matrix:
- Genome identifiers must correspond to those in the pan-genome matrix.
- Remaining columns must represent individual ARGs.
- Matrix values must indicate gene presence or absence.

Genome matching:
- Genome identifiers should be consistent between datasets.
- Only appropriately matched genomes should be included in integrated analyses.
- Unmatched genome identifiers should be documented in the genome matching report.

Analytical methods:
- PCA
- t-SNE
- Gene-family presence/absence profiling
- ARG burden quantification
- ARG frequency analysis

Expected outputs:
- Pan-genome PCA coordinates.
- Pan-genome t-SNE coordinates.
- ARG-profile PCA coordinates.
- ARG-profile t-SNE coordinates.
- PCA explained variance summaries.
- Per-genome ARG burden table.
- ARG frequency distribution table.
- Genome matching report.
- Four-panel pangenome and ARG landscape figure.
- Publication-resolution PNG and TIFF figures.
- Analysis summary and relevant execution metadata.

Reproducibility: The dimensionality reduction procedures, including PCA and t-SNE, should use documented parameters and 
random seeds where applicable. Genome matching criteria, preprocessing procedures, and analytical settings should be 
maintained to facilitate reproducibility.

Software requirements:
- Python 3.11
- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn

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

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


#Global settings

BASE_FONT_SIZE = 12

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],

    "axes.labelsize": BASE_FONT_SIZE + 8,
    "axes.titlesize": BASE_FONT_SIZE + 10,
    "xtick.labelsize": BASE_FONT_SIZE + 8,
    "ytick.labelsize": BASE_FONT_SIZE + 8,
    "legend.fontsize": BASE_FONT_SIZE + 8,
    "figure.titlesize": BASE_FONT_SIZE + 12,

    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,

    "savefig.dpi": 300,
    "figure.dpi": 150,
    "savefig.bbox": "tight"
})


#Default analysis parameters

DEFAULT_PCA_COMPONENTS = 50
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_RANDOM_STATE = 42

TSNE_COMPONENTS = 2

POINT_SIZE = 30
POINT_ALPHA = 0.75

ARG_BURDEN_BINS = 20
ARG_FREQUENCY_BINS = 30

FIG_WIDTH = 14
FIG_HEIGHT = 12


#Logging

def setup_logging(output_dir):
    """Configure logging to both terminal and file."""

    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "analysis.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8")
        ],
        force=True
    )


#Command-line arguments

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Perform pan-genome and ARG-profile PCA-tSNE analysis "
            "and generate publication-quality figures."
        )
    )

    parser.add_argument(
        "--pan-genome",
        required=True,
        help="Panaroo gene presence/absence TSV file."
    )

    parser.add_argument(
        "--arg-matrix",
        required=True,
        help="ARG presence/absence TSV file."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for saving analysis results."
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="Maximum number of PCA components. Default: 50."
    )

    parser.add_argument(
        "--perplexity",
        type=float,
        default=DEFAULT_TSNE_PERPLEXITY,
        help="t-SNE perplexity. Default: 30."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed. Default: 42."
    )

    parser.add_argument(
        "--width",
        type=float,
        default=FIG_WIDTH,
        help="Figure width in inches."
    )

    parser.add_argument(
        "--height",
        type=float,
        default=FIG_HEIGHT,
        help="Figure height in inches."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure resolution. Default: 600."
    )

    parser.add_argument(
        "--min-genomes",
        type=int,
        default=4,
        help="Minimum number of matched genomes required."
    )

    return parser.parse_args()


#Input validation

def validate_input_file(file_path, description):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"{description} is not a file: {file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"{description} is empty: {file_path}"
        )

    return file_path


#Load presence/absence matrices

def load_matrix(file_path, dataset_name):

    logging.info("Loading %s matrix: %s", dataset_name, file_path)

    df = pd.read_csv(
        file_path,
        sep="\t",
        index_col=0
    )

    if df.empty:
        raise ValueError(
            f"{dataset_name} matrix contains no data."
        )

    ##Standardise genome identifiers.
    df.index = df.index.astype(str).str.strip()

    if (df.index == "").any():
        raise ValueError(
            f"{dataset_name} matrix contains empty genome identifiers."
        )

    if df.index.duplicated().any():
        duplicates = df.index[df.index.duplicated()].unique().tolist()

        raise ValueError(
            f"Duplicate genome identifiers in {dataset_name}: "
            f"{duplicates[:10]}"
        )

    if df.columns.duplicated().any():
        raise ValueError(
            f"{dataset_name} matrix contains duplicate feature names."
        )

    ##Convert feature values to numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    ##Missing values are treated as absence.
    df = df.fillna(0)

    ##Convert to binary presence/absence.
    df = (df > 0).astype(np.uint8)

    ##Remove features with no variation.
    variable_features = df.nunique(axis=0) > 1

    removed = int((~variable_features).sum())

    df = df.loc[:, variable_features]

    logging.info(
        "%s: %d genomes, %d variable features; "
        "%d constant features removed.",
        dataset_name,
        df.shape[0],
        df.shape[1],
        removed
    )

    if df.shape[1] == 0:
        raise ValueError(
            f"{dataset_name} matrix has no variable features."
        )

    return df


#Match genomes between datasets

def match_genomes(panaroo, rgi, output_dir, min_genomes):

    panaroo_ids = set(panaroo.index)
    rgi_ids = set(rgi.index)

    shared_ids = sorted(panaroo_ids.intersection(rgi_ids))

    panaroo_only = sorted(panaroo_ids - rgi_ids)
    rgi_only = sorted(rgi_ids - panaroo_ids)

    logging.info("Genome matching summary:")
    logging.info("Pan-genome genomes: %d", len(panaroo_ids))
    logging.info("ARG genomes: %d", len(rgi_ids))
    logging.info("Shared genomes: %d", len(shared_ids))
    logging.info("Pan-genome only: %d", len(panaroo_only))
    logging.info("ARG only: %d", len(rgi_only))

    if len(shared_ids) < min_genomes:
        raise ValueError(
            f"Only {len(shared_ids)} shared genomes were found. "
            f"At least {min_genomes} are required."
        )

    ##Preserve exactly the same genome order in both matrices.
    panaroo = panaroo.loc[shared_ids].copy()
    rgi = rgi.loc[shared_ids].copy()

    if not panaroo.index.equals(rgi.index):
        raise RuntimeError(
            "Genome matching failed. Matrix indices are not identical."
        )

    ##Save genome matching report.
    matching_report = pd.DataFrame({
        "Genome": sorted(panaroo_ids.union(rgi_ids)),
    })

    matching_report["In_Pan_genome"] = (
        matching_report["Genome"].isin(panaroo_ids)
    )

    matching_report["In_ARG_matrix"] = (
        matching_report["Genome"].isin(rgi_ids)
    )

    matching_report["Included_in_analysis"] = (
        matching_report["In_Pan_genome"]
        & matching_report["In_ARG_matrix"]
    )

    matching_report.to_csv(
        output_dir / "Genome_matching_report.tsv",
        sep="\t",
        index=False
    )

    return panaroo, rgi


#PCA + t-SNE analysis

def pca_tsne_embedding(
    df,
    dataset_name,
    pca_components,
    perplexity,
    random_state
):

    X = df.to_numpy(dtype=np.float64)

    n_samples, n_features = X.shape

    if n_samples < 4:
        raise ValueError(
            f"{dataset_name}: at least four genomes are required."
        )

    if n_features < 2:
        raise ValueError(
            f"{dataset_name}: at least two variable features are required."
        )

    ##PCA component limit
    n_pca = min(
        pca_components,
        n_samples - 1,
        n_features
    )

    if n_pca < 2:
        raise ValueError(
            f"{dataset_name}: insufficient data for PCA."
        )

    if not 0 < perplexity < n_samples:
        raise ValueError(
            f"{dataset_name}: perplexity must be greater than zero "
            f"and smaller than the number of genomes ({n_samples})."
        )

    logging.info(
        "%s: running PCA with %d components.",
        dataset_name,
        n_pca
    )

    pca = PCA(
        n_components=n_pca,
        random_state=random_state
    )

    X_pca = pca.fit_transform(X)

    explained_variance = float(
        pca.explained_variance_ratio_.sum()
    )

    logging.info(
        "%s: cumulative PCA explained variance = %.2f%%.",
        dataset_name,
        explained_variance * 100
    )

    ##Save individual PCA variance values
    pca_variance_df = pd.DataFrame({
        "Principal_Component": [
            f"PC{i + 1}" for i in range(n_pca)
        ],
        "Explained_Variance_Ratio": pca.explained_variance_ratio_,
        "Cumulative_Explained_Variance": np.cumsum(
            pca.explained_variance_ratio_
        )
    })

    pca_variance_df.to_csv(
        output_dir / f"{dataset_name}_PCA_variance.tsv",
        sep="\t",
        index=False
    )

    ##t-SNE analysis
    logging.info(
        "%s: running t-SNE with perplexity %.2f.",
        dataset_name,
        perplexity
    )

    tsne = TSNE(
        n_components=TSNE_COMPONENTS,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=1000
    )

    embedding = tsne.fit_transform(X_pca)

    return embedding, pca, explained_variance


#ARG burden & frequency

def calculate_arg_statistics(rgi, output_dir):

    logging.info("Calculating ARG burden and frequency.")

    ##ARG burden: number of distinct ARG features per genome
    arg_burden = rgi.sum(axis=1)

    ##ARG frequency: number of genomes containing each ARG
    arg_frequency = rgi.sum(axis=0)

    ##Save ARG burden
    burden_df = pd.DataFrame({
        "Genome": rgi.index,
        "ARG_burden": arg_burden.to_numpy()
    })

    burden_df.to_csv(
        output_dir / "ARG_burden_per_genome.tsv",
        sep="\t",
        index=False
    )

    ##Save ARG frequency
    frequency_df = pd.DataFrame({
        "ARG": rgi.columns,
        "Number_of_genomes": arg_frequency.to_numpy()
    })

    frequency_df["Frequency_percent"] = (
        frequency_df["Number_of_genomes"]
        / len(rgi)
        * 100
    )

    frequency_df = frequency_df.sort_values(
        "Number_of_genomes",
        ascending=False
    )

    frequency_df.to_csv(
        output_dir / "ARG_frequency_across_genomes.tsv",
        sep="\t",
        index=False
    )

    ##Summary statistics
    burden_summary = arg_burden.describe()

    frequency_summary = arg_frequency.describe()

    logging.info("ARG burden summary:\n%s", burden_summary)
    logging.info("ARG frequency summary:\n%s", frequency_summary)

    return arg_burden, arg_frequency


#Save t-SNE coordinates

def save_tsne_coordinates(embedding, genome_ids, filename, output_dir):

    coordinates = pd.DataFrame(
        embedding,
        index=genome_ids,
        columns=["tSNE1", "tSNE2"]
    )

    coordinates.index.name = "Genome"

    coordinates.to_csv(
        output_dir / filename,
        sep="\t"
    )

    return coordinates


#Generate publication figure

def generate_figure(
    emb_panaroo,
    emb_rgi,
    arg_burden,
    arg_frequency,
    output_dir,
    width,
    height,
    dpi
):

    logging.info("Generating publication-quality figure.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width, height)
    )

    #Panel A: PAN-genome structure

    axes[0, 0].scatter(
        emb_panaroo[:, 0],
        emb_panaroo[:, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        c="steelblue",
        edgecolors="none"
    )

    axes[0, 0].set_title(
        "a) Pan-genome structure",
        pad=12
    )

    axes[0, 0].set_xlabel("t-SNE1")
    axes[0, 0].set_ylabel("t-SNE2")

    #Panel B: ARG profile structure

    axes[0, 1].scatter(
        emb_rgi[:, 0],
        emb_rgi[:, 1],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        c="darkred",
        edgecolors="none"
    )

    axes[0, 1].set_title(
        "b) ARG profile structure",
        pad=12
    )

    axes[0, 1].set_xlabel("t-SNE1")
    axes[0, 1].set_ylabel("t-SNE2")

    #Panel C: ARG burden

    sns.histplot(
        x=arg_burden,
        bins=ARG_BURDEN_BINS,
        kde=arg_burden.nunique() > 1,
        ax=axes[1, 0],
        color="darkred"
    )

    axes[1, 0].set_title(
        "c) ARG burden per genome",
        pad=12
    )

    axes[1, 0].set_xlabel(
        "Number of ARG features per genome"
    )

    axes[1, 0].set_ylabel(
        "Number of genomes"
    )

    #Panel D: ARG frequency

    sns.histplot(
        x=arg_frequency,
        bins=ARG_FREQUENCY_BINS,
        ax=axes[1, 1],
        color="navy"
    )

    axes[1, 1].set_title(
        "d) ARG frequency distribution",
        pad=12
    )

    axes[1, 1].set_xlabel(
        "Number of genomes containing ARG"
    )

    axes[1, 1].set_ylabel(
        "Number of ARG features"
    )

    ##Final formatting

    for ax in axes.flat:

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.tick_params(
            axis="both",
            which="major",
            width=1.2,
            length=6
        )

        ax.grid(False)

    plt.tight_layout(pad=2.0)

    ##Save PNG
    png_path = (
        output_dir /
        "Genomic_Structure_ARG_Landscape.png"
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    ##Save TIFF
    tiff_path = (
        output_dir /
        "Genomic_Structure_ARG_Landscape.tiff"
    )

    fig.savefig(
        tiff_path,
        dpi=dpi,
        format="tiff",
        bbox_inches="tight"
    )

    plt.close(fig)

    logging.info("PNG figure saved: %s", png_path)
    logging.info("TIFF figure saved: %s", tiff_path)

    return png_path, tiff_path


#Save analysis summary

def save_analysis_summary(
    panaroo,
    rgi,
    arg_burden,
    arg_frequency,
    pca_panaroo,
    pca_rgi,
    variance_panaroo,
    variance_rgi,
    perplexity,
    random_state,
    output_dir
):

    summary = {
        "number_of_genomes": int(len(panaroo)),

        "pan_genome": {
            "number_of_genomes": int(panaroo.shape[0]),
            "number_of_variable_gene_families": int(panaroo.shape[1]),
            "pca_components": int(pca_panaroo.n_components_),
            "pca_cumulative_explained_variance": float(
                variance_panaroo
            )
        },

        "arg_profile": {
            "number_of_genomes": int(rgi.shape[0]),
            "number_of_variable_arg_features": int(rgi.shape[1]),
            "pca_components": int(pca_rgi.n_components_),
            "pca_cumulative_explained_variance": float(
                variance_rgi
            )
        },

        "arg_burden": {
            "mean": float(arg_burden.mean()),
            "median": float(arg_burden.median()),
            "minimum": int(arg_burden.min()),
            "maximum": int(arg_burden.max())
        },

        "arg_frequency": {
            "mean_genomes_per_arg": float(arg_frequency.mean()),
            "minimum": int(arg_frequency.min()),
            "maximum": int(arg_frequency.max())
        },

        "tsne": {
            "perplexity": float(perplexity),
            "random_state": int(random_state),
            "dimensions": TSNE_COMPONENTS
        }
    }

    summary_path = output_dir / "analysis_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    logging.info("Analysis summary saved: %s", summary_path)


# Main workflow

def main():

    args = parse_arguments()

    global output_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logging.info("=" * 65)
    logging.info("PAN-GENOME AND ARG LANDSCAPE ANALYSIS")
    logging.info("=" * 65)

    # Validate input files.
    panaroo_path = validate_input_file(
        args.pan_genome,
        "Pan-genome matrix"
    )

    rgi_path = validate_input_file(
        args.arg_matrix,
        "ARG matrix"
    )

    if args.pca_components < 2:
        raise ValueError("PCA components must be at least 2.")

    if args.width <= 0 or args.height <= 0:
        raise ValueError("Figure width and height must be positive.")

    if args.dpi <= 0:
        raise ValueError("DPI must be positive.")

    # Load matrices.
    panaroo = load_matrix(
        panaroo_path,
        "Pan-genome"
    )

    rgi = load_matrix(
        rgi_path,
        "ARG"
    )

    ##Match genomes
    panaroo, rgi = match_genomes(
        panaroo,
        rgi,
        output_dir,
        args.min_genomes
    )

    logging.info(
        "Matched dataset: %d genomes.",
        len(panaroo)
    )

    ##Pan-genome PCA + t-SNE
    logging.info("=" * 65)
    logging.info("PAN-GENOME PCA + t-SNE")
    logging.info("=" * 65)

    emb_panaroo, pca_panaroo, variance_panaroo = (
        pca_tsne_embedding(
            panaroo,
            "Pan_genome",
            args.pca_components,
            args.perplexity,
            args.random_state
        )
    )

    ##ARG PCA + t-SNE
    logging.info("=" * 65)
    logging.info("ARG PROFILE PCA + t-SNE")
    logging.info("=" * 65)

    emb_rgi, pca_rgi, variance_rgi = (
        pca_tsne_embedding(
            rgi,
            "ARG",
            args.pca_components,
            args.perplexity,
            args.random_state
        )
    )

    ##Calculate ARG burden and frequency
    arg_burden, arg_frequency = calculate_arg_statistics(
        rgi,
        output_dir
    )

    ##Save t-SNE coordinates
    save_tsne_coordinates(
        emb_panaroo,
        panaroo.index,
        "Pan_genome_tSNE_coordinates.tsv",
        output_dir
    )

    save_tsne_coordinates(
        emb_rgi,
        rgi.index,
        "ARG_tSNE_coordinates.tsv",
        output_dir
    )

    ##Generate publication figure
    generate_figure(
        emb_panaroo,
        emb_rgi,
        arg_burden,
        arg_frequency,
        output_dir,
        args.width,
        args.height,
        args.dpi
    )

    ##Save analysis summary
    save_analysis_summary(
        panaroo,
        rgi,
        arg_burden,
        arg_frequency,
        pca_panaroo,
        pca_rgi,
        variance_panaroo,
        variance_rgi,
        args.perplexity,
        args.random_state,
        output_dir
    )

    ##Final report
    logging.info("=" * 65)
    logging.info("ANALYSIS COMPLETED SUCCESSFULLY")
    logging.info("=" * 65)

    logging.info("Matched genomes: %d", len(panaroo))
    logging.info("Pan-genome gene families: %d", panaroo.shape[1])
    logging.info("ARG features: %d", rgi.shape[1])

    logging.info("Mean ARG burden: %.2f", arg_burden.mean())
    logging.info("Median ARG burden: %.2f", arg_burden.median())
    logging.info("Minimum ARG burden: %d", arg_burden.min())
    logging.info("Maximum ARG burden: %d", arg_burden.max())

    logging.info(
        "Pan-genome PCA variance explained: %.2f%%",
        variance_panaroo * 100
    )

    logging.info(
        "ARG PCA variance explained: %.2f%%",
        variance_rgi * 100
    )

    logging.info("All results saved in: %s", output_dir)


#Execution

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        logging.exception(
            "Analysis failed: %s",
            error
        )

        sys.exit(1)

