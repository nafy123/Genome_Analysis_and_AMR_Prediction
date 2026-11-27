#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
NCBI & BV-BRC feature intersection and filtering pipeline

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description:
This script performs feature harmonization, intersection, quality
assessment, and filtering of genomic feature matrices obtained
from the NCBI & BV-BRC.

The pipeline identifies shared and dataset-specific genomic
features, evaluates feature prevalence and data quality, and
generates harmonized feature matrices suitable for downstream
ML model development and external validation.

The filtering procedure is designed to retain informative,
comparable genomic features across both datasets while removing
aggregate variables, non-informative metadata fields, highly
prevalent features, and rare features according to predefined
filtering thresholds.

Objectives:
a. Load and validate genomic feature matrices obtained from
   NCBI and BV-BRC.
b. Verify dataset structure, genome identifiers, class labels,
   and required metadata fields.
c. Identify shared, NCBI-specific, and BV-BRC-specific features.
d. Evaluate feature completeness, prevalence, and data quality.
e. Compare feature distributions and prevalence between datasets.
f. Exclude aggregate variables, including columns beginning
   with the "group_" prefix.
g. Remove non-informative metadata fields from the feature matrices.
h. Identify common genomic features shared between NCBI and BV-BRC.
i. Exclude near-constant features with prevalence greater than
   the specified maximum threshold (default: 99%).
j. Exclude rare features detected in fewer than the specified
    minimum number of genomes (default: 5).
k. Retain common genomic features that satisfy the filtering
    criteria in both datasets.
l. Generate harmonized NCBI and BV-BRC feature matrices for
    downstream machine-learning analysis.
m. Export feature intersection results, filtering reports,
    dataset summaries, and pipeline execution metadata.

Analytical workflow:

a. Load the NCBI and BV-BRC tab-separated feature matrices.
b. Validate input files and confirm the presence of required
   metadata columns.
c. Extract genome identifiers, class labels, and genomic
   feature columns.
d. Assess feature types, missing values, and dataset consistency.
e. Identify shared and dataset-specific genomic features.
f. Calculate feature prevalence within each dataset.
g. Compare feature prevalence and data quality between NCBI
   and BV-BRC.
h. Remove aggregate variables and non-informative metadata
   fields from the candidate feature sets.
i. Identify the intersection of genomic features available
   in both datasets.
j. Apply prevalence-based filtering to remove near-constant
    features exceeding the specified maximum prevalence.
k. Apply minimum occurrence filtering to exclude rare features
    detected in fewer than the specified number of genomes.
l. Retain features satisfying the filtering criteria in BOTH
    datasets.
m. Generate harmonized feature matrices while preserving
    genome identifiers and class labels.
n. Export filtering summaries, feature comparison reports,
    and final processed datasets.

Input files:

a. NCBI_shared_features.tsv
   Genomic feature matrix obtained from the NCBI dataset,
   containing genome identifiers, class labels, and genomic
   feature variables.

b. BVBRC_shared_features.tsv
   Genomic feature matrix obtained from the BV-BRC dataset,
   containing genome identifiers, class labels, and genomic
   feature variables.

Input data requirements:

Both datasets must be tab-separated files (.tsv) containing:

Required metadata columns:
- Genome ID: Unique identifier for each bacterial genome.
- label: Class or phenotype label used for downstream
  machine-learning analysis.

Genomic feature columns:
- Individual genomic features represented as binary variables.
- 0: Feature absent.
- 1: Feature present.

Data preprocessing:
- Missing values are treated as undetected features during
  prevalence calculations, according to the implemented
  filtering procedure.
- Non-numeric feature values are converted to missing values.
- Aggregate variables beginning with "group_" are excluded.
- Non-informative metadata fields are removed from candidate
  feature sets.

Feature filtering criteria:
- Maximum prevalence threshold: 0.99 (99%).
- Minimum feature occurrence: 5 genomes.
- Features must be shared between NCBI and BV-BRC.
- Retained features must satisfy the filtering criteria
  independently in BOTH datasets.

Command-line arguments:

--ncbi:
    Path to the NCBI genomic feature matrix.

--bvbrc:
    Path to the BV-BRC genomic feature matrix.

--output:
    Directory for saving harmonized datasets, filtering
    reports, and pipeline summaries.

--max-prevalence:
    Maximum permitted feature prevalence.
    Default: 0.99.

--min-feature-count:
    Minimum number of genomes required to retain a feature.
    Default: 5.

Example execution:
python feature_intersection_filtering.py \
    --ncbi data/NCBI_shared_features.tsv \
    --bvbrc data/BVBRC_shared_features.tsv \
    --output results/ \
    --max-prevalence 0.99 \
    --min-feature-count 5

Expected outputs:
- Shared genomic feature list.
- NCBI-specific feature list.
- BV-BRC-specific feature list.
- Feature prevalence comparison results.
- Feature quality assessment report.
- Feature filtering summary.
- Final filtered NCBI feature matrix.
- Final filtered BV-BRC feature matrix.
- Harmonized feature intersection dataset.
- Pipeline execution summary and metadata.

Reproducibility:
Feature intersection and filtering are performed using
documented prevalence and minimum occurrence thresholds.
The same filtering criteria are applied independently
to both datasets to maintain feature consistency.

The resulting harmonized datasets are intended for
downstream feature engineering, machine-learning model
development, and external validation.

Software requirements:
- Python 3.11
- Pandas
- NumPy

Installation:
pip install pandas numpy

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
import logging
import sys

from pathlib import Path

import pandas as pd
import numpy as np

#Configuration

METADATA_COLS = [
    "Genome ID",
    "label"
]

METADATA_PREFIXES = (
    "metadata_",
    "meta_",
    "sample_",
    "accession_"
)

GROUP_PREFIX = "group_"

DEFAULT_MAX_PREVALENCE = 0.99

DEFAULT_MIN_FEATURE_COUNT = 5

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(message)s"
)

#Logging configuration

def configure_logging(output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = output_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_file,
                mode="w",
                encoding="utf-8"
            )
        ],
        force=True
    )

#Command-line arguments

def parse_arguments():

    parser = argparse.ArgumentParser(

        description=(
            "Identify shared NCBI and BV-BRC gene features "
            "and generate filtered ML datasets."
        )

    )

    parser.add_argument(

        "--ncbi",
        required=True,
        type=Path,
        help="Path to the NCBI feature matrix TSV file."

    )

    parser.add_argument(

        "--bvbrc",
        required=True,
        type=Path,
        help="Path to the BV-BRC feature matrix TSV file."

    )

    parser.add_argument(

        "--output",
        required=True,
        type=Path,
        help="Output directory."

    )

    parser.add_argument(

        "--max-prevalence",
        type=float,
        default=DEFAULT_MAX_PREVALENCE,
        help="Maximum permitted feature prevalence (default: 0.99)."

    )

    parser.add_argument(

        "--min-feature-count",
        type=int,
        default=DEFAULT_MIN_FEATURE_COUNT,
        help="Minimum number of genomes containing a feature (default: 5)."

    )

    return parser.parse_args()

#Input validation

def validate_arguments(args):

    if not args.ncbi.is_file():

        raise FileNotFoundError(
            f"NCBI input file not found: {args.ncbi}"
        )

    if not args.bvbrc.is_file():

        raise FileNotFoundError(
            f"BV-BRC input file not found: {args.bvbrc}"
        )

    if not 0 <= args.max_prevalence <= 1:

        raise ValueError(
            "--max-prevalence must be between 0 and 1."
        )

    if args.min_feature_count < 1:

        raise ValueError(
            "--min-feature-count must be at least 1."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )


#Load datasets

def load_dataset(file_path, dataset_name):

    logging.info(
        "Loading %s dataset: %s",
        dataset_name,
        file_path
    )

    df = pd.read_csv(
        file_path,
        sep="\t",
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    if df.empty:

        raise ValueError(
            f"{dataset_name} dataset is empty."
        )

    if df.columns.duplicated().any():

        raise ValueError(
            f"{dataset_name} contains duplicate column names."
        )

    missing_cols = [
        col for col in METADATA_COLS
        if col not in df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_cols}"
        )

    if df["Genome ID"].isna().any():

        raise ValueError(
            f"{dataset_name} contains missing Genome IDs."
        )

    if df["Genome ID"].astype(str).str.strip().eq("").any():

        raise ValueError(
            f"{dataset_name} contains empty Genome IDs."
        )

    if df["Genome ID"].duplicated().any():

        raise ValueError(
            f"{dataset_name} contains duplicate Genome IDs."
        )

    if df["label"].isna().any():

        raise ValueError(
            f"{dataset_name} contains missing labels."
        )

    if df["label"].astype(str).str.strip().eq("").any():

        raise ValueError(
            f"{dataset_name} contains empty labels."
        )

    logging.info(
        "%s dataset loaded successfully: %s",
        dataset_name,
        df.shape
    )

    return df


#Identify feature columns

def identify_features(df):

    return [

        col for col in df.columns

        if col not in METADATA_COLS

    ]


#Identify common gene features

def identify_common_genes(ncbi, bvbrc):

    ncbi_genes = [

        col for col in ncbi.columns

        if col not in METADATA_COLS

        and not col.lower().startswith(GROUP_PREFIX)

    ]

    bvbrc_genes = [

        col for col in bvbrc.columns

        if col not in METADATA_COLS

        and not col.lower().startswith(GROUP_PREFIX)

    ]

    common_genes = sorted(

        set(ncbi_genes) & set(bvbrc_genes)

    )

    if not common_genes:

        raise ValueError(
            "No common gene features were identified."
        )

    logging.info(
        "NCBI gene features: %d",
        len(ncbi_genes)
    )

    logging.info(
        "BV-BRC gene features: %d",
        len(bvbrc_genes)
    )

    logging.info(
        "Common gene features: %d",
        len(common_genes)
    )

    return (
        ncbi_genes,
        bvbrc_genes,
        common_genes
    )


#Save feature intersection reports

def save_intersection_reports(

    ncbi_features,
    bvbrc_features,
    common_features,
    output_dir

):

    intersection_dir = (
        output_dir / "01_Feature_Intersection"
    )

    intersection_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ncbi_only = sorted(
        set(ncbi_features) - set(bvbrc_features)
    )

    bvbrc_only = sorted(
        set(bvbrc_features) - set(ncbi_features)
    )

    reports = {

        "Shared_features.tsv": common_features,

        "NCBI_only_features.tsv": ncbi_only,

        "BVBRC_only_features.tsv": bvbrc_only

    }

    for filename, features in reports.items():

        pd.Series(
            features,
            name="Feature"
        ).to_csv(

            intersection_dir / filename,

            sep="\t",

            index=False

        )

    logging.info(
        "Feature intersection reports saved."
    )

    return (
        ncbi_only,
        bvbrc_only,
        intersection_dir
    )


#Feature data quality assessment

def assess_data_quality(

    ncbi,
    bvbrc,
    shared_features,
    output_dir

):

    ncbi_numeric = ncbi[
        shared_features
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    bvbrc_numeric = bvbrc[
        shared_features
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    quality_records = []

    for name, numeric in [

        ("NCBI", ncbi_numeric),

        ("BV-BRC", bvbrc_numeric)

    ]:

        nonbinary = [

            col for col in shared_features

            if not numeric[col].dropna().isin([0, 1]).all()

        ]

        missing = int(
            numeric.isna().sum().sum()
        )

        quality_records.append({

            "Dataset": name,

            "Total_Genomes": len(numeric),

            "Shared_Features": len(shared_features),

            "Missing_Value_Entries": missing,

            "Nonbinary_Features": len(nonbinary),

            "Nonbinary_Feature_Names": ";".join(nonbinary)

        })

    quality_summary = pd.DataFrame(
        quality_records
    )

    quality_summary.to_csv(

        output_dir /
        "Feature_Data_Quality_Summary.tsv",

        sep="\t",

        index=False

    )

    logging.info(
        "Feature data quality assessment completed."
    )

    return quality_summary


#Feature prevalence comparison

def compare_feature_prevalence(

    ncbi,
    bvbrc,
    shared_features,
    output_dir

):

    records = []

    for name, df in [

        ("NCBI", ncbi),

        ("BV-BRC", bvbrc)

    ]:

        numeric = df[
            shared_features
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        counts = numeric.eq(1).sum()

        prevalence = (
            counts / len(df)
        )

        records.append({

            "Dataset": name,

            "Counts": counts,

            "Prevalence": prevalence

        })

    ncbi_counts = records[0]["Counts"]

    bvbrc_counts = records[1]["Counts"]

    ncbi_prevalence = records[0]["Prevalence"]

    bvbrc_prevalence = records[1]["Prevalence"]

    comparison = pd.DataFrame({

        "Feature": shared_features,

        "NCBI_Detected_Genomes":
            ncbi_counts.reindex(shared_features).values,

        "BV_BRC_Detected_Genomes":
            bvbrc_counts.reindex(shared_features).values,

        "NCBI_Prevalence":
            ncbi_prevalence.reindex(shared_features).values,

        "BV_BRC_Prevalence":
            bvbrc_prevalence.reindex(shared_features).values

    })

    comparison["Prevalence_Difference"] = (

        comparison["BV_BRC_Prevalence"]

        - comparison["NCBI_Prevalence"]

    )

    comparison["Absolute_Difference"] = (

        comparison["Prevalence_Difference"].abs()

    )

    comparison = comparison.sort_values(

        "Absolute_Difference",

        ascending=False

    )

    comparison.to_csv(

        output_dir /
        "Shared_Feature_Prevalence_Comparison.tsv",

        sep="\t",

        index=False

    )

    logging.info(
        "Feature prevalence comparison completed."
    )

    return comparison


#Identify excluded aggregate and metadata columns

def identify_excluded_columns(df):

    aggregate_cols = [

        col for col in df.columns

        if col.lower().startswith(GROUP_PREFIX)

    ]

    metadata_cols = [

        col for col in df.columns

        if col not in METADATA_COLS

        and col.lower().startswith(METADATA_PREFIXES)

    ]

    return (
        aggregate_cols,
        metadata_cols
    )


#Filter near-constant and rare features

def filter_features(

    df,
    dataset_name,
    common_genes,
    max_prevalence,
    min_feature_count

):

    numeric = df[
        common_genes
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    ##A feature is detected only when its value is exactly 1

    feature_counts = numeric.eq(1).sum()

    ##Missing values remain in the denominator
    ##This gives prevalence relative to all genomes

    feature_prevalence = (
        feature_counts / len(df)
    )

    near_constant = [

        col for col in common_genes

        if feature_prevalence[col] > max_prevalence

    ]

    rare_features = [

        col for col in common_genes

        if feature_counts[col] < min_feature_count

    ]

    excluded = sorted(

        set(near_constant) | set(rare_features)

    )

    retained = [

        col for col in common_genes

        if col not in excluded

    ]

    ##Record all filtering decisions

    filtering_report = pd.DataFrame({

        "Feature": common_genes,

        "Detected_Genomes": [

            int(feature_counts[col])

            for col in common_genes

        ],

        "Prevalence": [

            float(feature_prevalence[col])

            for col in common_genes

        ],

        "Near_Constant": [

            col in near_constant

            for col in common_genes

        ],

        "Rare_Feature": [

            col in rare_features

            for col in common_genes

        ],

        "Retained": [

            col in retained

            for col in common_genes

        ]

    })

    ##Explicit exclusion reason

    filtering_report["Exclusion_Reason"] = [

        ";".join(reason)

        if reason else "Retained"

        for reason in [

            [

                label for condition, label in [

                    (
                        col in near_constant,
                        "Near_constant"
                    ),

                    (
                        col in rare_features,
                        "Rare_feature"
                    )

                ]

                if condition

            ]

            for col in common_genes

        ]

    ]

    logging.info(
        "%s filtering results:",
        dataset_name
    )

    logging.info(
        "Original common features: %d",
        len(common_genes)
    )

    logging.info(
        "Near-constant features: %d",
        len(near_constant)
    )

    logging.info(
        "Rare features: %d",
        len(rare_features)
    )

    logging.info(
        "Unique excluded features: %d",
        len(excluded)
    )

    logging.info(
        "Retained features: %d",
        len(retained)
    )

    return {

        "report": filtering_report,

        "near_constant": near_constant,

        "rare": rare_features,

        "excluded": excluded,

        "retained": retained

    }


#Save feature filtering reports

def save_filtering_reports(

    ncbi_result,
    bvbrc_result,
    output_dir

):

    reports_dir = (
        output_dir / "04_Feature_Filtering_Reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {

        "NCBI": ncbi_result,

        "BV-BRC": bvbrc_result

    }

    for name, result in results.items():

        result["report"].to_csv(

            reports_dir /
            f"{name}_Feature_Filtering_Report.tsv",

            sep="\t",

            index=False

        )

        excluded_report = pd.DataFrame({

            "Near_Constant_Features": pd.Series(
                result["near_constant"]
            ),

            "Rare_Features": pd.Series(
                result["rare"]
            ),

            "All_Excluded_Features": pd.Series(
                result["excluded"]
            )

        })

        excluded_report.to_csv(

            reports_dir /
            f"{name}_Excluded_Features.tsv",

            sep="\t",

            index=False

        )

    logging.info(
        "Feature filtering reports saved."
    )

    return reports_dir


#Generate final common feature set

def generate_final_features(

    ncbi_result,
    bvbrc_result

):

    final_features = sorted(

        set(ncbi_result["retained"])

        &

        set(bvbrc_result["retained"])

    )

    if not final_features:

        raise ValueError(
            "No common features remain after filtering."
        )

    logging.info(
        "Final common features retained: %d",
        len(final_features)
    )

    return final_features


#Save final datasets

def save_final_datasets(

    ncbi,
    bvbrc,
    final_features,
    output_dir

):

    final_dir = (
        output_dir / "03_Final_Filtered_Datasets"
    )

    shared_dir = (
        output_dir / "02_Shared_Gene_Analysis"
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    shared_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ncbi_final = ncbi[
        METADATA_COLS + final_features
    ].copy()

    bvbrc_final = bvbrc[
        METADATA_COLS + final_features
    ].copy()

    ##Save common gene features.

    pd.Series(

        final_features,

        name="Final_Retained_Features"

    ).to_csv(

        final_dir /
        "Final_Common_Features.tsv",

        sep="\t",

        index=False

    )

    ##Save harmonized datasets.

    ncbi_final.to_csv(

        final_dir /
        "NCBI_Final_ML_Dataset.tsv",

        sep="\t",

        index=False

    )

    bvbrc_final.to_csv(

        final_dir /
        "BVBRC_Final_ML_Dataset.tsv",

        sep="\t",

        index=False

    )

    logging.info(
        "Final datasets saved successfully."
    )

    return (
        ncbi_final,
        bvbrc_final,
        final_dir
    )


#Generate exclusion summary

def generate_exclusion_summary(

    ncbi_aggregate,
    bvbrc_aggregate,
    ncbi_metadata,
    bvbrc_metadata,
    ncbi_result,
    bvbrc_result,
    final_features,
    output_dir

):

    reports_dir = (
        output_dir / "04_Feature_Filtering_Reports"
    )

    summary = pd.DataFrame({

        "Dataset": [
            "NCBI",
            "BV-BRC"
        ],

        "Excluded_Aggregate_Columns": [

            len(ncbi_aggregate),

            len(bvbrc_aggregate)

        ],

        "Noninformative_Metadata_Fields": [

            len(ncbi_metadata),

            len(bvbrc_metadata)

        ],

        "Near_Constant_Features": [

            len(ncbi_result["near_constant"]),

            len(bvbrc_result["near_constant"])

        ],

        "Rare_Features": [

            len(ncbi_result["rare"]),

            len(bvbrc_result["rare"])

        ],

        "Unique_Filtered_Features": [

            len(ncbi_result["excluded"]),

            len(bvbrc_result["excluded"])

        ],

        "Final_Retained_Common_Features": [

            len(final_features),

            len(final_features)

        ]

    })

    summary.to_csv(

        reports_dir /
        "Feature_Exclusion_Summary.tsv",

        sep="\t",

        index=False

    )

    return summary


#Generate pipeline summary

def generate_pipeline_summary(

    ncbi,
    bvbrc,
    ncbi_features,
    bvbrc_features,
    shared_features,
    ncbi_only,
    bvbrc_only,
    common_genes,
    ncbi_result,
    bvbrc_result,
    final_features,
    ncbi_final,
    bvbrc_final,
    ncbi_quality,
    bvbrc_quality,
    args,
    output_dir

):

    summary = {

        "NCBI_Genomes": len(ncbi),

        "BV_BRC_Genomes": len(bvbrc),

        "NCBI_Original_Features": len(ncbi_features),

        "BV_BRC_Original_Features": len(bvbrc_features),

        "Initial_Shared_Features": len(shared_features),

        "NCBI_Only_Features": len(ncbi_only),

        "BV_BRC_Only_Features": len(bvbrc_only),

        "Common_Gene_Features": len(common_genes),

        "NCBI_Near_Constant_Features":
            len(ncbi_result["near_constant"]),

        "BV_BRC_Near_Constant_Features":
            len(bvbrc_result["near_constant"]),

        "NCBI_Rare_Features":
            len(ncbi_result["rare"]),

        "BV_BRC_Rare_Features":
            len(bvbrc_result["rare"]),

        "NCBI_Unique_Filtered_Features":
            len(ncbi_result["excluded"]),

        "BV_BRC_Unique_Filtered_Features":
            len(bvbrc_result["excluded"]),

        "Final_Common_Features":
            len(final_features),

        "NCBI_Missing_Entries":
            ncbi_quality["Missing_Value_Entries"],

        "BV_BRC_Missing_Entries":
            bvbrc_quality["Missing_Value_Entries"],

        "NCBI_Nonbinary_Features":
            ncbi_quality["Nonbinary_Features"],

        "BV_BRC_Nonbinary_Features":
            bvbrc_quality["Nonbinary_Features"],

        "NCBI_Final_Features":
            ncbi_final.shape[1] - len(METADATA_COLS),

        "BV_BRC_Final_Features":
            bvbrc_final.shape[1] - len(METADATA_COLS),

        "Maximum_Prevalence_Threshold":
            args.max_prevalence,

        "Minimum_Feature_Count":
            args.min_feature_count

    }

    summary_df = pd.DataFrame(

        list(summary.items()),

        columns=["Metric", "Value"]

    )

    summary_df.to_csv(

        output_dir /
        "Final_ML_Feature_Analysis_Summary.tsv",

        sep="\t",

        index=False

    )

    return summary_df


#Main pipeline

def main():

    args = parse_arguments()

    validate_arguments(args)

    configure_logging(args.output)

    logging.info("=" * 75)

    logging.info(
        "NCBI AND BV-BRC FEATURE INTERSECTION PIPELINE"
    )

    logging.info("=" * 75)

    ##Load datasets

    ncbi = load_dataset(
        args.ncbi,
        "NCBI"
    )

    bvbrc = load_dataset(
        args.bvbrc,
        "BV-BRC"
    )

    ##Identify features

    ncbi_features = identify_features(ncbi)

    bvbrc_features = identify_features(bvbrc)

    shared_features = sorted(

        set(ncbi_features)

        &

        set(bvbrc_features)

    )

    if not shared_features:

        raise ValueError(
            "No shared features between NCBI and BV-BRC."
        )

    ##Feature intersection

    (
        ncbi_only,
        bvbrc_only,
        intersection_dir

    ) = save_intersection_reports(

        ncbi_features,

        bvbrc_features,

        shared_features,

        args.output

    )

    ##Data quality assessment

    quality_dir = intersection_dir

    quality = assess_data_quality(

        ncbi,

        bvbrc,

        shared_features,

        quality_dir

    )

    ncbi_quality = quality[
        quality["Dataset"] == "NCBI"
    ].iloc[0]

    bvbrc_quality = quality[
        quality["Dataset"] == "BV-BRC"
    ].iloc[0]

    ##Feature prevalence comparison

    compare_feature_prevalence(

        ncbi,

        bvbrc,

        shared_features,

        intersection_dir

    )

    ##Identify common gene features

    (
        ncbi_genes,
        bvbrc_genes,
        common_genes

    ) = identify_common_genes(

        ncbi,

        bvbrc

    )

    ##Identify excluded aggregate and metadata columns

    (
        ncbi_aggregate,
        ncbi_metadata

    ) = identify_excluded_columns(ncbi)

    (
        bvbrc_aggregate,
        bvbrc_metadata

    ) = identify_excluded_columns(bvbrc)

    ##Filter common gene features independently

    ncbi_result = filter_features(

        ncbi,

        "NCBI",

        common_genes,

        args.max_prevalence,

        args.min_feature_count

    )

    bvbrc_result = filter_features(

        bvbrc,

        "BV-BRC",

        common_genes,

        args.max_prevalence,

        args.min_feature_count

    )

    ##Identify features retained in BOTH datasets

    final_features = generate_final_features(

        ncbi_result,

        bvbrc_result

    )

    ##Save final datasets

    (
        ncbi_final,
        bvbrc_final,
        final_dir

    ) = save_final_datasets(

        ncbi,

        bvbrc,

        final_features,

        args.output

    )

    ##Save filtering reports

    reports_dir = save_filtering_reports(

        ncbi_result,

        bvbrc_result,

        args.output

    )

    ##Save exclusion summary

    generate_exclusion_summary(

        ncbi_aggregate,

        bvbrc_aggregate,

        ncbi_metadata,

        bvbrc_metadata,

        ncbi_result,

        bvbrc_result,

        final_features,

        args.output

    )

    ##Save final pipeline summary

    generate_pipeline_summary(

        ncbi,

        bvbrc,

        ncbi_features,

        bvbrc_features,

        shared_features,

        ncbi_only,

        bvbrc_only,

        common_genes,

        ncbi_result,

        bvbrc_result,

        final_features,

        ncbi_final,

        bvbrc_final,

        ncbi_quality,

        bvbrc_quality,

        args,

        args.output

    )

    ##Final results

    logging.info("=" * 75)

    logging.info(
        "FEATURE PROCESSING COMPLETED SUCCESSFULLY"
    )

    logging.info("=" * 75)

    logging.info(
        "NCBI original dimensions: %s",
        ncbi.shape
    )

    logging.info(
        "NCBI final dimensions: %s",
        ncbi_final.shape
    )

    logging.info(
        "BV-BRC original dimensions: %s",
        bvbrc.shape
    )

    logging.info(
        "BV-BRC final dimensions: %s",
        bvbrc_final.shape
    )

    logging.info(
        "Initial shared features: %d",
        len(shared_features)
    )

    logging.info(
        "Common gene features: %d",
        len(common_genes)
    )

    logging.info(
        "Final shared features: %d",
        len(final_features)
    )

    logging.info(
        "Maximum prevalence threshold: %.2f",
        args.max_prevalence
    )

    logging.info(
        "Minimum feature detection count: %d",
        args.min_feature_count
    )

    logging.info(
        "Output directory: %s",
        args.output.resolve()
    )

    logging.info(
        "Intersection reports: %s",
        intersection_dir
    )

    logging.info(
        "Final datasets: %s",
        final_dir
    )

    logging.info(
        "Filtering reports: %s",
        reports_dir
    )

    logging.info("=" * 75)


#Script entry point

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logging.error(
            "Pipeline failed: %s",
            error
        )

        sys.exit(1)
#End

