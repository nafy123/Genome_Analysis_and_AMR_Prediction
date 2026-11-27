#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Integration of AMR gene and pan-genome features for ML analysis

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description:
This script integrates ARG and pan-genome gene-family
presence/absence matrices to generate combined genomic
feature datasets for ML analysis.

Separate integration procedures are performed for NCBI
internal validation and BV-BRC external validation datasets.

Main operations:
a. Load preprocessed ARG and pan-genome datasets.
b. Validate input files and required metadata columns.
c. Verify genome identifiers and classification labels.
d. Check dataset dimensions and feature consistency.
e. Identify overlapping feature names.
f. Combine ARG and pan-genome genomic features.
g. Preserve Genome ID and label columns.
h. Export separate integrated datasets for internal
   and external validation.

Input files:
- ARG_preprocessed_ML.tsv
- PAN_preprocessed_ML.tsv

Required columns:
- Genome ID
- label
- Binary genomic features (0 = absent, 1 = present).

Input directories:
- data/internal_validation/
- data/external_validation/

Expected outputs:
- ARG_PAN_combined_ML.tsv for internal validation.
- ARG_PAN_combined_ML.tsv for external validation.
- GenomeID_label_mismatches.tsv (if mismatches occur).

Output directories:
- results/internal_validation/model_development/
- results/external_validation/model_development/

Reproducibility:
Genome identifiers and classification labels are checked
row by row before integration. Original genome ordering
is preserved, and both datasets are processed independently.

Software requirements:
- Python 3.11
- Pandas

Installation:
pip install pandas

Execution:
python combine_ARG_PAN_features.py

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

from pathlib import Path
import pandas as pd

#Define project directories

PROJECT_DIR = Path(__file__).resolve().parent.parent

#Define dataset locations

datasets = {

    "internal_validation": {
        "arg": (
            PROJECT_DIR / "data" / "internal_validation"
            / "ARG_preprocessed_ML.tsv"
        ),
        "pan": (
            PROJECT_DIR / "data" / "internal_validation"
            / "PAN_preprocessed_ML.tsv"
        )
    },

    "external_validation": {
        "arg": (
            PROJECT_DIR / "data" / "external_validation"
            / "ARG_preprocessed_ML.tsv"
        ),
        "pan": (
            PROJECT_DIR / "data" / "external_validation"
            / "PAN_preprocessed_ML.tsv"
        )
    }
}

#Define combination function

def combine_arg_pan(arg_file, pan_file, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    ##Verify input files
    for file in [arg_file, pan_file]:

        if not file.is_file():
            raise FileNotFoundError(
                f"Input file not found: {file}"
            )

    ##Read datasets
    arg_df = pd.read_csv(arg_file, sep="\t")
    pan_df = pd.read_csv(pan_file, sep="\t")

    print("\nARG dataset shape:", arg_df.shape)
    print("PAN dataset shape:", pan_df.shape)

    ##Required columns
    required_cols = ["Genome ID", "label"]

    for name, df in [("ARG", arg_df), ("PAN", pan_df)]:

        missing = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if df.columns.duplicated().any():
            raise ValueError(
                f"Duplicate column names in {name} dataset."
            )

    ##Check row counts
    if len(arg_df) != len(pan_df):

        raise ValueError(
            "ARG and PAN datasets have different row counts."
        )

    ##Compare genome_IDs and labels
    genome_match = arg_df["Genome ID"].equals(
        pan_df["Genome ID"]
    )

    label_match = arg_df["label"].equals(
        pan_df["label"]
    )

    if not (genome_match and label_match):

        mismatch_df = pd.DataFrame({

            "ARG_Genome_ID": arg_df["Genome ID"],
            "PAN_Genome_ID": pan_df["Genome ID"],
            "ARG_label": arg_df["label"],
            "PAN_label": pan_df["label"]

        })

        mismatches = mismatch_df.loc[
            (arg_df["Genome ID"] != pan_df["Genome ID"]) |
            (arg_df["label"] != pan_df["label"])
        ]

        mismatch_file = (
            output_dir / "GenomeID_label_mismatches.tsv"
        )

        mismatches.to_csv(
            mismatch_file,
            sep="\t",
            index=False
        )

        raise ValueError(
            f"Genome ID/label mismatch. Report: {mismatch_file}"
        )

    print("Genome IDs and labels match successfully.")

    ##Identify feature columns
    arg_features = [
        col for col in arg_df.columns
        if col not in required_cols
    ]

    pan_features = [
        col for col in pan_df.columns
        if col not in required_cols
    ]

    ##Check overlapping feature names
    duplicates = sorted(
        set(arg_features) & set(pan_features)
    )

    if duplicates:
        raise ValueError(
            f"Duplicate feature names detected: {duplicates[:20]}"
        )

    ##Remove duplicate key columns from PAN
    pan_features_df = pan_df.drop(
        columns=required_cols
    )

    ##Combine ARG and PAN features
    combined_df = pd.concat(
        [
            arg_df.reset_index(drop=True),
            pan_features_df.reset_index(drop=True)
        ],
        axis=1
    )

    ##Save combined dataset
    output_file = (
        output_dir / "ARG_PAN_combined_ML.tsv"
    )

    combined_df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print("\nCombined dataset saved:", output_file)
    print("Combined dataset shape:", combined_df.shape)

    return combined_df

#Process both datasets

for dataset_name, files in datasets.items():

    print("\n" + "=" * 70)
    print(f"PROCESSING: {dataset_name.upper()}")
    print("=" * 70)

    output_dir = (
        PROJECT_DIR
        / "results"
        / dataset_name
        / "model_development"
    )

    combine_arg_pan(
        arg_file=files["arg"],
        pan_file=files["pan"],
        output_dir=output_dir
    )

#Completion

print("\n" + "=" * 70)
print("ARG-PAN combination completed for both datasets")
print("=" * 70)