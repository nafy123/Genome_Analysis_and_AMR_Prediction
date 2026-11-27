#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
External validation of NCBI-trained AMR prediction models using an independent BV-BRC genomic dataset

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description: This script performs independent external validation of ML models developed and trained using the
NCBI genomic feature dataset, using an independent BV-BRC genomic dataset.

The validation evaluates the generalizability and predictive performance of four previously developed AMR classification models: 
Random Forest (RF),Extreme Gradient Boosting (XGBoost), Logistic Regression (LR),& Support Vector Machine (SVM).

Previously fitted models are loaded from disk and evaluated directly using the harmonized BV-BRC genomic feature dataset.
No model refitting, feature selection, or hyperparameter optimization is performed during external validation.

Classification thresholds are retrieved from the corresponding NCBI model metadata to maintain consistency with the original
model development and decision-making procedures.

Objectives:

a. Assess the generalizability of NCBI-trained AMR prediction models using an independent BV-BRC genomic dataset.
b. Evaluate the predictive performance of four ML algorithms on external genomic data.
c. Maintain consistency between the NCBI training features and BV-BRC external validation features.
d. Apply the previously optimized and fitted NCBI models without refitting or retraining.
e. Retrieve and apply model-specific classification thresholds from the original NCBI model metadata.
f. Generate genome-level AMR predictions & prediction probabilities, where supported.
g. Evaluate model discrimination, classification performance,& predictive generalizability.
h. Calculate class-stratified bootstrap 95% confidence intervals (CIs)for external validation performance estimates.
i. Compare the external validation performance of the four ML algorithms.
j. Export performance metrics, prediction results, CIs, figures, and validation metadata.

ML models:

a. RF: Previously optimized and fitted using the NCBI development dataset.

b. XGB: Previously optimized and fitted using the NCBI development dataset.

c. LR: Previously optimized and fitted using the NCBI development dataset.

d. SVM: Previously optimized and fitted using the NCBI development dataset.

External validation principles:

a. Model development and training are performed exclusively using the NCBI development dataset.
b. Final fitted NCBI models are loaded from their saved model artifacts.
c. No model refitting or retraining is performed during external validation.
d. No additional feature selection is performed.
e. No hyperparameter optimization is performed.
f. The original NCBI feature definitions and ordering are maintained when preparing the external dataset.
g. Model-specific classification thresholds are obtained from the corresponding NCBI model metadata.
h. The BV-BRC dataset is used exclusively for independent external evaluation.
i. Class-stratified bootstrap resampling is performed to estimate 95% CIs for performance metrics.

Analytical workflow:

a. Load the original NCBI development dataset.
b. Load the independent BV-BRC external validation dataset.
c. Load the final fitted NCBI ML models.
d. Load the corresponding model metadata and classification thresholds.
e. Validate genome identifiers, classification labels, and genomic feature matrices.
f. Verify compatibility between the NCBI and BV-BRC feature definitions.
g. Align BV-BRC genomic features with the feature structure expected by each saved NCBI model.
h. Handle missing or incompatible features according to the predefined feature harmonization procedure.
i. Generate predictions using the previously fitted models without refitting.
j. Obtain prediction probabilities where supported by the respective classifiers.
k. Apply the corresponding NCBI-derived classification thresholds to generate binary predictions.
l. Calculate external validation performance metrics.
m. Perform class-stratified bootstrap resampling to obtain 95% CIs.
n. Compare the external predictive performance of RF,XGB, LR, & SVM.
o. Generate performance visualizations and genome-level prediction reports.
p. Export evaluation metrics, prediction tables, CIs, and validation metadata.

Input files:

a. NCBI_Final_ML_Dataset.tsv: Original NCBI genomic feature dataset used during model development and internal validation.

b. BVBRC_Final_ML_Dataset.tsv: Independent BV-BRC genomic feature dataset used exclusively for external validation.

c. Saved NCBI model artifacts: Previously fitted and optimized ML classifiers for RF, XGBoost, LR, and SVM.

d. NCBI model metadata: Corresponding model configurations, feature information,classification thresholds, and relevant training metadata.

Input data requirements:

Required metadata columns:
- Genome ID: Unique identifier for each bacterial genome.
- label: Binary antimicrobial resistance classification label.

Genomic feature columns:
- Genomic features must correspond to the feature definitions
  used during NCBI model development.
- Features must use the expected binary encoding.
- 0: Feature absent.
- 1: Feature present.

Classification encoding:
- 0: Susceptible.
- 1: Resistant.

Feature compatibility:
- NCBI and BV-BRC feature definitions must be harmonized.
- Feature names and ordering must be consistent with
  the saved model requirements.
- Missing features must be handled using the predefined
  feature alignment procedure.
- Additional external features not required by the trained
  models must not be used for prediction.

Bootstrap validation: Class-stratified bootstrap resampling is used to estimate 95% CIs for external validation metrics. 
The resampling procedure preserves the class structure of the external validation dataset.

Command-line arguments:

--ncbi:
    Path to the original NCBI development dataset.

--external:
    Path to the independent BV-BRC external validation dataset.

--model-dir:
    Directory containing the previously fitted NCBI models
    and corresponding model metadata.

--output:
    Directory for saving external validation results,
    figures, prediction tables, and evaluation reports.

Example execution:

python scripts/external_validation.py \
    --ncbi data/NCBI_Final_ML_Dataset.tsv \
    --external data/BVBRC_Final_ML_Dataset.tsv \
    --model-dir models/ \
    --output results/

Expected outputs:
- External validation performance metrics.
- Model-specific classification predictions.
- Genome-level prediction probabilities, where supported.
- Model-specific classification threshold summaries.
- Bootstrap 95% confidence intervals.
- RF external validation results.
- XGB external validation results.
- LR external validation results.
- SVM external validation results.
- Comparative model performance summaries.
- External validation figures.
- Genome-level prediction tables.
- Model evaluation reports.
- Reproducibility and validation metadata.

Reproducibility: Previously fitted NCBI models, model-specific classification thresholds, feature definitions, and 
preprocessing procedures are maintained during external validation.

No model refitting, feature selection, or hyperparameter optimization is performed using the BV-BRC dataset.

The bootstrap random seed, number of resampling iterations, classification thresholds, and evaluation parameters should
be documented to facilitate reproducibility.

Software requirements:
- Python 3.11
- Pandas
- NumPy
- SciPy
- Scikit-learn
- XGBoost
- Matplotlib
- Joblib

Installation:
pip install -r requirements.txt

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
import platform
import sys
import warnings

from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)


#Default settings

ID_COL = "Genome ID"
LABEL_COL = "label"

EXPECTED_FEATURES = 96
EXPECTED_THRESHOLD = 0.50

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

CONFIDENCE_LEVEL = 0.95

MODEL_FILES = {
    "RF": "RF_final_NCBI_development.pkl",
    "XGB": "XGB_final_NCBI_development.pkl",
    "LR": "LR_final_NCBI_development.pkl",
    "SVM": "SVM_final_NCBI_development.pkl",
}

METADATA_FILENAME = "NCBI_model_metadata.json"
FEATURE_NAMES_FILENAME = "NCBI_96_feature_names.pkl"

METRIC_NAMES = [
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1",
    "MCC",
    "ROC_AUC",
    "Average_Precision",
]


#Command-line arguments

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final NCBI-trained antimicrobial resistance "
            "prediction models using an external BV-BRC dataset."
        )
    )

    parser.add_argument(
        "--ncbi",
        type=Path,
        required=True,
        help="Full NCBI development dataset in TSV format."
    )

    parser.add_argument(
        "--external",
        type=Path,
        required=True,
        help="Independent BV-BRC external-validation dataset in TSV format."
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing saved NCBI models and metadata."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/external_validation"),
        help="Directory for external-validation outputs."
    )

    parser.add_argument(
        "--expected-features",
        type=int,
        default=EXPECTED_FEATURES,
        help="Expected number of model input features."
    )

    parser.add_argument(
        "--expected-threshold",
        type=float,
        default=EXPECTED_THRESHOLD,
        help="Expected classification threshold."
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of class-stratified bootstrap replicates."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for bootstrap sampling."
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=CONFIDENCE_LEVEL,
        help="Confidence level for bootstrap intervals."
    )

    parser.add_argument(
        "--font",
        type=str,
        default="Times New Roman",
        help="Font used in publication figures."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution of the saved PNG figure."
    )

    return parser.parse_args()


#Data validation

def check_binary_labels(y, dataset_name):
    """
    Require both classes:
        0 = Susceptible
        1 = Resistant
    """

    values = set(pd.Series(y).dropna().unique())

    if values != {0, 1}:

        raise ValueError(
            f"{dataset_name}: labels must contain both classes "
            f"0 and 1 only. Observed labels: {values}"
        )


def validate_dataset(df, dataset_name):
    """
    Validate genome identifiers, labels, and binary genomic features.
    """

    df = df.copy()

    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():

        duplicates = df.columns[df.columns.duplicated()].tolist()

        raise ValueError(
            f"{dataset_name}: duplicated column names: {duplicates}"
        )

    required = {ID_COL, LABEL_COL}

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df.empty:

        raise ValueError(f"{dataset_name} is empty.")

    if df[ID_COL].isna().any():

        raise ValueError(
            f"{dataset_name} contains missing Genome IDs."
        )

    if df[LABEL_COL].isna().any():

        raise ValueError(
            f"{dataset_name} contains missing labels."
        )

    genome_ids = df[ID_COL].astype(str).str.strip()

    if (genome_ids == "").any():

        raise ValueError(
            f"{dataset_name} contains empty Genome IDs."
        )

    if genome_ids.duplicated().any():

        duplicated = genome_ids[
            genome_ids.duplicated(keep=False)
        ].unique()

        raise ValueError(
            f"{dataset_name} contains duplicate Genome IDs. "
            f"Examples: {duplicated[:10].tolist()}"
        )

    df[ID_COL] = genome_ids

    y_numeric = pd.to_numeric(
        df[LABEL_COL],
        errors="raise"
    )

    if not np.isfinite(y_numeric.to_numpy(dtype=float)).all():

        raise ValueError(
            f"{dataset_name} contains non-finite labels."
        )

    if not np.equal(
        y_numeric.to_numpy(dtype=float),
        np.floor(y_numeric.to_numpy(dtype=float))
    ).all():

        raise ValueError(
            f"{dataset_name} contains non-integer labels."
        )

    y = y_numeric.astype(int)

    check_binary_labels(y, dataset_name)

    X = df.drop(columns=[ID_COL, LABEL_COL])

    if X.empty or X.shape[1] == 0:

        raise ValueError(
            f"{dataset_name} contains no genomic features."
        )

    if X.columns.duplicated().any():

        raise ValueError(
            f"{dataset_name} contains duplicate feature names."
        )

    X = X.apply(pd.to_numeric, errors="raise")

    if X.isna().any().any():

        raise ValueError(
            f"{dataset_name} contains missing feature values."
        )

    X_array = X.to_numpy(dtype=float)

    if not np.isfinite(X_array).all():

        raise ValueError(
            f"{dataset_name} contains non-finite feature values."
        )

    unique_values = set(np.unique(X_array))

    if not unique_values.issubset({0.0, 1.0}):

        raise ValueError(
            f"{dataset_name} genomic features must be binary 0/1. "
            f"Observed values: {sorted(unique_values)[:20]}"
        )

    return df, X.astype(np.uint8), y


#Genome overlap check

def check_genome_overlap(ncbi, external, output_dir):
    """
    Identify shared genome IDs between the NCBI development dataset
    and BV-BRC external-validation dataset.

    This check is important when describing the external cohort
    as independent.
    """

    ncbi_ids = set(ncbi[ID_COL].astype(str))

    external_ids = set(external[ID_COL].astype(str))

    shared_ids = sorted(ncbi_ids.intersection(external_ids))

    n_ncbi = len(ncbi_ids)
    n_external = len(external_ids)

    overlap_percentage = (
        100 * len(shared_ids) / n_external
        if n_external else 0
    )

    overlap_df = pd.DataFrame({
        "NCBI_Genomes": [n_ncbi],
        "External_Genomes": [n_external],
        "Shared_Genomes": [len(shared_ids)],
        "External_Genomes_Shared_Percentage": [
            overlap_percentage
        ],
        "No_Genome_Overlap": [len(shared_ids) == 0],
    })

    overlap_df.to_csv(
        output_dir / "Genome_Overlap_Summary.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame({
        ID_COL: shared_ids
    }).to_csv(
        output_dir / "Shared_Genome_IDs.tsv",
        sep="\t",
        index=False
    )

    print("\nGENOME OVERLAP CHECK")

    print(f"NCBI genomes: {n_ncbi}")
    print(f"External genomes: {n_external}")
    print(f"Shared genomes: {len(shared_ids)}")
    print(
        f"Shared external genomes: {overlap_percentage:.2f}%"
    )

    if shared_ids:

        print(
            "\nWARNING: Genome IDs overlap between the NCBI "
            "development dataset and BV-BRC validation dataset."
        )

        print(
            "Interpret the external-validation results cautiously. "
            "The shared genomes should be investigated before "
            "describing the cohort as fully independent."
        )

    else:

        print(
            "No shared genome IDs were detected."
        )

    return shared_ids


#Model metadata validation

def validate_model_metadata(
    metadata,
    saved_features,
    expected_features,
    expected_threshold
):
    """
    Validate the metadata generated during NCBI model development.
    """

    required_keys = [
        "classification_threshold",
        "negative_class",
        "positive_class",
        "negative_class_name",
        "positive_class_name",
        "expected_features",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in metadata
    ]

    if missing_keys:

        raise ValueError(
            "Model metadata is missing required entries: "
            f"{missing_keys}"
        )

    threshold = float(
        metadata["classification_threshold"]
    )

    if not np.isclose(threshold, expected_threshold):

        raise ValueError(
            f"Expected threshold {expected_threshold}, "
            f"but metadata contains {threshold}."
        )

    if not 0 <= threshold <= 1:

        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    if metadata["negative_class"] != 0:

        raise ValueError(
            "Metadata negative class must be 0."
        )

    if metadata["positive_class"] != 1:

        raise ValueError(
            "Metadata positive class must be 1."
        )

    if metadata["negative_class_name"] != "Susceptible":

        raise ValueError(
            "Metadata class 0 must be Susceptible."
        )

    if metadata["positive_class_name"] != "Resistant":

        raise ValueError(
            "Metadata class 1 must be Resistant."
        )

    if len(saved_features) != expected_features:

        raise ValueError(
            f"Expected {expected_features} saved features, "
            f"found {len(saved_features)}."
        )

    if len(set(saved_features)) != len(saved_features):

        raise ValueError(
            "Saved feature names contain duplicates."
        )

    if metadata["expected_features"] != len(saved_features):

        raise ValueError(
            "Feature count disagrees with model metadata."
        )

    return threshold


#Performance metrics

def calculate_metrics(y_true, y_prob, threshold):
    """
    Calculate external-validation classification metrics.
    """

    y_true = np.asarray(y_true, dtype=int)

    y_prob = np.asarray(y_prob, dtype=float)

    y_pred = (y_prob >= threshold).astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    fp = int(np.sum((y_true == 0) & (y_pred == 1)))

    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    metrics = {
        "N": len(y_true),

        "Positive_n": int((y_true == 1).sum()),

        "Negative_n": int((y_true == 0).sum()),

        "Threshold": float(threshold),

        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Sensitivity": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Specificity": specificity,

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            y_prob
        ),

        "Average_Precision": average_precision_score(
            y_true,
            y_prob
        ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    return metrics, y_pred


#Class-stratified bootstrap confidence intervals

def stratified_bootstrap_ci(
    y_true,
    y_prob,
    threshold,
    n_bootstrap,
    random_state,
    confidence_level
):
    """
    Calculate class-stratified percentile bootstrap confidence intervals.

    Both classes are sampled independently with replacement.
    The original class counts are preserved in each replicate.
    """

    y_true = np.asarray(y_true, dtype=int)

    y_prob = np.asarray(y_prob, dtype=float)

    rng = np.random.default_rng(random_state)

    idx_0 = np.where(y_true == 0)[0]

    idx_1 = np.where(y_true == 1)[0]

    if len(idx_0) == 0 or len(idx_1) == 0:

        raise ValueError(
            "Bootstrap confidence intervals require both classes."
        )

    values = {
        name: [] for name in METRIC_NAMES
    }

    for _ in range(n_bootstrap):

        sample = np.concatenate([
            rng.choice(
                idx_0,
                size=len(idx_0),
                replace=True
            ),

            rng.choice(
                idx_1,
                size=len(idx_1),
                replace=True
            ),
        ])

        yt = y_true[sample]

        yp = y_prob[sample]

        pred = (yp >= threshold).astype(int)

        tn = np.sum((yt == 0) & (pred == 0))

        fp = np.sum((yt == 0) & (pred == 1))

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

        values["Accuracy"].append(
            accuracy_score(yt, pred)
        )

        values["Precision"].append(
            precision_score(
                yt,
                pred,
                zero_division=0
            )
        )

        values["Sensitivity"].append(
            recall_score(
                yt,
                pred,
                zero_division=0
            )
        )

        values["Specificity"].append(specificity)

        values["F1"].append(
            f1_score(
                yt,
                pred,
                zero_division=0
            )
        )

        values["MCC"].append(
            matthews_corrcoef(yt, pred)
        )

        values["ROC_AUC"].append(
            roc_auc_score(yt, yp)
        )

        values["Average_Precision"].append(
            average_precision_score(yt, yp)
        )

    alpha = 1.0 - confidence_level

    lower_percentile = 100 * (alpha / 2)

    upper_percentile = 100 * (1 - alpha / 2)

    rows = []

    for name, scores in values.items():

        scores = np.asarray(scores, dtype=float)

        scores = scores[np.isfinite(scores)]

        if len(scores) == 0:

            lower = np.nan
            upper = np.nan

        else:

            lower = np.percentile(
                scores,
                lower_percentile
            )

            upper = np.percentile(
                scores,
                upper_percentile
            )

        rows.append({

            "Metric": name,

            "CI_lower": lower,

            "CI_upper": upper,

            "Confidence_Level": confidence_level,

            "Bootstrap_replicates_requested": n_bootstrap,

            "Bootstrap_replicates_valid": len(scores),
        })

    return pd.DataFrame(rows)


#Extract selected features from saved pipelines

def get_selected_features(
    model,
    feature_names,
    model_name
):
    """
    Extract feature names retained by the fitted variance filter
    and feature selector.
    """

    if not hasattr(model, "named_steps"):

        return None

    steps = model.named_steps

    if "variance" not in steps or "selector" not in steps:

        return None

    variance_support = steps["variance"].get_support()

    selector_support = steps["selector"].get_support()

    if len(variance_support) != len(feature_names):

        raise ValueError(
            f"{model_name}: variance-filter feature count mismatch."
        )

    after_variance = np.asarray(
        feature_names
    )[variance_support]

    if len(selector_support) != len(after_variance):

        raise ValueError(
            f"{model_name}: selector feature count mismatch."
        )

    return after_variance[selector_support]


#Save ROC, PR, & calibration figure

def save_combined_roc_pr_calibration(
    y_true,
    probabilities,
    output_dir,
    font,
    dpi
):
    """
    Generate a combined three-panel figure:

        A. ROC curves
        B. Precision-recall curves
        C. Calibration curves
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(28, 14)
    )

    #Panel A: ROC curve

    ax0 = axes[0]

    for name, prob in probabilities.items():

        fpr, tpr, _ = roc_curve(
            y_true,
            prob
        )

        auc = roc_auc_score(
            y_true,
            prob
        )

        ax0.plot(
            fpr,
            tpr,
            linewidth=2.2,
            label=f"{name} (AUC={auc:.3f})"
        )

    ax0.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        linewidth=1.8,
        label="Chance"
    )

    ax0.set_xlim(0, 1)

    ax0.set_ylim(0, 1)

    ax0.set_title(
        "ROC curve",
        fontname=font,
        fontsize=28,
        pad=14
    )

    ax0.set_xlabel(
        "False positive rate",
        fontname=font,
        fontsize=28,
        labelpad=10
    )

    ax0.set_ylabel(
        "True positive rate",
        fontname=font,
        fontsize=28,
        labelpad=10
    )

    #Panel B: PR curve

    ax1 = axes[1]

    prevalence = float(np.mean(y_true))

    for name, prob in probabilities.items():

        precision, recall, _ = precision_recall_curve(
            y_true,
            prob
        )

        ap = average_precision_score(
            y_true,
            prob
        )

        ax1.plot(
            recall,
            precision,
            linewidth=2.2,
            label=f"{name} (AP={ap:.3f})"
        )

    ax1.axhline(
        y=prevalence,
        linestyle="--",
        color="grey",
        linewidth=1.8,
        label=f"Prevalence={prevalence:.3f}"
    )

    ax1.set_xlim(0, 1)

    ax1.set_ylim(0, 1)

    ax1.set_title(
        "Precision-recall curve",
        fontname=font,
        fontsize=28,
        pad=14
    )

    ax1.set_xlabel(
        "Recall",
        fontname=font,
        fontsize=28,
        labelpad=10
    )

    ax1.set_ylabel(
        "Precision",
        fontname=font,
        fontsize=28,
        labelpad=10
    )

    #Panel C: calibration curve

    ax2 = axes[2]

    for name, prob in probabilities.items():

        observed, predicted = calibration_curve(
            y_true,
            prob,
            n_bins=10,
            strategy="quantile"
        )

        ax2.plot(
            predicted,
            observed,
            marker="o",
            markersize=7,
            linewidth=2.2,
            label=name
        )

    ax2.plot(
        [0, 1],
        [0, 1],
        "--",
        color="grey",
        linewidth=1.8,
        label="Perfect calibration"
    )

    ax2.set_xlim(0, 1)

    ax2.set_ylim(0, 1)

    ax2.set_title(
        "Calibration curve",
        fontname=font,
        fontsize=28,
        pad=14
    )

    ax2.set_xlabel(
        "Mean predicted probability",
        fontname=font,
        fontsize=28,
        labelpad=10
    )

    ax2.set_ylabel(
        "Observed fraction of positives",
        fontname=font,
        fontsize=28,
        labelpad=10
    )

    ##Formatting all panels

    for ax in axes:

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=26
        )

        for tick in (
            ax.get_xticklabels() +
            ax.get_yticklabels()
        ):

            tick.set_fontname(font)

            tick.set_fontsize(26)

        legend = ax.legend(
            loc="best",
            prop={
                "family": font,
                "size": 22
            }
        )

        for item in legend.get_texts():

            item.set_fontname(font)

            item.set_fontsize(22)

    ###Panel labels

    axes[0].text(
        -0.12,
        1.05,
        "A",
        transform=axes[0].transAxes,
        fontsize=28,
        fontweight="bold",
        fontname=font,
        va="top"
    )

    axes[1].text(
        -0.12,
        1.05,
        "B",
        transform=axes[1].transAxes,
        fontsize=28,
        fontweight="bold",
        fontname=font,
        va="top"
    )

    axes[2].text(
        -0.12,
        1.05,
        "C",
        transform=axes[2].transAxes,
        fontsize=28,
        fontweight="bold",
        fontname=font,
        va="top"
    )

    fig.tight_layout(
        pad=3.0,
        w_pad=3.0
    )

    png_path = (
        output_dir /
        "External_Validation_ROC_PR_Calibration_ABC.png"
    )

    pdf_path = (
        output_dir /
        "External_Validation_ROC_PR_Calibration_ABC.pdf"
    )

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nFigure saved: {png_path}")

    print(f"Figure saved: {pdf_path}")


#Feature consistency check

def check_feature_consistency(
    saved_features,
    X_ncbi,
    X_external,
    output_dir
):
    """
    Require exact feature-name agreement between the saved model,
    NCBI source data, and external BV-BRC data.

    Features are reordered to match the saved model feature order.
    No missing features are imputed and no extra features are discarded.
    """

    ncbi_features = list(X_ncbi.columns)

    external_features = list(X_external.columns)

    missing_ncbi = sorted(
        set(saved_features) - set(ncbi_features)
    )

    extra_ncbi = sorted(
        set(ncbi_features) - set(saved_features)
    )

    missing_external = sorted(
        set(saved_features) - set(external_features)
    )

    extra_external = sorted(
        set(external_features) - set(saved_features)
    )

    checks = pd.DataFrame({

        "Check": [
            "Expected feature count",
            "NCBI source feature count",
            "External feature count",
            "Missing NCBI features",
            "Extra NCBI features",
            "Missing external features",
            "Extra external features",
            "NCBI feature set exact match",
            "External feature set exact match",
        ],

        "Value": [
            len(saved_features),
            len(ncbi_features),
            len(external_features),
            len(missing_ncbi),
            len(extra_ncbi),
            len(missing_external),
            len(extra_external),
            set(saved_features) == set(ncbi_features),
            set(saved_features) == set(external_features),
        ]
    })

    checks.to_csv(
        output_dir / "Feature_Consistency_Check.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame({
        "Missing_NCBI_Feature": missing_ncbi
    }).to_csv(
        output_dir / "Missing_NCBI_Features.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame({
        "Extra_NCBI_Feature": extra_ncbi
    }).to_csv(
        output_dir / "Extra_NCBI_Features.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame({
        "Missing_External_Feature": missing_external
    }).to_csv(
        output_dir / "Missing_External_Features.tsv",
        sep="\t",
        index=False
    )

    pd.DataFrame({
        "Extra_External_Feature": extra_external
    }).to_csv(
        output_dir / "Extra_External_Features.tsv",
        sep="\t",
        index=False
    )

    if (
        missing_ncbi
        or extra_ncbi
        or missing_external
        or extra_external
    ):

        raise ValueError(
            "Feature consistency check failed. "
            "NCBI and external feature sets must exactly match "
            "the saved model feature set."
        )

    X_ncbi_ordered = X_ncbi.loc[:, saved_features]

    X_external_ordered = X_external.loc[:, saved_features]

    print("\nFEATURE CONSISTENCY CHECK PASSED")

    print(f"Expected features: {len(saved_features)}")

    print("NCBI features: exact match")

    print("External features: exact match")

    return X_ncbi_ordered, X_external_ordered


#Load & validate saved models

def load_final_models(
    model_dir,
    saved_features,
    external_X
):
    """
    Load final NCBI models without refitting.

    Verify model classes, predict_proba availability, and input
    feature compatibility before external prediction.
    """

    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = model_dir / filename

        if not model_path.is_file():

            raise FileNotFoundError(
                f"Final model missing for {model_name}:\n"
                f"{model_path}\n"
                "Run the NCBI model-development script first."
            )

        print(f"\nLoading model: {model_name}")

        model = joblib.load(model_path)

        if not hasattr(model, "classes_"):

            raise AttributeError(
                f"{model_name} does not expose classes_."
            )

        model_classes = np.asarray(model.classes_)

        if not np.array_equal(
            np.sort(model_classes),
            np.array([0, 1])
        ):

            raise ValueError(
                f"{model_name}: expected classes [0, 1], "
                f"found {model_classes}."
            )

        if not hasattr(model, "predict_proba"):

            raise AttributeError(
                f"{model_name} does not provide predict_proba()."
            )

        if hasattr(model, "n_features_in_"):

            if model.n_features_in_ != len(saved_features):

                raise ValueError(
                    f"{model_name}: expected "
                    f"{model.n_features_in_} input features, "
                    f"but the saved feature list contains "
                    f"{len(saved_features)}."
                )

        if hasattr(model, "feature_names_in_"):

            fitted_features = list(model.feature_names_in_)

            if fitted_features != list(saved_features):

                raise ValueError(
                    f"{model_name}: fitted feature order "
                    "does not match saved feature order."
                )

        ##Test input compatibility without using external labels
        test_probabilities = np.asarray(
            model.predict_proba(external_X.iloc[:1]),
            dtype=float
        )

        if test_probabilities.shape != (1, 2):

            raise ValueError(
                f"{model_name}: unexpected probability output "
                f"shape {test_probabilities.shape}."
            )

        loaded_models[model_name] = model

    return loaded_models


#Evaluate a single model

def evaluate_model(
    model_name,
    model,
    external,
    X_external,
    y_external,
    saved_features,
    threshold,
    output_dir,
    n_bootstrap,
    random_state,
    confidence_level
):
    """
    Perform external prediction and evaluation for one fitted model.

    No refitting, feature selection, or tuning is performed.
    """

    print("\n" + "=" * 70)

    print(f"EXTERNAL VALIDATION: {model_name}")

    print("=" * 70)

    ##Extract selected features

    selected = get_selected_features(
        model,
        saved_features,
        model_name
    )

    if selected is not None:

        pd.DataFrame({

            "Selected_Feature": selected

        }).to_csv(
            output_dir / f"{model_name}_Selected_Features.tsv",
            sep="\t",
            index=False
        )

        print(
            f"Selected features in fitted pipeline: {len(selected)}"
        )

    ##Generate external probabilities

    model_classes = np.asarray(model.classes_)

    positive_index = np.where(model_classes == 1)[0][0]

    probability_matrix = np.asarray(
        model.predict_proba(X_external),
        dtype=float
    )

    expected_shape = (
        len(X_external),
        len(model_classes)
    )

    if probability_matrix.shape != expected_shape:

        raise ValueError(
            f"{model_name}: unexpected predict_proba shape. "
            f"Expected {expected_shape}, "
            f"found {probability_matrix.shape}."
        )

    probabilities = probability_matrix[:, positive_index]

    if not np.isfinite(probabilities).all():

        raise ValueError(
            f"{model_name}: predicted probabilities "
            "contain non-finite values."
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
    ):

        raise ValueError(
            f"{model_name}: predicted probabilities "
            "fall outside [0, 1]."
        )

    ##Calculate metrics

    metrics, predictions = calculate_metrics(
        y_external,
        probabilities,
        threshold
    )

    metrics["Model"] = model_name

    ##Per-genome prediction output

    prediction_df = pd.DataFrame({

        ID_COL: external[ID_COL].astype(str),

        "True_Label": y_external.to_numpy(),

        "Predicted_Probability_Resistant": probabilities,

        f"Predicted_Label_at_{threshold:.2f}": predictions,

        "Correct_Classification": (
            predictions == y_external.to_numpy()
        ),
    })

    prediction_df.to_csv(
        output_dir / f"{model_name}_External_Predictions.tsv",
        sep="\t",
        index=False
    )

    diagnostic = prediction_df.rename(
        columns={
            "True_Label": "Actual_Label",
            "Predicted_Probability_Resistant":
                "Predicted_Probability",
        }
    )

    diagnostic.to_csv(
        output_dir / f"{model_name}_Probability_Diagnostics.tsv",
        sep="\t",
        index=False
    )

    ##Bootstrap confidence intervals

    ci = stratified_bootstrap_ci(

        y_true=y_external,

        y_prob=probabilities,

        threshold=threshold,

        n_bootstrap=n_bootstrap,

        random_state=random_state,

        confidence_level=confidence_level
    )

    ci.insert(
        0,
        "Model",
        model_name
    )

    ##Print results

    print("\nPredicted class counts:")

    print(
        pd.Series(predictions)
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(
            index={
                0: "Susceptible (0)",
                1: "Resistant (1)"
            }
        )
    )

    print("\nExternal-validation metrics:")

    print(
        pd.Series(metrics).to_string()
    )

    print("\nBootstrap confidence intervals:")

    print(
        ci.to_string(index=False)
    )

    return metrics, ci, probabilities, predictions


#Feature prevalence shift

def calculate_prevalence_shift(
    X_ncbi,
    X_external,
    saved_features,
    output_dir
):
    """
    Calculate descriptive differences in feature prevalence.

    This is descriptive only. No model fitting or feature
    selection is performed.
    """

    prevalence = pd.DataFrame({

        "Feature": saved_features,

        "NCBI_Development_Prevalence":
            X_ncbi.mean(axis=0).to_numpy(),

        "External_Prevalence":
            X_external.mean(axis=0).to_numpy(),
    })

    prevalence["External_minus_NCBI"] = (
        prevalence["External_Prevalence"]
        - prevalence["NCBI_Development_Prevalence"]
    )

    prevalence["Absolute_Difference"] = (
        prevalence["External_minus_NCBI"].abs()
    )

    prevalence.sort_values(
        "Absolute_Difference",
        ascending=False
    ).to_csv(
        output_dir / "Feature_Prevalence_Shift.tsv",
        sep="\t",
        index=False
    )

    return prevalence


#Combine performance & confidence intervals

def combine_performance_and_ci(
    metrics_df,
    ci_df,
    threshold
):
    """
    Combine point estimates and bootstrap confidence intervals
    into a publication-friendly table.
    """

    combined = []

    for _, row in metrics_df.iterrows():

        model_name = row["Model"]

        model_ci = (
            ci_df[
                ci_df["Model"] == model_name
            ]
            .set_index("Metric")
        )

        output = {

            "Model": model_name,

            "Threshold": threshold,

            "N": int(row["N"]),
        }

        for metric in METRIC_NAMES:

            output[metric] = row[metric]

            lower = model_ci.loc[
                metric,
                "CI_lower"
            ]

            upper = model_ci.loc[
                metric,
                "CI_upper"
            ]

            if pd.notna(lower) and pd.notna(upper):

                output[f"{metric}_95CI"] = (
                    f"{lower:.3f}-{upper:.3f}"
                )

            else:

                output[f"{metric}_95CI"] = "NA"

        combined.append(output)

    return pd.DataFrame(combined)


#Save run metadata

def save_run_metadata(
    args,
    threshold,
    saved_features,
    ncbi,
    external,
    y_external,
    shared_genomes,
    all_probabilities,
    output_dir
):
    """
    Save reproducibility information for the external-validation run.
    """

    try:

        import matplotlib

        matplotlib_version = matplotlib.__version__

    except ImportError:

        matplotlib_version = "Unavailable"

    metadata = {

        "Run_timestamp_UTC": datetime.now(
            timezone.utc
        ).isoformat(),

        "Python_version": platform.python_version(),

        "Platform": platform.platform(),

        "NumPy_version": np.__version__,

        "Pandas_version": pd.__version__,

        "Scikit_learn_version": sklearn.__version__,

        "Joblib_version": joblib.__version__,

        "Matplotlib_version": matplotlib_version,

        "NCBI_source_file": str(args.ncbi.resolve()),

        "External_validation_file": str(
            args.external.resolve()
        ),

        "Model_directory": str(
            args.model_dir.resolve()
        ),

        "Output_directory": str(
            output_dir.resolve()
        ),

        "Threshold": threshold,

        "Threshold_source": METADATA_FILENAME,

        "Threshold_optimisation_performed": False,

        "Model_refitting_performed": False,

        "Feature_selection_performed": False,

        "Hyperparameter_tuning_performed": False,

        "External_data_used_for_training": False,

        "NCBI_samples_loaded_for_description": len(ncbi),

        "External_samples": len(external),

        "External_positive": int(
            (y_external == 1).sum()
        ),

        "External_negative": int(
            (y_external == 0).sum()
        ),

        "Shared_genome_count": len(shared_genomes),

        "Genome_overlap_detected": (
            len(shared_genomes) > 0
        ),

        "Feature_count": len(saved_features),

        "Bootstrap_replicates": args.bootstrap,

        "Bootstrap_method": (
            "Class-stratified percentile bootstrap"
        ),

        "Confidence_interval": (
            f"{args.confidence_level * 100:.1f}%"
        ),

        "Random_state": args.random_state,

        "Models_evaluated": list(
            all_probabilities.keys()
        ),

        "Model_files": MODEL_FILES,

        "Negative_class": 0,

        "Negative_class_name": "Susceptible",

        "Positive_class": 1,

        "Positive_class_name": "Resistant",

        "Evaluation_dataset": "BV-BRC",

        "Validation_type": "External validation",
    }

    with open(
        output_dir / "External_Validation_Run_Metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


#Main workflow

def main():

    args = parse_arguments()

    ##Validate arguments

    if not 0 <= args.expected_threshold <= 1:

        raise ValueError(
            "--expected-threshold must be between 0 and 1."
        )

    if args.expected_features < 1:

        raise ValueError(
            "--expected-features must be at least 1."
        )

    if args.bootstrap < 100:

        raise ValueError(
            "At least 100 bootstrap replicates are recommended."
        )

    if not 0 < args.confidence_level < 1:

        raise ValueError(
            "--confidence-level must be between 0 and 1."
        )

    if args.dpi < 100:

        raise ValueError(
            "--dpi should be at least 100."
        )

    ##Create output directory

    output_dir = args.output

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    ##Required file checks

    metadata_file = (
        args.model_dir / METADATA_FILENAME
    )

    feature_names_file = (
        args.model_dir / FEATURE_NAMES_FILENAME
    )

    required_files = [
        args.ncbi,
        args.external,
        metadata_file,
        feature_names_file,
    ]

    for path in required_files:

        if not path.is_file():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    ##Load metadata

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        model_metadata = json.load(file)

    saved_features = list(
        joblib.load(feature_names_file)
    )

    threshold = validate_model_metadata(
        metadata=model_metadata,
        saved_features=saved_features,
        expected_features=args.expected_features,
        expected_threshold=args.expected_threshold
    )

    print("\nMODEL CONFIGURATION")

    print(f"Classification threshold: {threshold:.2f}")

    print(f"Number of saved features: {len(saved_features)}")

    print("Class 0 = Susceptible")

    print("Class 1 = Resistant")

    ##Load input datasets

    print("\nLoading NCBI and BV-BRC datasets...")

    ncbi_raw = pd.read_csv(
        args.ncbi,
        sep="\t"
    )

    external_raw = pd.read_csv(
        args.external,
        sep="\t"
    )

    ncbi, X_ncbi, y_ncbi = validate_dataset(
        ncbi_raw,
        "Full NCBI development"
    )

    external, X_external_raw, y_external = validate_dataset(
        external_raw,
        "External BV-BRC validation"
    )

    ##Check dataset sizes and classes

    print("\nDATASET SUMMARY")

    print(
        f"Full NCBI development genomes: {len(ncbi)}"
    )

    print(
        f"External validation genomes: {len(external)}"
    )

    print(
        f"Number of genomic features: {len(saved_features)}"
    )

    print("\nNCBI class counts:")

    print(
        y_ncbi.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Susceptible (0)",
                1: "Resistant (1)"
            }
        )
    )

    print("\nExternal class counts:")

    print(
        y_external.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Susceptible (0)",
                1: "Resistant (1)"
            }
        )
    )

    ##Genome overlap check

    shared_genomes = check_genome_overlap(
        ncbi,
        external,
        output_dir
    )

    ##Feature consistency check

    X_ncbi, X_external = check_feature_consistency(
        saved_features,
        X_ncbi,
        X_external_raw,
        output_dir
    )

    ##Descriptive feature prevalence comparison

    calculate_prevalence_shift(
        X_ncbi,
        X_external,
        saved_features,
        output_dir
    )

    ##Load final NCBI models

    print("\nLoading final NCBI-trained models...")

    models = load_final_models(
        model_dir=args.model_dir,
        saved_features=saved_features,
        external_X=X_external
    )

    ##External validation

    all_metrics = []

    all_ci = []

    all_probabilities = {}

    all_predictions = {}

    for model_name, model in models.items():

        metrics, ci, probabilities, predictions = (
            evaluate_model(

                model_name=model_name,

                model=model,

                external=external,

                X_external=X_external,

                y_external=y_external,

                saved_features=saved_features,

                threshold=threshold,

                output_dir=output_dir,

                n_bootstrap=args.bootstrap,

                random_state=args.random_state,

                confidence_level=args.confidence_level,
            )
        )

        all_metrics.append(metrics)

        all_ci.append(ci)

        all_probabilities[model_name] = probabilities

        all_predictions[model_name] = predictions

    ##Consolidated performance table

    metrics_df = pd.DataFrame(all_metrics)

    metric_columns = [
        "Model",
        "N",
        "Positive_n",
        "Negative_n",
        "Threshold",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1",
        "MCC",
        "ROC_AUC",
        "Average_Precision",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    metrics_df = metrics_df[metric_columns]

    metrics_df.to_csv(
        output_dir / "External_Validation_Performance.tsv",
        sep="\t",
        index=False
    )

    ##Consolidated confidence intervals

    ci_df = pd.concat(
        all_ci,
        ignore_index=True
    )

    ci_df.to_csv(
        output_dir / "External_Validation_95CI.tsv",
        sep="\t",
        index=False
    )

    #Performance estimates with CIs

    combined_df = combine_performance_and_ci(
        metrics_df,
        ci_df,
        threshold
    )

    combined_df.to_csv(
        output_dir /
        "External_Validation_Performance_with_CI.tsv",
        sep="\t",
        index=False
    )

    ##Combined ROC, PR, & calibration figure

    save_combined_roc_pr_calibration(
        y_true=y_external,
        probabilities=all_probabilities,
        output_dir=output_dir,
        font=args.font,
        dpi=args.dpi
    )

    ##Save run metadata

    save_run_metadata(
        args=args,
        threshold=threshold,
        saved_features=saved_features,
        ncbi=ncbi,
        external=external,
        y_external=y_external,
        shared_genomes=shared_genomes,
        all_probabilities=all_probabilities,
        output_dir=output_dir
    )

    ##Final report

    print("\n" + "=" * 70)

    print("EXTERNAL VALIDATION COMPLETED")

    print("=" * 70)

    print(f"Fixed threshold: {threshold:.2f}")

    print(f"Bootstrap replicates: {args.bootstrap}")

    print(f"Number of models evaluated: {len(models)}")

    print(f"Shared genomes: {len(shared_genomes)}")

    print(f"Results directory: {output_dir.resolve()}")

    print("\nEXTERNAL VALIDATION PERFORMANCE")

    print(
        metrics_df.to_string(index=False)
    )

    print("\nPERFORMANCE WITH 95% CONFIDENCE INTERVALS")

    print(
        combined_df.to_string(index=False)
    )

    print("\nIMPORTANT INTERPRETATION NOTE")

    print(
        "These results represent evaluation of the saved "
        "NCBI-trained models on the BV-BRC dataset."
    )

    print(
        "No model refitting, feature reselection, "
        "hyperparameter tuning, or threshold optimisation "
        "was performed using the external dataset."
    )

    if shared_genomes:

        print(
            "Genome overlap was detected. Investigate shared "
            "genomes before describing the validation cohort "
            "as fully independent."
        )

    else:

        print(
            "No genome ID overlap was detected between "
            "the NCBI and BV-BRC datasets."
        )

    print("\nExternal validation workflow finished successfully.")


#Script entry point

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr
        )

        raise

