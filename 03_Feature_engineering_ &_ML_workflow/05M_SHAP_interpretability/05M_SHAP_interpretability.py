#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
SHAP-based and multi-model biomarker interpretability analysis of NCBI AMR prediction models

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description: This script performs genomic feature interpretation and candidate biomarker identification 
using four previously trained NCBI antimicrobial resistance prediction models.

Interpretation methods:
    - Random Forest (RF): TreeSHAP.
    - XGBoost (XGB): TreeSHAP.
    - Logistic Regression (LR): Absolute regression coefficients.
    - Support Vector Machine (SVM): Permutation importance
      using ROC-AUC.

Main operations:
a. Reconstruct the original stratified training/test split.
b. Load saved models without refitting.
c. Extract model-specific selected genomic features.
d. Calculate model-specific feature importance.
e. Identify consensus candidate biomarkers across models.
f. Assess agreement between model importance rankings.
g. Generate publication-quality interpretability figures.
h. Export feature importance tables and analysis metadata.

Input files:
- Original NCBI machine-learning dataset.
- Four previously trained model files (RF, XGB, LR and SVM).

Classification:
    0 = Susceptible
    1 = Resistant

Interpretation dataset:
    Original NCBI internal 20% test set.

Requirements:
    Python 3.11
    pandas
    numpy
    scikit-learn
    shap
    xgboost
    matplotlib
    joblib

Execution:
python 05M_SHAP_interpretability.py \
    --input data/NCBI_Final_ML_Dataset.tsv \
    --model-dir models/ \
    --output results/biomarker_analysis/

Important: Saved models are loaded without refitting. The original training/test split must be reproducible,
including the original dataset row order and preprocessing.

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
import logging
import warnings

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


#Configuration

RANDOM_STATE = 42

DEFAULT_TEST_SIZE = 0.20

EXPECTED_FEATURES = 96

TOP_N = 20

TOP_N_COMPARISON = 10

N_PERMUTATIONS = 20

CLASSIFICATION_THRESHOLD = 0.50

MODEL_NAMES = ("RF", "XGB", "LR", "SVM")

ID_COL = "Genome ID"

LABEL_COL = "label"

METADATA_COLUMNS = {ID_COL, LABEL_COL}


#Logging & warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


#Plotting style

def configure_plotting() -> None:
    """Configure publication-style Matplotlib settings."""

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 26,
        "axes.titlesize": 28,
        "axes.titleweight": "bold",
        "axes.labelsize": 26,
        "axes.labelweight": "bold",
        "legend.fontsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


#Command-line arguments

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Perform SHAP and biomarker interpretability analysis "
            "using internally trained NCBI classification models."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Original NCBI TSV dataset used during model development.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the four saved internal models.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/biomarker_analysis"),
        help="Output directory.",
    )

    parser.add_argument(
        "--expected-features",
        type=int,
        default=EXPECTED_FEATURES,
        help="Expected number of original gene-family features.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test-set proportion used during model development.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used during model development.",
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=N_PERMUTATIONS,
        help="Number of SVM permutation-importance repeats.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Number of consensus biomarkers to report.",
    )

    parser.add_argument(
        "--top-n-comparison",
        type=int,
        default=TOP_N_COMPARISON,
        help="Number of biomarkers used in the model comparison plot.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CLASSIFICATION_THRESHOLD,
        help="Classification threshold recorded for the analysis.",
    )

    parser.add_argument(
        "--shap-background-size",
        type=int,
        default=100,
        help=(
            "Maximum number of training observations used as "
            "the SHAP background dataset where applicable."
        ),
    )

    return parser.parse_args()


#Create output directories

def create_output_directories(output_dir: Path) -> None:
    """Create output directory structure."""

    directories = [
        output_dir,
        output_dir / "feature_importance",
        output_dir / "feature_selection",
        output_dir / "consensus",
        output_dir / "figures",
        output_dir / "metadata",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


#Validate input files

def validate_input_files(
    input_file: Path,
    model_dir: Path,
) -> dict[str, Path]:

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input dataset not found: {input_file}"
        )

    model_files = {
        "RF": model_dir / "RF_best_internal_model.pkl",
        "XGB": model_dir / "XGB_best_internal_model.pkl",
        "LR": model_dir / "LR_best_internal_model.pkl",
        "SVM": model_dir / "SVM_best_internal_model.pkl",
    }

    for model_name, model_path in model_files.items():
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Saved model missing for {model_name}: {model_path}"
            )

    return model_files


#Identify gene features

def identify_gene_features(
    columns: pd.Index,
) -> list[str]:
    """
    Identify gene features.

    Excludes:
        Genome ID
        label
        Columns beginning with group_ (case-insensitive)
    """

    return [
        str(column)
        for column in columns
        if str(column) not in METADATA_COLUMNS
        and not str(column).lower().startswith("group_")
    ]


#Load & validate dataset

def load_dataset(
    input_file: Path,
    expected_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    logger.info("Loading NCBI dataset: %s", input_file)

    df = pd.read_csv(input_file, sep="\t")

    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    required_columns = {ID_COL, LABEL_COL}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df[ID_COL].isna().any():
        raise ValueError("Missing Genome IDs detected.")

    if df[ID_COL].astype(str).duplicated().any():
        raise ValueError("Duplicate Genome IDs detected.")

    feature_columns = identify_gene_features(df.columns)

    if len(feature_columns) != expected_features:
        raise ValueError(
            f"Expected {expected_features} gene features, "
            f"found {len(feature_columns)}."
        )

    X = df.loc[:, feature_columns].copy()

    y = pd.to_numeric(
        df[LABEL_COL],
        errors="raise",
    )

    if y.isna().any():
        raise ValueError("Missing classification labels detected.")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite classification labels detected.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    y = y.astype(int)

    if y.nunique() != 2:
        raise ValueError("Both classification classes must be present.")

    X = X.apply(
        pd.to_numeric,
        errors="raise",
    )

    if X.isna().any().any():
        raise ValueError("Missing gene-feature values detected.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite gene-feature values detected.")

    if not X.isin([0, 1]).all().all():
        raise ValueError("Gene features must be binary 0/1.")

    X = X.astype(np.uint8)

    logger.info("Dataset loaded successfully.")
    logger.info("Genome count: %d", len(df))
    logger.info("Feature count: %d", X.shape[1])

    return df, X, y


#Recreate stratified train/test split

def recreate_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    logger.info("Recreating stratified training/test split.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_df = df.loc[X_train.index].copy()

    test_df = df.loc[X_test.index].copy()

    logger.info("Training genomes: %d", len(X_train))
    logger.info("Internal test genomes: %d", len(X_test))

    logger.info(
        "Training class counts: %s",
        y_train.value_counts().sort_index().to_dict(),
    )

    logger.info(
        "Test class counts: %s",
        y_test.value_counts().sort_index().to_dict(),
    )

    return (
        train_df,
        test_df,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


#Load saved models

def load_models(
    model_files: dict[str, Path],
    expected_features: int,
) -> dict[str, Any]:

    models = {}

    required_steps = {"variance", "selector", "clf"}

    for model_name, model_path in model_files.items():

        logger.info("Loading %s model.", model_name)

        model = joblib.load(model_path)

        if not hasattr(model, "named_steps"):
            raise ValueError(
                f"{model_name} is not a sklearn Pipeline."
            )

        missing_steps = required_steps - set(model.named_steps)

        if missing_steps:
            raise ValueError(
                f"{model_name} pipeline is missing steps: "
                f"{sorted(missing_steps)}"
            )

        if not hasattr(model, "n_features_in_"):
            raise ValueError(
                f"{model_name} does not expose n_features_in_."
            )

        if model.n_features_in_ != expected_features:
            raise ValueError(
                f"{model_name} was trained with "
                f"{model.n_features_in_} input features, "
                f"not {expected_features}."
            )

        if not hasattr(model, "classes_"):
            raise ValueError(
                f"{model_name} does not expose classes_."
            )

        if not np.array_equal(
            np.sort(model.classes_),
            np.array([0, 1]),
        ):
            raise ValueError(
                f"{model_name} does not have binary classes [0, 1]."
            )

        models[model_name] = model

        logger.info(
            "%s pipeline steps: %s",
            model_name,
            list(model.named_steps.keys()),
        )

    return models


#Extract selected features

def extract_pipeline_features(
    pipeline: Any,
    original_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:

    variance = pipeline.named_steps["variance"]

    selector = pipeline.named_steps["selector"]

    variance_mask = variance.get_support()

    selector_mask = selector.get_support()

    if len(variance_mask) != len(original_features):
        raise ValueError(
            "Variance-filter feature count does not match "
            "the original feature count."
        )

    variance_features = np.asarray(original_features)[variance_mask]

    if len(selector_mask) != len(variance_features):
        raise ValueError(
            "Selector feature count does not match "
            "variance-filtered feature count."
        )

    selected_features = variance_features[selector_mask]

    if len(selected_features) == 0:
        raise ValueError("No features remain after feature selection.")

    return variance_features, selected_features


#Normalize importance

def normalize_importance(
    importance_values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        importance_values,
        dtype=float,
    ).copy()

    values[~np.isfinite(values)] = 0

    ##Negative permutation importance is retained in the raw output but not treated as positive importance
    values = np.maximum(values, 0)

    maximum = np.max(values) if len(values) else 0

    if maximum > 0:
        return values / maximum

    return values


#Save feature selection results

def save_feature_selection(
    models: dict[str, Any],
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, np.ndarray]:

    logger.info("Extracting model-specific selected features.")

    selected_features = {}

    summary = []

    for model_name, model in models.items():

        variance_features, selected = extract_pipeline_features(
            model,
            feature_names,
        )

        selected_features[model_name] = selected

        summary.append({
            "Model": model_name,
            "Initial_features": len(feature_names),
            "After_variance_filter": len(variance_features),
            "SelectKBest_features": len(selected),
        })

        pd.DataFrame({
            "Feature": selected,
            "Rank": np.arange(1, len(selected) + 1),
        }).to_csv(
            output_dir / "feature_selection"
            / f"{model_name}_Selected_Features.tsv",
            sep="\t",
            index=False,
        )

    pd.DataFrame(summary).to_csv(
        output_dir / "feature_selection"
        / "Model_Feature_Selection_Summary.tsv",
        sep="\t",
        index=False,
    )

    return selected_features


#Extract positive-class SHAP values

def extract_positive_class_shap_values(
    shap_output: Any,
    n_features: int,
    positive_class_index: int = 1,
) -> np.ndarray:
    """
    Extract positive-class SHAP values.

    Handles common SHAP output formats:
        - list of arrays by class
        - Explanation objects
        - samples x features
        - samples x features x classes
        - classes x samples x features
    """

    if isinstance(shap_output, list):

        if len(shap_output) <= positive_class_index:
            raise ValueError("Positive-class SHAP output is missing.")

        values = np.asarray(
            shap_output[positive_class_index]
        )

    elif hasattr(shap_output, "values"):

        values = np.asarray(shap_output.values)

    else:

        values = np.asarray(shap_output)

    if values.ndim == 3:

        if values.shape[1] == n_features:
            values = values[:, :, positive_class_index]

        elif values.shape[2] == n_features:
            values = values[positive_class_index, :, :]

        else:
            raise ValueError(
                f"Unexpected SHAP array shape: {values.shape}"
            )

    if values.ndim != 2:
        raise ValueError(
            f"Expected two-dimensional SHAP values, got {values.shape}"
        )

    if values.shape[1] != n_features:
        raise ValueError(
            f"SHAP feature mismatch: {values.shape[1]} versus "
            f"{n_features}."
        )

    if not np.isfinite(values).all():
        raise ValueError("SHAP output contains non-finite values.")

    return values


#Save importance table

def create_importance_table(
    features: np.ndarray,
    raw_importance: np.ndarray,
    model_name: str,
    output_dir: Path,
    filename: str,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:

    features = np.asarray(features)

    raw_importance = np.asarray(raw_importance, dtype=float)

    if len(features) != len(raw_importance):
        raise ValueError(
            f"{model_name}: feature and importance lengths differ."
        )

    table = pd.DataFrame({
        "Feature": features,
        "Raw_Importance": raw_importance,
        "Importance": normalize_importance(raw_importance),
    })

    if extra_columns:
        for column_name, values in extra_columns.items():

            if len(values) != len(features):
                raise ValueError(
                    f"{model_name}: invalid length for {column_name}."
                )

            table[column_name] = values

    table = table.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)

    table.to_csv(
        output_dir / "feature_importance" / filename,
        sep="\t",
        index=False,
    )

    return table


#RF tree SHAP

def analyze_random_forest(
    model: Any,
    X_test: pd.DataFrame,
    features: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:

    logger.info("Calculating Random Forest TreeSHAP.")

    variance = model.named_steps["variance"]

    selector = model.named_steps["selector"]

    classifier = model.named_steps["clf"]

    X_transformed = selector.transform(
        variance.transform(X_test)
    )

    explainer = shap.TreeExplainer(classifier)

    shap_output = explainer.shap_values(X_transformed)

    shap_values = extract_positive_class_shap_values(
        shap_output,
        len(features),
    )

    importance = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    table = create_importance_table(
        features=features,
        raw_importance=importance,
        model_name="RF",
        output_dir=output_dir,
        filename="RF_SHAP_Feature_Importance.tsv",
    )

    logger.info("Random Forest SHAP completed.")

    return table[["Feature", "Importance"]]


#XGB tree SHAP

def analyze_xgboost(
    model: Any,
    X_test: pd.DataFrame,
    features: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:

    logger.info("Calculating XGBoost TreeSHAP.")

    variance = model.named_steps["variance"]

    selector = model.named_steps["selector"]

    classifier = model.named_steps["clf"]

    X_transformed = selector.transform(
        variance.transform(X_test)
    )

    explainer = shap.TreeExplainer(classifier)

    shap_output = explainer.shap_values(X_transformed)

    shap_values = extract_positive_class_shap_values(
        shap_output,
        len(features),
    )

    importance = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    table = create_importance_table(
        features=features,
        raw_importance=importance,
        model_name="XGB",
        output_dir=output_dir,
        filename="XGB_SHAP_Feature_Importance.tsv",
    )

    logger.info("XGBoost SHAP completed.")

    return table[["Feature", "Importance"]]


#LR importance

def analyze_logistic_regression(
    model: Any,
    features: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:

    logger.info("Extracting Logistic Regression coefficients.")

    classifier = model.named_steps["clf"]

    coefficients = np.asarray(classifier.coef_)

    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise ValueError(
            f"Unexpected logistic regression coefficient shape: "
            f"{coefficients.shape}"
        )

    coefficients = coefficients[0]

    if len(coefficients) != len(features):
        raise ValueError(
            "Logistic Regression coefficient count does not match "
            "selected feature count."
        )

    table = create_importance_table(
        features=features,
        raw_importance=np.abs(coefficients),
        model_name="LR",
        output_dir=output_dir,
        filename="LR_Coefficient_Feature_Importance.tsv",
        extra_columns={
            "Coefficient": coefficients,
            "Absolute_Coefficient": np.abs(coefficients),
        },
    )

    logger.info("Logistic Regression analysis completed.")

    return table[["Feature", "Importance"]]


#SVM permutation importance

def analyze_svm(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: np.ndarray,
    n_permutations: int,
    random_state: int,
    output_dir: Path,
) -> pd.DataFrame:

    logger.info("Calculating SVM permutation importance.")

    variance = model.named_steps["variance"]

    selector = model.named_steps["selector"]

    classifier = model.named_steps["clf"]

    X_transformed = selector.transform(
        variance.transform(X_test)
    )

    ##This is the classifier's input space after all preprocessing performed in the saved pipeline

    ##If scaling is a separate pipeline step, it must also be applied here. The model below assumes scaling is not an additional step between selector and clf

    if not hasattr(classifier, "predict_proba") and not hasattr(
        classifier, "decision_function"
    ):
        raise ValueError(
            "SVM must support predict_proba or decision_function "
            "for ROC-AUC permutation importance."
        )

    permutation = permutation_importance(
        estimator=classifier,
        X=X_transformed,
        y=y_test,
        scoring="roc_auc",
        n_repeats=n_permutations,
        random_state=random_state,
        n_jobs=-1,
    )

    table = create_importance_table(
        features=features,
        raw_importance=permutation.importances_mean,
        model_name="SVM",
        output_dir=output_dir,
        filename="SVM_Permutation_Feature_Importance.tsv",
        extra_columns={
            "Permutation_Importance_Mean": permutation.importances_mean,
            "Permutation_Importance_SD": permutation.importances_std,
        },
    )

    logger.info("SVM permutation analysis completed.")

    return table[["Feature", "Importance"]]


#Consensus biomarker analysis

def calculate_consensus(
    importance_tables: dict[str, pd.DataFrame],
    output_dir: Path,
    top_n: int,
    top_n_comparison: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logger.info("Calculating consensus biomarker rankings.")

    renamed = []

    for model_name in MODEL_NAMES:

        table = importance_tables[model_name].rename(
            columns={"Importance": model_name}
        )

        renamed.append(table)

    consensus = renamed[0]

    for table in renamed[1:]:
        consensus = consensus.merge(
            table,
            on="Feature",
            how="outer",
        )

    importance_columns = list(MODEL_NAMES)

    consensus[importance_columns] = (
        consensus[importance_columns]
        .fillna(0)
        .astype(float)
    )

    ##Sum of model-specific normalized importance scores
    consensus["Consensus"] = consensus[
        importance_columns
    ].sum(axis=1)

    ##Count models with nonzero importance
    ##This is not a feature-selection frequency
    consensus["Models_With_Nonzero_Importance"] = (
        consensus[importance_columns] > 0
    ).sum(axis=1)

    consensus = consensus.sort_values(
        [
            "Consensus",
            "Models_With_Nonzero_Importance",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    consensus["Consensus_Rank"] = np.arange(
        1,
        len(consensus) + 1,
    )

    consensus = consensus[
        [
            "Consensus_Rank",
            "Feature",
            "RF",
            "XGB",
            "LR",
            "SVM",
            "Consensus",
            "Models_With_Nonzero_Importance",
        ]
    ]

    consensus_dir = output_dir / "consensus"

    consensus.to_csv(
        consensus_dir / "Consensus_Biomarkers.tsv",
        sep="\t",
        index=False,
    )

    top20 = consensus.head(top_n).copy()

    top10 = consensus.head(top_n_comparison).copy()

    top20.to_csv(
        consensus_dir / f"Top{top_n}_Consensus_Biomarkers.tsv",
        sep="\t",
        index=False,
    )

    top10.to_csv(
        consensus_dir
        / f"Top{top_n_comparison}_Consensus_Biomarkers.tsv",
        sep="\t",
        index=False,
    )

    correlation = consensus[
        importance_columns
    ].corr(method="spearman")

    correlation.to_csv(
        consensus_dir / "Model_Importance_Spearman_Correlation.tsv",
        sep="\t",
    )

    contribution = pd.DataFrame({
        "Model": importance_columns,
        "Total_Importance": [
            consensus[name].sum()
            for name in importance_columns
        ],
    })

    contribution = contribution.sort_values(
        "Total_Importance",
        ascending=False,
    ).reset_index(drop=True)

    contribution.to_csv(
        consensus_dir / "Model_Contribution.tsv",
        sep="\t",
        index=False,
    )

    return consensus, top20, top10, correlation, contribution


#Create interpretability figure

def create_interpretability_figure(
    top20: pd.DataFrame,
    top10: pd.DataFrame,
    correlation: pd.DataFrame,
    contribution: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:

    logger.info("Creating interpretability figure.")

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(26, 24),
    )

    #A.Consensus biomarker ranking

    plot_top20 = top20.sort_values(
        "Consensus",
        ascending=True,
    )

    axs[0, 0].barh(
        plot_top20["Feature"],
        plot_top20["Consensus"],
        height=0.72,
    )

    axs[0, 0].set_title(
        "(a) Consensus biomarker ranking",
        pad=15,
    )

    axs[0, 0].set_xlabel(
        "Combined normalized importance"
    )

    axs[0, 0].set_ylabel(
        "Gene-family feature"
    )

    #B.Model agreement

    image = axs[0, 1].imshow(
        correlation.values,
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axs[0, 1].set_title(
        "(b) Model agreement",
        pad=15,
    )

    axs[0, 1].set_xticks(
        np.arange(len(correlation.columns))
    )

    axs[0, 1].set_yticks(
        np.arange(len(correlation.index))
    )

    axs[0, 1].set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right",
    )

    axs[0, 1].set_yticklabels(
        correlation.index
    )

    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):

            value = correlation.iloc[i, j]

            annotation = (
                f"{value:.2f}"
                if np.isfinite(value)
                else "NA"
            )

            axs[0, 1].text(
                j,
                i,
                annotation,
                ha="center",
                va="center",
                fontsize=28,
            )

    axs[0, 1].set_xlabel("Model")

    axs[0, 1].set_ylabel("Model")

    colorbar = fig.colorbar(
        image,
        ax=axs[0, 1],
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Spearman correlation",
        fontsize=26,
        fontweight="bold",
    )

    colorbar.ax.tick_params(labelsize=24)

    #C.Model contribution

    axs[1, 0].bar(
        contribution["Model"],
        contribution["Total_Importance"],
        width=0.65,
    )

    axs[1, 0].set_title(
        "(c) Total model contribution",
        pad=15,
    )

    axs[1, 0].set_ylabel(
        "Sum of normalized importance"
    )

    axs[1, 0].set_xlabel("Model")

    maximum = contribution["Total_Importance"].max()

    for i, value in enumerate(
        contribution["Total_Importance"]
    ):

        axs[1, 0].text(
            i,
            value + (maximum * 0.02 if maximum > 0 else 0.02),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=28,
        )

    if maximum > 0:
        axs[1, 0].set_ylim(0, maximum * 1.15)

    #D.Top biomarker comparison

    x = np.arange(len(top10))

    width = 0.20

    for index, model_name in enumerate(MODEL_NAMES):

        offset = (
            index - (len(MODEL_NAMES) - 1) / 2
        ) * width

        axs[1, 1].bar(
            x + offset,
            top10[model_name],
            width=width,
            label=model_name,
        )

    axs[1, 1].set_xticks(x)

    axs[1, 1].set_xticklabels(
        top10["Feature"],
        rotation=75,
        ha="right",
    )

    axs[1, 1].set_title(
        "(d) Importance of top consensus biomarkers",
        pad=15,
    )

    axs[1, 1].set_ylabel(
        "Normalized importance"
    )

    axs[1, 1].set_xlabel(
        "Gene-family feature"
    )

    axs[1, 1].legend(
        loc="upper right",
        frameon=True,
    )

    ##Final layout

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.12,
        wspace=0.28,
        hspace=0.35,
    )

    for ax in axs.flat:

        ax.title.set_fontname("Times New Roman")
        ax.title.set_fontsize(28)
        ax.title.set_fontweight("bold")

        ax.xaxis.label.set_fontname("Times New Roman")
        ax.xaxis.label.set_fontsize(26)
        ax.xaxis.label.set_fontweight("bold")

        ax.yaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontsize(26)
        ax.yaxis.label.set_fontweight("bold")

        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(22)

    figure_dir = output_dir / "figures"

    png_path = figure_dir / "SHAP_interpretability.png"

    pdf_path = figure_dir / "SHAP_interpretability.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info("Saved figure: %s", png_path)
    logger.info("Saved figure: %s", pdf_path)

    return png_path, pdf_path


#Save analysis metadata

def save_analysis_metadata(
    args: argparse.Namespace,
    model_files: dict[str, Path],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    consensus: pd.DataFrame,
    top20: pd.DataFrame,
    top10: pd.DataFrame,
    output_dir: Path,
) -> None:

    metadata = {
        "dataset": "NCBI",
        "analysis": "SHAP and biomarker interpretability",
        "split_method": "Recreated stratified train/test split",
        "test_size": args.test_size,
        "random_state": args.random_state,
        "split_stratification": "Classification label",
        "original_dataset": str(args.input.resolve()),
        "training_samples": int(len(train_df)),
        "internal_test_samples": int(len(test_df)),
        "expected_gene_features": args.expected_features,
        "classification_threshold": args.threshold,
        "positive_class": 1,
        "positive_class_name": "Resistant",
        "negative_class": 0,
        "negative_class_name": "Susceptible",
        "models_used": list(model_files.keys()),
        "model_files": {
            name: str(path.resolve())
            for name, path in model_files.items()
        },
        "model_refitting_performed": False,
        "interpretation_dataset": "Internal held-out test set",
        "model_interpretation_methods": {
            "RF": "TreeSHAP",
            "XGB": "TreeSHAP",
            "LR": "Absolute logistic regression coefficients",
            "SVM": "Permutation importance using ROC-AUC",
        },
        "permutation_repeats": args.permutations,
        "importance_normalization": (
            "Maximum positive importance scaling per model"
        ),
        "consensus_method": (
            "Sum of four model-specific normalized importance scores"
        ),
        "consensus_feature_count": int(len(consensus)),
        "top_consensus_biomarkers": int(len(top20)),
        "top_comparison_biomarkers": int(len(top10)),
        "output_directory": str(output_dir.resolve()),
    }

    metadata_path = (
        output_dir / "metadata" / "SHAP_analysis_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Analysis metadata saved: %s", metadata_path)


#Main workflow

def main() -> None:

    args = parse_arguments()

    configure_plotting()

    create_output_directories(args.output)

    logger.info("=" * 75)
    logger.info("NCBI SHAP AND BIOMARKER INTERPRETABILITY ANALYSIS")
    logger.info("=" * 75)

    model_files = validate_input_files(
        args.input,
        args.model_dir,
    )

    ##Load and validate original dataset

    df, X, y = load_dataset(
        args.input,
        args.expected_features,
    )

    feature_names = list(X.columns)

    ##Recreate split

    train_df, test_df, y_train, y_test = recreate_split(
        df,
        X,
        y,
        args.test_size,
        args.random_state,
    )

    X_train = X.loc[train_df.index].reset_index(drop=True)

    X_test = X.loc[test_df.index].reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)

    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Training/test feature ordering differs.")

    if len(X_train.columns) != args.expected_features:
        raise ValueError("Unexpected feature count after splitting.")

    ##Save recreated split Genome IDs for reproducibility

    split_dir = args.output / "metadata"

    pd.DataFrame({
        ID_COL: train_df[ID_COL].astype(str),
    }).to_csv(
        split_dir / "Recreated_Training_Genome_IDs.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame({
        ID_COL: test_df[ID_COL].astype(str),
    }).to_csv(
        split_dir / "Recreated_Internal_Test_Genome_IDs.tsv",
        sep="\t",
        index=False,
    )

    ##Load saved models

    models = load_models(
        model_files,
        args.expected_features,
    )

    ###Verify training sample count when available.
    for model_name, model in models.items():

        classifier = model.named_steps["clf"]

        fitted_count = getattr(
            classifier,
            "n_samples_fit_",
            None,
        )

        if fitted_count is not None and fitted_count != len(X_train):
            raise ValueError(
                f"{model_name} training sample count mismatch. "
                f"Saved model: {fitted_count}; "
                f"recreated training set: {len(X_train)}. "
                "Check row order, split settings, and input dataset."
            )

    ##Feature selection

    selected_features = save_feature_selection(
        models,
        feature_names,
        args.output,
    )

    ##Model-specific importance

    importance_tables = {}

    importance_tables["RF"] = analyze_random_forest(
        models["RF"],
        X_test,
        selected_features["RF"],
        args.output,
    )

    importance_tables["XGB"] = analyze_xgboost(
        models["XGB"],
        X_test,
        selected_features["XGB"],
        args.output,
    )

    importance_tables["LR"] = analyze_logistic_regression(
        models["LR"],
        selected_features["LR"],
        args.output,
    )

    importance_tables["SVM"] = analyze_svm(
        models["SVM"],
        X_test,
        y_test,
        selected_features["SVM"],
        args.permutations,
        args.random_state,
        args.output,
    )

    ##Consensus analysis

    (
        consensus,
        top20,
        top10,
        correlation,
        contribution,
    ) = calculate_consensus(
        importance_tables,
        args.output,
        args.top_n,
        args.top_n_comparison,
    )

    ##Figure

    png_path, pdf_path = create_interpretability_figure(
        top20,
        top10,
        correlation,
        contribution,
        args.output,
    )

    ##Metadata

    save_analysis_metadata(
        args,
        model_files,
        train_df,
        test_df,
        consensus,
        top20,
        top10,
        args.output,
    )

    ##Final report

    logger.info("=" * 75)
    logger.info("INTERPRETABILITY ANALYSIS COMPLETED")
    logger.info("=" * 75)

    logger.info("Complete NCBI genomes: %d", len(X))
    logger.info("Training genomes: %d", len(X_train))
    logger.info("Internal test genomes: %d", len(X_test))
    logger.info("Gene-family features: %d", X.shape[1])

    logger.info("Consensus biomarkers: %d", len(consensus))

    logger.info("Top biomarkers:")
    logger.info("\n%s", top20.to_string(index=False))

    logger.info("Figures:")
    logger.info("%s", png_path)
    logger.info("%s", pdf_path)

    logger.info("Output directory: %s", args.output.resolve())

    logger.info("=" * 75)
    logger.info("DONE")


if __name__ == "__main__":
    main()