#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
NCBI ML model development and internal validation

Project: Pan-genomic Analysis and Explainable Machine Learning Reveal Genomic 
Determinants of Methicillin Resistance in Staphylococcus aureus

Description:
This script develops, optimizes, and evaluates ML classification models using genomic 
features derived from the NCBI dataset to predict AMR phenotypes.

Four supervised ML algorithms are implemented:
Random Forest (RF), Extreme Gradient Boosting (XGBoost),
Logistic Regression (LR), and Support Vector Machine (SVM).

The workflow incorporates stratified dataset partitioning,
mutual information-based feature selection, hyperparameter
optimization, nested cross-validation (CV), independent internal
test-set evaluation, bootstrap confidence intervals (CIs), and
model interpretability through selected feature identification.

Optimized models are evaluated using multiple classification
performance metrics and graphical assessments to characterize
predictive performance, discrimination, and probability
calibration.

Objectives:
a. Develop supervised ML classifiers using NCBI-derived genomic features.
b. Validate genomic feature matrices & AMR classification labels.
c. Partition the NCBI dataset into stratified training (80%) and internal testing (20%) subsets.
d. Perform mutual information-based feature selection within ML pipelines to minimize information leakage.
e. Optimize model hyperparameters using randomized search.
f. Estimate model performance through nested stratified CV.
g. Evaluate optimized models using the untouched internal test dataset.
h. Calculate bootstrap-based 95% CIs for model performance estimates.
i. Generate genome-level AMR predictions.
j. Identify model-specific selected genomic features.
k. Save optimized models evaluated during internal validation.
l. Train final development models using the complete NCBI dataset.
m. Generate receiver operating characteristic (ROC), precision-recall (PR), and calibration curves.
n. Export model performance metrics, prediction results, selected features, model artifacts, and 
reproducibility records.

ML algorithms:

a. RF: An ensemble learning algorithm based on multiple decision trees.

b. XGB: A gradient-boosted decision tree algorithm.

c. LR: A linear classification algorithm for binary prediction.

d. SVM: A supervised learning algorithm for classification using an optimized decision boundary.

Analytical workflow:

a. Load the NCBI genomic feature dataset.
b. Validate the dataset structure, genome identifiers, classification labels, and binary genomic features.
c. Separate genome identifiers, target labels, and predictor variables.
d. Perform stratified 80:20 training/internal-test partitioning.
e. Save the training and internal-test datasets.
f. Construct ML pipelines incorporating mutual information-based feature selection and classification algorithms.
g. Perform randomized hyperparameter optimization using the specified CV strategy.
h. Estimate model performance using nested stratified CV within the training dataset.
i. Select optimized model configurations based on the predefined optimization metric.
j. Evaluate optimized models on the untouched internal test dataset.
k. Calculate bootstrap-based 95% CIs for the selected performance metrics.
l. Generate genome-level predictions and associated prediction probabilities, where supported.
m. Extract and export model-specific selected genomic features.
n. Save optimized internal-validation model artifacts.
o. Retrain final development models using the complete NCBI dataset.
p. Generate ROC, precision-recall, and calibration visualizations.
q. Export performance reports, prediction tables, model artifacts, and reproducibility metadata.

Input file:

NCBI_Final_ML_Dataset.tsv

Description: Tab-separated genomic feature matrix containing genome identifiers, binary genomic predictors, and AMR classification labels.

Input data requirements:

Required metadata columns:
- Genome ID: Unique identifier for each bacterial genome.
- label: Binary AMR classification label.

Genomic feature columns:
- Individual genomic features represented as binary variables.
- 0: Feature absent.
- 1: Feature present.

Classification encoding:
- 0: Susceptible.
- 1: Resistant.

Dataset partitioning:
- Training dataset: 80%.
- Internal test dataset: 20%.
- Stratification: Based on the classification label.

Feature selection: Mutual information-based feature selection is performed within the ML pipelines to reduce the risk
of data leakage during model training and evaluation.

Hyperparameter optimization: Randomized search is used to identify optimized hyperparameter configurations for each classifier.

Model evaluation:
- Nested stratified CV.
- Internal test-set evaluation.
- Bootstrap-based 95% CIs.
- Receiver operating characteristic (ROC) analysis.
- PR analysis.
- Probability calibration assessment.

Expected outputs:
- Stratified training dataset.
- Untouched internal test dataset.
- Optimized RF model.
- Optimized XGB model.
- Optimized LR model.
- Optimized SVM model.
- Nested CV performance summaries.
- Internal test-set performance reports.
- Bootstrap 95% CIs.
- Genome-level prediction results.
- Model-specific selected feature lists.
- Final development model artifacts.
- ROC curves.
- PR curves.
- Calibration curves.
- Model comparison summaries.
- Reproducibility and execution metadata.

Reproducibility: The random seed, data partitioning strategy, feature selection procedures, CV configuration,
hyperparameter search settings, and model evaluation parameters should be documented and retained.

The internal test dataset is kept separate from model training and hyperparameter optimization to provide
an independent assessment of predictive performance within the NCBI dataset.

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

Example execution:
python scripts/ncbi_model_development.py \
    --input data/NCBI_Final_ML_Dataset.tsv \
    --output results/

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
import warnings

from pathlib import Path
from functools import partial
from importlib.metadata import version, PackageNotFoundError

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.calibration import calibration_curve


#Default settings

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5

N_ITER_SEARCH = 20

EXPECTED_FEATURES = 96

CLASSIFICATION_THRESHOLD = 0.50

N_BOOTSTRAP = 2000

CONFIDENCE_LEVEL = 0.95

METADATA_COLUMNS = [
    "Genome ID",
    "label"
]

CLASS_NAMES = {
    0: "Susceptible",
    1: "Resistant"
}


#Command-line arguments

def parse_arguments():

    parser = argparse.ArgumentParser(

        description=(
            "NCBI genomic machine learning model development "
            "and internal validation pipeline."
        )

    )

    parser.add_argument(

        "--input",
        type=Path,
        required=True,
        help="Path to the NCBI filtered feature matrix."

    )

    parser.add_argument(

        "--output",
        type=Path,
        required=True,
        help="Directory for model development outputs."

    )

    parser.add_argument(

        "--expected-features",
        type=int,
        default=EXPECTED_FEATURES,
        help=(
            "Expected number of gene features. "
            "Use 0 to disable this check."
        )

    )

    parser.add_argument(

        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Internal test proportion. Default: 0.20."

    )

    parser.add_argument(

        "--cv-folds",
        type=int,
        default=N_SPLITS,
        help="Number of stratified CV folds. Default: 5."

    )

    parser.add_argument(

        "--search-iterations",
        type=int,
        default=N_ITER_SEARCH,
        help="Randomized search iterations. Default: 20."

    )

    parser.add_argument(

        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap replicates. Default: 2000."

    )

    parser.add_argument(

        "--threshold",
        type=float,
        default=CLASSIFICATION_THRESHOLD,
        help="Classification probability threshold. Default: 0.50."

    )

    return parser.parse_args()


#Logging configuration

def configure_logging(output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logging.basicConfig(

        level=logging.INFO,

        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),

        handlers=[

            logging.StreamHandler(sys.stdout),

            logging.FileHandler(

                output_dir / "model_development.log",

                mode="w",

                encoding="utf-8"

            )

        ],

        force=True

    )


#Validate settings and create directories

def validate_settings(args):

    if not args.input.is_file():

        raise FileNotFoundError(
            f"Input dataset not found: {args.input}"
        )

    if not 0 < args.test_size < 1:

        raise ValueError(
            "Test size must be between 0 and 1."
        )

    if args.cv_folds < 2:

        raise ValueError(
            "At least two CV folds are required."
        )

    if args.search_iterations < 1:

        raise ValueError(
            "Search iterations must be at least 1."
        )

    if args.bootstrap < 100:

        raise ValueError(
            "Use at least 100 bootstrap replicates."
        )

    if not 0 <= args.threshold <= 1:

        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    if args.expected_features < 0:

        raise ValueError(
            "Expected feature count cannot be negative."
        )

    directories = {

        "output": args.output,

        "data": args.output / "data_splits",

        "models": args.output / "saved_models",

        "results": args.output / "results",

        "search": args.output / "hyperparameter_search",

        "figures": args.output / "figures",

        "predictions": args.output / "predictions"

    }

    for directory in directories.values():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    return directories


#Load & validate dataset

def load_dataset(input_file):

    logging.info("Loading NCBI dataset.")

    df = pd.read_csv(

        input_file,

        sep="\t",

        low_memory=False

    )

    df.columns = df.columns.str.strip()

    if df.empty:

        raise ValueError(
            "Input dataset is empty."
        )

    if df.columns.duplicated().any():

        raise ValueError(
            "Duplicate column names detected."
        )

    missing_columns = [

        col for col in METADATA_COLUMNS

        if col not in df.columns

    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["Genome ID"].isna().any():

        raise ValueError(
            "Missing Genome IDs detected."
        )

    if df["Genome ID"].astype(str).str.strip().eq("").any():

        raise ValueError(
            "Empty Genome IDs detected."
        )

    if df["Genome ID"].duplicated().any():

        raise ValueError(
            "Duplicate Genome IDs detected."
        )

    if df["label"].isna().any():

        raise ValueError(
            "Missing classification labels detected."
        )

    logging.info(
        "Dataset loaded successfully: %s",
        df.shape
    )

    return df


#Identify & validate gene features

def prepare_features(df, expected_features):

    feature_columns = [

        col for col in df.columns

        if col not in METADATA_COLUMNS

        and not col.lower().startswith("group_")

    ]

    if not feature_columns:

        raise ValueError(
            "No gene features identified."
        )

    if expected_features > 0:

        if len(feature_columns) != expected_features:

            raise ValueError(

                f"Expected {expected_features} gene features, "

                f"but found {len(feature_columns)}."

            )

    X = df[feature_columns].apply(

        pd.to_numeric,

        errors="coerce"

    )

    if X.isna().any().any():

        raise ValueError(
            "Missing or non-numeric gene feature values detected."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():

        raise ValueError(
            "Infinite values detected in gene features."
        )

    if not X.isin([0, 1]).all().all():

        raise ValueError(
            "Gene features must contain only binary values (0/1)."
        )

    X = X.astype(np.uint8)

    y = pd.to_numeric(

        df["label"],

        errors="coerce"

    )

    if y.isna().any():

        raise ValueError(
            "Invalid or missing classification labels detected."
        )

    if not np.isfinite(y.to_numpy(dtype=float)).all():

        raise ValueError(
            "Non-finite classification labels detected."
        )

    if not set(y.unique()).issubset({0, 1}):

        raise ValueError(
            "Classification labels must be 0 or 1."
        )

    y = y.astype(int)

    if y.nunique() != 2:

        raise ValueError(
            "Both classification classes must be present."
        )

    ##Check that each class has sufficient observations for the requested outer & inner CV

    minority_count = int(y.value_counts().min())

    minimum_required = (
        1 + (1 + args_cv_folds_placeholder())
    )

    if minority_count < minimum_required:

        raise ValueError(
            "Insufficient minority-class observations for "
            "the requested nested cross-validation."
        )

    logging.info(
        "Gene features identified: %d",
        len(feature_columns)
    )

    logging.info(
        "Classification labels validated."
    )

    return X, y, feature_columns


def args_cv_folds_placeholder():

    ##This function is replaced by the configured CV value when the pipeline calls the validation routine

    return N_SPLITS


#Class distribution

def summarize_class_distribution(y):

    counts = y.value_counts().sort_index()

    proportions = y.value_counts(
        normalize=True
    ).sort_index()

    summary = pd.DataFrame({

        "Class": [
            CLASS_NAMES[int(i)]
            for i in counts.index
        ],

        "Label": counts.index,

        "Genomes": counts.values,

        "Proportion": proportions.values

    })

    logging.info(
        "Class distribution:\n%s",
        summary.to_string(index=False)
    )

    return summary


#Stratified training and internal test split

def create_data_split(

    df,
    X,
    y,
    directories,
    test_size

):

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=RANDOM_STATE,

        stratify=y

    )

    if y_train.value_counts().min() < N_SPLITS:

        raise ValueError(
            "Insufficient training observations for nested CV."
        )

    train_df = df.loc[X_train.index].copy()

    test_df = df.loc[X_test.index].copy()

    train_df.to_csv(

        directories["data"] /
        "NCBI_training_dataset.tsv",

        sep="\t",

        index=False

    )

    test_df.to_csv(

        directories["data"] /
        "NCBI_internal_test_dataset.tsv",

        sep="\t",

        index=False

    )

    split_summary = pd.DataFrame({

        "Dataset": [
            "Complete_NCBI",
            "Training",
            "Internal_Test"
        ],

        "Genomes": [
            len(X),
            len(X_train),
            len(X_test)
        ],

        "Susceptible_0": [
            int((y == 0).sum()),
            int((y_train == 0).sum()),
            int((y_test == 0).sum())
        ],

        "Resistant_1": [
            int((y == 1).sum()),
            int((y_train == 1).sum()),
            int((y_test == 1).sum())
        ]

    })

    split_summary.to_csv(

        directories["results"] /
        "NCBI_80_20_split_summary.tsv",

        sep="\t",

        index=False

    )

    logging.info(
        "Training genomes: %d",
        len(X_train)
    )

    logging.info(
        "Internal test genomes: %d",
        len(X_test)
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df
    )


#CV

def create_cv():

    return StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )


#Mutual information feature selection

mi_score = partial(

    mutual_info_classif,

    random_state=RANDOM_STATE

)

candidate_k = [
    25,
    50,
    75,
    96
]


#Define ML pipelines

def create_pipelines():

    pipelines = {

        "RF": Pipeline([

            (
                "variance",
                VarianceThreshold(threshold=0.0)
            ),

            (
                "selector",
                SelectKBest(score_func=mi_score)
            ),

            (
                "clf",
                RandomForestClassifier(

                    random_state=RANDOM_STATE,

                    class_weight="balanced",

                    n_jobs=1

                )
            )

        ]),

        "XGB": Pipeline([

            (
                "variance",
                VarianceThreshold(threshold=0.0)
            ),

            (
                "selector",
                SelectKBest(score_func=mi_score)
            ),

            (
                "clf",
                XGBClassifier(

                    eval_metric="logloss",

                    random_state=RANDOM_STATE,

                    tree_method="hist",

                    n_jobs=1

                )
            )

        ]),

        "LR": Pipeline([

            (
                "variance",
                VarianceThreshold(threshold=0.0)
            ),

            (
                "selector",
                SelectKBest(score_func=mi_score)
            ),

            (
                "clf",
                LogisticRegression(

                    max_iter=5000,

                    solver="liblinear",

                    class_weight="balanced",

                    random_state=RANDOM_STATE

                )
            )

        ]),

        "SVM": Pipeline([

            (
                "variance",
                VarianceThreshold(threshold=0.0)
            ),

            (
                "selector",
                SelectKBest(score_func=mi_score)
            ),

            (
                "clf",
                SVC(

                    probability=True,

                    class_weight="balanced",

                    random_state=RANDOM_STATE

                )
            )

        ])

    }

    return pipelines


#Hyperparameter search spaces

def create_parameter_distributions():

    return {

        "RF": {

            "selector__k": candidate_k,

            "clf__n_estimators": randint(200, 800),

            "clf__max_depth": [
                None,
                10,
                20,
                30
            ]

        },

        "XGB": {

            "selector__k": candidate_k,

            "clf__n_estimators": randint(200, 800),

            "clf__max_depth": randint(3, 10),

            "clf__learning_rate": uniform(0.01, 0.20),

            "clf__subsample": uniform(0.60, 0.40),

            "clf__colsample_bytree": uniform(0.60, 0.40)

        },

        "LR": {

            "selector__k": candidate_k,

            "clf__C": uniform(0.01, 10.0),

            "clf__penalty": [
                "l1",
                "l2"
            ]

        },

        "SVM": {

            "selector__k": candidate_k,

            "clf__C": uniform(0.10, 10.0),

            "clf__kernel": [
                "linear",
                "rbf"
            ]

        }

    }


#Randomized hyperparameter search

def run_randomized_search(

    pipeline,
    parameter_space,
    X,
    y,
    cv,
    n_iter

):

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=parameter_space,

        n_iter=n_iter,

        scoring="matthews_corrcoef",

        cv=cv,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        refit=True,

        return_train_score=False,

        error_score="raise"

    )

    search.fit(
        X,
        y
    )

    return search


#Nested CV

def nested_cross_validation(

    model_name,
    pipeline,
    parameter_space,
    X_train,
    y_train,
    directories,
    n_iter

):

    logging.info(
        "Starting nested CV for %s.",
        model_name
    )

    outer_cv = create_cv()

    inner_cv = create_cv()

    oof_probabilities = np.full(

        len(X_train),

        np.nan,

        dtype=float

    )

    fold_records = []

    for fold, (train_idx, validation_idx) in enumerate(

        outer_cv.split(X_train, y_train),

        start=1

    ):

        logging.info(
            "%s: outer fold %d/%d",
            model_name,
            fold,
            N_SPLITS
        )

        X_outer_train = X_train.iloc[train_idx]

        X_outer_validation = X_train.iloc[validation_idx]

        y_outer_train = y_train.iloc[train_idx]

        y_outer_validation = y_train.iloc[validation_idx]

        if y_outer_train.value_counts().min() < N_SPLITS:

            raise ValueError(
                "Insufficient class observations in nested "
                f"CV training fold {fold}."
            )

        search = run_randomized_search(

            pipeline,

            parameter_space,

            X_outer_train,

            y_outer_train,

            inner_cv,

            n_iter

        )

        fold_model = search.best_estimator_

        probabilities = fold_model.predict_proba(

            X_outer_validation

        )[:, list(fold_model.classes_).index(1)]

        oof_probabilities[validation_idx] = probabilities

        fold_records.append({

            "Model": model_name,

            "Outer_Fold": fold,

            "Best_Inner_CV_MCC": search.best_score_,

            "Best_Parameters": json.dumps(

                make_json_serializable(
                    search.best_params_
                )

            )

        })

    if np.isnan(oof_probabilities).any():

        raise RuntimeError(
            "Some training samples did not receive OOF predictions."
        )

    ##Calculate pooled nested OOF performance

    metrics, cm, predictions = calculate_metrics(

        y_train,

        oof_probabilities

    )

    metrics["Model"] = model_name

    metrics["Dataset"] = "Nested_Training_OOF_CV"

    prediction_df = pd.DataFrame({

        "Genome_ID": X_train.index,

        "True_Label": y_train.values,

        "Predicted_Probability": oof_probabilities,

        "Predicted_Label": predictions

    })

    prediction_df.to_csv(

        directories["predictions"] /
        f"{model_name}_Nested_Training_OOF_Predictions.tsv",

        sep="\t",

        index=False

    )

    pd.DataFrame(
        fold_records
    ).to_csv(

        directories["search"] /
        f"{model_name}_Nested_CV_Fold_Results.tsv",

        sep="\t",

        index=False

    )

    pd.DataFrame(
        [metrics]
    ).to_csv(

        directories["results"] /
        f"{model_name}_Nested_Training_CV_Metrics.tsv",

        sep="\t",

        index=False

    )

    save_confusion_matrix(

        cm,

        model_name,

        "Nested_Training_OOF_CV",

        directories

    )

    return (
        metrics,
        oof_probabilities,
        predictions
    )


#JSON serialization

def make_json_serializable(value):

    if isinstance(value, dict):

        return {

            str(key): make_json_serializable(item)

            for key, item in value.items()

        }

    if isinstance(value, (list, tuple)):

        return [

            make_json_serializable(item)

            for item in value

        ]

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, np.generic):

        return value.item()

    if isinstance(value, Path):

        return str(value)

    return value


#Classification metrics

def calculate_metrics(

    y_true,

    y_prob,

    threshold=None

):

    if threshold is None:

        threshold = CLASSIFICATION_THRESHOLD

    y_true = np.asarray(y_true).astype(int)

    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) != len(y_prob):

        raise ValueError(
            "True labels and predicted probabilities differ in length."
        )

    if not np.isfinite(y_prob).all():

        raise ValueError(
            "Non-finite predicted probabilities detected."
        )

    if np.any((y_prob < 0) | (y_prob > 1)):

        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]

    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else np.nan

    )

    metrics = {

        "Threshold": threshold,

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

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

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp)

    }

    return (
        metrics,
        cm,
        y_pred
    )


#Save confusion matrices

def save_confusion_matrix(

    cm,

    model_name,

    dataset_name,

    directories

):

    confusion_df = pd.DataFrame(

        cm,

        index=[
            "Actual_Susceptible",
            "Actual_Resistant"
        ],

        columns=[
            "Predicted_Susceptible",
            "Predicted_Resistant"
        ]

    )

    confusion_df.to_csv(

        directories["results"] /
        f"{model_name}_{dataset_name}_Confusion_Matrix.tsv",

        sep="\t"

    )


#Stratified bootstrap confidence intervals

def bootstrap_confidence_intervals(

    y_true,

    y_prob,

    n_bootstrap,

    random_state=RANDOM_STATE

):

    y_true = np.asarray(y_true).astype(int)

    y_prob = np.asarray(y_prob, dtype=float)

    class0 = np.flatnonzero(y_true == 0)

    class1 = np.flatnonzero(y_true == 1)

    if len(class0) == 0 or len(class1) == 0:

        raise ValueError(
            "Both classes are required for bootstrap."
        )

    rng = np.random.default_rng(
        random_state
    )

    metric_names = [

        "Accuracy",

        "Precision",

        "Sensitivity",

        "Specificity",

        "F1",

        "MCC",

        "ROC_AUC",

        "Average_Precision"

    ]

    samples = {

        metric: []

        for metric in metric_names

    }

    for _ in range(n_bootstrap):

        sampled0 = rng.choice(

            class0,

            size=len(class0),

            replace=True

        )

        sampled1 = rng.choice(

            class1,

            size=len(class1),

            replace=True

        )

        indices = np.concatenate([

            sampled0,

            sampled1

        ])

        rng.shuffle(indices)

        boot_metrics, _, _ = calculate_metrics(

            y_true[indices],

            y_prob[indices]

        )

        for metric in metric_names:

            value = boot_metrics[metric]

            if np.isfinite(value):

                samples[metric].append(value)

    original_metrics, _, _ = calculate_metrics(

        y_true,

        y_prob

    )

    rows = []

    for metric in metric_names:

        values = np.asarray(

            samples[metric],

            dtype=float

        )

        if len(values) == 0:

            lower = np.nan

            upper = np.nan

        else:

            lower, upper = np.percentile(

                values,

                [

                    (1 - CONFIDENCE_LEVEL) * 50,

                    100 - (1 - CONFIDENCE_LEVEL) * 50

                ]

            )

        rows.append({

            "Metric": metric,

            "Estimate": original_metrics[metric],

            "CI_95_Lower": lower,

            "CI_95_Upper": upper,

            "Bootstrap_Replicates_Used": len(values)

        })

    return pd.DataFrame(rows)


#Save evaluation results

def save_evaluation(

    model_name,

    dataset_name,

    y_true,

    y_prob,

    directories,

    n_bootstrap

):

    metrics, cm, predictions = calculate_metrics(

        y_true,

        y_prob

    )

    metrics["Model"] = model_name

    metrics["Dataset"] = dataset_name

    ci_df = bootstrap_confidence_intervals(

        y_true,

        y_prob,

        n_bootstrap

    )

    ci_df.insert(
        0,
        "Model",
        model_name
    )

    ci_df.insert(
        1,
        "Dataset",
        dataset_name
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(

        directories["results"] /
        f"{model_name}_{dataset_name}_Metrics.tsv",

        sep="\t",

        index=False

    )

    ci_df.to_csv(

        directories["results"] /
        f"{model_name}_{dataset_name}_95CI.tsv",

        sep="\t",

        index=False

    )

    save_confusion_matrix(

        cm,

        model_name,

        dataset_name,

        directories

    )

    return (
        metrics,
        ci_df,
        cm,
        predictions
    )


#Print performance metrics

def print_metrics(metrics, ci_df):

    ci_lookup = ci_df.set_index("Metric")

    metric_names = [

        "Accuracy",

        "Precision",

        "Sensitivity",

        "Specificity",

        "F1",

        "MCC",

        "ROC_AUC",

        "Average_Precision"

    ]

    for metric in metric_names:

        row = ci_lookup.loc[metric]

        logging.info(

            "%-20s: %.4f (95%% CI %.4f-%.4f)",

            metric,

            metrics[metric],

            row["CI_95_Lower"],

            row["CI_95_Upper"]

        )


#Extract selected features

def extract_selected_features(model, feature_names):

    variance_filter = model.named_steps[
        "variance"
    ]

    selector = model.named_steps[
        "selector"
    ]

    after_variance = np.asarray(feature_names)[

        variance_filter.get_support()

    ]

    selected = after_variance[

        selector.get_support()

    ]

    return selected.tolist()


#Save model & feature information

def save_model_information(

    model_name,

    model,

    feature_names,

    directories,

    model_type

):

    selected = extract_selected_features(

        model,

        feature_names

    )

    pd.DataFrame({

        "Feature": selected

    }).to_csv(

        directories["results"] /
        f"{model_name}_{model_type}_Selected_Features.tsv",

        sep="\t",

        index=False

    )

    joblib.dump(

        model,

        directories["models"] /
        f"{model_name}_{model_type}.pkl"

    )

    return selected


#Model development

def develop_models(

    X,

    y,

    X_train,

    X_test,

    y_train,

    y_test,

    train_df,

    test_df,

    feature_names,

    directories,

    args

):

    pipelines = create_pipelines()

    parameter_spaces = create_parameter_distributions()

    internal_metrics_all = []

    nested_metrics_all = []

    best_parameters = {}

    selected_features = {}

    roc_data = {}

    pr_data = {}

    calibration_data = {}

    model_paths = {}

    inner_cv = create_cv()

    for name, pipeline in pipelines.items():

        logging.info("=" * 75)

        logging.info(
            "MODEL DEVELOPMENT: %s",
            name
        )

        logging.info("=" * 75)

        #A.Nested CV on training data

        nested_metrics, oof_prob, oof_pred = nested_cross_validation(

            name,

            pipeline,

            parameter_spaces[name],

            X_train,

            y_train,

            directories,

            args.search_iterations

        )

        nested_metrics_all.append(
            nested_metrics
        )

        #B.Final hyperparameter search on complete training data

        logging.info(
            "%s: optimizing final training hyperparameters.",
            name
        )

        search = run_randomized_search(

            pipeline,

            parameter_spaces[name],

            X_train,

            y_train,

            inner_cv,

            args.search_iterations

        )

        best_model = search.best_estimator_

        logging.info(
            "%s best inner CV MCC: %.4f",
            name,
            search.best_score_
        )

        logging.info(
            "%s best parameters: %s",
            name,
            search.best_params_
        )

        pd.DataFrame(
            search.cv_results_
        ).to_csv(

            directories["search"] /
            f"{name}_RandomizedSearchCV_Results.tsv",

            sep="\t",

            index=False

        )

        best_parameters[name] = make_json_serializable(

            search.best_params_

        )

        #C.Save final search results

        with open(

            directories["search"] /
            f"{name}_Best_Parameters.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                best_parameters[name],

                file,

                indent=4

            )

        #D.Evaluate on untouched internal test data

        best_model.fit(

            X_train,

            y_train

        )

        test_prob = best_model.predict_proba(

            X_test

        )[:, list(best_model.classes_).index(1)]

        (
            test_metrics,
            test_ci,
            test_cm,
            test_predictions

        ) = save_evaluation(

            name,

            "NCBI_Internal_Test",

            y_test,

            test_prob,

            directories,

            args.bootstrap

        )

        internal_metrics_all.append(
            test_metrics
        )

        logging.info(
            "%s internal test performance:",
            name
        )

        print_metrics(
            test_metrics,
            test_ci
        )

        #E.Save genome-level internal test predictions

        prediction_df = pd.DataFrame({

            "Genome_ID": test_df["Genome ID"].values,

            "True_Label": y_test.values,

            "True_Class": [

                CLASS_NAMES[int(value)]

                for value in y_test.values

            ],

            "Predicted_Probability_Resistant": test_prob,

            "Predicted_Label": test_predictions,

            "Predicted_Class": [

                CLASS_NAMES[int(value)]

                for value in test_predictions

            ],

            "Correct_Classification": (

                y_test.to_numpy() == test_predictions

            )

        })

        prediction_df.to_csv(

            directories["predictions"] /
            f"{name}_NCBI_Internal_Test_Predictions.tsv",

            sep="\t",

            index=False

        )

        #F.ROC curve data

        fpr, tpr, _ = roc_curve(

            y_test,

            test_prob

        )

        roc_data[name] = (

            fpr,

            tpr,

            test_metrics["ROC_AUC"]

        )

        #G.PR curve data

        precision, recall, _ = precision_recall_curve(

            y_test,

            test_prob

        )

        pr_data[name] = (

            precision,

            recall,

            test_metrics["Average_Precision"]

        )

        #H.Calibration data

        calibration_data[name] = (

            y_test.to_numpy(),

            test_prob

        )

        #I.Extract selected features

        selected = save_model_information(

            name,

            best_model,

            feature_names,

            directories,

            "Internal_Model"

        )

        selected_features[name] = selected

        logging.info(
            "%s selected features: %d",
            name,
            len(selected)
        )

        #J.Save internal model

        ##This model is fitted only on the training dataset
        ##It is the model corresponding to internal test evaluation

        internal_model_path = (

            directories["models"] /

            f"{name}_Best_Internal_Model.pkl"

        )

        joblib.dump(

            best_model,

            internal_model_path

        )

        #K.Final development model

        ##Fit the selected pipeline on the complete NCBI dataset
        ##This model is intended for subsequent external validation
        ##Do not evaluate it on the same complete NCBI dataset

        logging.info(
            "%s: fitting final development model.",
            name
        )

        final_model = Pipeline(

            steps=best_model.steps

        )

        final_model.fit(

            X,

            y

        )

        final_model_path = (

            directories["models"] /

            f"{name}_Final_NCBI_Development_Model.pkl"

        )

        joblib.dump(

            final_model,

            final_model_path

        )

        model_paths[name] = {

            "Internal_Model": str(internal_model_path),

            "Final_Development_Model": str(final_model_path)

        }

        # Save final model feature selection.

        final_selected = save_model_information(

            name,

            final_model,

            feature_names,

            directories,

            "Final_Model"

        )

        logging.info(
            "%s final model saved.",
            name
        )

    #L.Save combined results

    pd.DataFrame(
        internal_metrics_all
    ).to_csv(

        directories["results"] /
        "NCBI_Internal_Test_Performance_All_Models.tsv",

        sep="\t",

        index=False

    )

    pd.DataFrame(
        nested_metrics_all
    ).to_csv(

        directories["results"] /
        "NCBI_Nested_Training_CV_Performance_All_Models.tsv",

        sep="\t",

        index=False

    )

    with open(

        directories["models"] /
        "NCBI_Best_Hyperparameters.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            best_parameters,

            file,

            indent=4

        )

    joblib.dump(

        selected_features,

        directories["models"] /
        "NCBI_Selected_Features_By_Model.pkl"

    )

    return {

        "internal_metrics": internal_metrics_all,

        "nested_metrics": nested_metrics_all,

        "best_parameters": best_parameters,

        "selected_features": selected_features,

        "roc_data": roc_data,

        "pr_data": pr_data,

        "calibration_data": calibration_data,

        "model_paths": model_paths

    }


#Publication-quality figures

def generate_performance_figure(

    results,

    y_test,

    directories

):

    roc_data = results["roc_data"]

    pr_data = results["pr_data"]

    calibration_data = results["calibration_data"]

    fig, axes = plt.subplots(

        1,

        3,

        figsize=(24, 9)

    )

    #Panel A: ROC curves

    for name, (fpr, tpr, auc_value) in roc_data.items():

        axes[0].plot(

            fpr,

            tpr,

            linewidth=2.2,

            label=f"{name} (AUC={auc_value:.3f})"

        )

    axes[0].plot(

        [0, 1],

        [0, 1],

        "k--",

        linewidth=1.2

    )

    axes[0].set_title("ROC curves")

    axes[0].set_xlabel("False positive rate")

    axes[0].set_ylabel("True positive rate")

    axes[0].set_xlim(0, 1)

    axes[0].set_ylim(0, 1.02)

    axes[0].legend(
        frameon=False,
        loc="lower right"
    )

    #Panel B: PR curves

    for name, (precision, recall, ap) in pr_data.items():

        axes[1].plot(

            recall,

            precision,

            linewidth=2.2,

            label=f"{name} (AP={ap:.3f})"

        )

    positive_prevalence = float(
        np.mean(y_test)
    )

    axes[1].axhline(

        positive_prevalence,

        color="black",

        linestyle="--",

        linewidth=1.2,

        label=f"Prevalence={positive_prevalence:.3f}"

    )

    axes[1].set_title("Precision-recall curves")

    axes[1].set_xlabel("Recall")

    axes[1].set_ylabel("Precision")

    axes[1].set_xlim(0, 1)

    axes[1].set_ylim(0, 1.02)

    axes[1].legend(
        frameon=False,
        loc="best"
    )

    #Panel C: calibration curves

    for name, (y_true, y_prob) in calibration_data.items():

        fraction_positive, mean_predicted = calibration_curve(

            y_true,

            y_prob,

            n_bins=10,

            strategy="uniform"

        )

        axes[2].plot(

            mean_predicted,

            fraction_positive,

            marker="o",

            linewidth=2.2,

            label=name

        )

    axes[2].plot(

        [0, 1],

        [0, 1],

        "k--",

        linewidth=1.2,

        label="Perfect calibration"

    )

    axes[2].set_title("Calibration curves")

    axes[2].set_xlabel("Mean predicted probability")

    axes[2].set_ylabel("Observed positive fraction")

    axes[2].set_xlim(0, 1)

    axes[2].set_ylim(0, 1)

    axes[2].legend(
        frameon=False,
        loc="best"
    )

    #Individual panel font settings

    panel_font_settings = {

        "A": {

            "title": 22,

            "axis_label": 19,

            "tick_label": 16,

            "legend": 14

        },

        "B": {

            "title": 22,

            "axis_label": 19,

            "tick_label": 16,

            "legend": 14

        },

        "C": {

            "title": 22,

            "axis_label": 19,

            "tick_label": 16,

            "legend": 14

        }

    }

    for ax, panel in zip(

        axes,

        ["A", "B", "C"]

    ):

        settings = panel_font_settings[panel]

        ax.title.set_fontname("Times New Roman")

        ax.title.set_fontsize(settings["title"])

        ax.title.set_fontweight("bold")

        ax.xaxis.label.set_fontname("Times New Roman")

        ax.xaxis.label.set_fontsize(settings["axis_label"])

        ax.xaxis.label.set_fontweight("bold")

        ax.yaxis.label.set_fontname("Times New Roman")

        ax.yaxis.label.set_fontsize(settings["axis_label"])

        ax.yaxis.label.set_fontweight("bold")

        ax.tick_params(

            axis="both",

            which="major",

            labelsize=settings["tick_label"]

        )

        for tick in (

            ax.get_xticklabels()

            + ax.get_yticklabels()

        ):

            tick.set_fontname("Times New Roman")

            tick.set_fontsize(settings["tick_label"])

        legend = ax.get_legend()

        if legend is not None:

            for item in legend.get_texts():

                item.set_fontname("Times New Roman")

                item.set_fontsize(settings["legend"])

    #Panel labels

    for ax, letter in zip(

        axes,

        ["A", "B", "C"]

    ):

        ax.text(

            -0.12,

            1.06,

            letter,

            transform=ax.transAxes,

            fontsize=24,

            fontweight="bold",

            fontname="Times New Roman"

        )

    plt.tight_layout()

    fig.savefig(

        directories["figures"] /
        "NCBI_Internal_Test_ROC_PR_Calibration.png",

        dpi=600,

        bbox_inches="tight"

    )

    fig.savefig(

        directories["figures"] /
        "NCBI_Internal_Test_ROC_PR_Calibration.pdf",

        bbox_inches="tight"

    )

    plt.close(fig)

    logging.info(
        "Performance figure generated."
    )


#Model metadata & reproducibility

def get_package_version(package):

    try:

        return version(package)

    except PackageNotFoundError:

        return "Not installed"


def save_model_metadata(

    X,

    y,

    X_train,

    X_test,

    feature_names,

    results,

    args,

    directories

):

    metadata = {

        "Dataset": "NCBI",

        "Random_State": RANDOM_STATE,

        "Test_Size": args.test_size,

        "Classification_Threshold": args.threshold,

        "Negative_Class": 0,

        "Negative_Class_Name": CLASS_NAMES[0],

        "Positive_Class": 1,

        "Positive_Class_Name": CLASS_NAMES[1],

        "Expected_Features": args.expected_features,

        "Total_Genomes": len(X),

        "Training_Genomes": len(X_train),

        "Internal_Test_Genomes": len(X_test),

        "Total_Features": len(feature_names),

        "Cross_Validation_Folds": args.cv_folds,

        "Search_Iterations": args.search_iterations,

        "Bootstrap_Replicates": args.bootstrap,

        "Confidence_Level": CONFIDENCE_LEVEL,

        "Class_Distribution": {

            CLASS_NAMES[int(label)]: int(count)

            for label, count in y.value_counts().items()

        },

        "Selected_Features_By_Model": {

            name: len(features)

            for name, features in results[
                "selected_features"
            ].items()

        },

        "Feature_Names": feature_names,

        "Best_Hyperparameters": results[
            "best_parameters"
        ],

        "Model_Paths": results[
            "model_paths"
        ],

        "Package_Versions": {

            "Python": sys.version,

            "numpy": get_package_version("numpy"),

            "pandas": get_package_version("pandas"),

            "scipy": get_package_version("scipy"),

            "scikit-learn": get_package_version("scikit-learn"),

            "xgboost": get_package_version("xgboost"),

            "matplotlib": get_package_version("matplotlib"),

            "joblib": get_package_version("joblib")

        }

    }

    with open(

        directories["models"] /
        "NCBI_Model_Development_Metadata.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            make_json_serializable(metadata),

            file,

            indent=4

        )

    joblib.dump(

        feature_names,

        directories["models"] /
        "NCBI_Feature_Names.pkl"

    )

    pd.DataFrame({

        "Feature": feature_names

    }).to_csv(

        directories["models"] /
        "NCBI_Feature_Names.tsv",

        sep="\t",

        index=False

    )


#Generate final pipeline summary

def generate_final_summary(

    df,

    X,

    X_train,

    X_test,

    y,

    y_train,

    y_test,

    results,

    directories,

    args

):

    summary = {

        "Total_NCBI_Genomes": len(df),

        "Training_Genomes": len(X_train),

        "Internal_Test_Genomes": len(X_test),

        "Total_Gene_Features": X.shape[1],

        "Susceptible_Genomes": int((y == 0).sum()),

        "Resistant_Genomes": int((y == 1).sum()),

        "Training_Susceptible": int((y_train == 0).sum()),

        "Training_Resistant": int((y_train == 1).sum()),

        "Test_Susceptible": int((y_test == 0).sum()),

        "Test_Resistant": int((y_test == 1).sum()),

        "Classification_Threshold": args.threshold,

        "CV_Folds": args.cv_folds,

        "Search_Iterations": args.search_iterations,

        "Bootstrap_Replicates": args.bootstrap,

        "Models_Developed": len(results["model_paths"]),

        "Algorithms": ", ".join(
            results["model_paths"].keys()
        )

    }

    summary_df = pd.DataFrame(

        list(summary.items()),

        columns=[
            "Metric",
            "Value"
        ]

    )

    summary_df.to_csv(

        directories["results"] /
        "NCBI_Model_Development_Summary.tsv",

        sep="\t",

        index=False

    )


#Main pipeline

def main():

    args = parse_arguments()

    validate_settings(args)

    global N_SPLITS

    N_SPLITS = args.cv_folds

    global CLASSIFICATION_THRESHOLD

    CLASSIFICATION_THRESHOLD = args.threshold

    global CONFIDENCE_LEVEL

    configure_logging(args.output)

    directories = {

        "output": args.output,

        "data": args.output / "data_splits",

        "models": args.output / "saved_models",

        "results": args.output / "results",

        "search": args.output / "hyperparameter_search",

        "figures": args.output / "figures",

        "predictions": args.output / "predictions"

    }

    logging.info("=" * 75)

    logging.info(
        "NCBI MACHINE LEARNING MODEL DEVELOPMENT"
    )

    logging.info("=" * 75)

    #A.Load dataset

    df = load_dataset(
        args.input
    )

    #B.Validate features & labels

    X, y, feature_names = prepare_features(

        df,

        args.expected_features

    )

    logging.info(
        "Original dataset dimensions: %s",
        df.shape
    )

    logging.info(
        "Number of gene features: %d",
        len(feature_names)
    )

    #C.Save feature names

    joblib.dump(

        feature_names,

        directories["models"] /
        "NCBI_Original_Feature_Names.pkl"

    )

    #D.Class distribution

    class_summary = summarize_class_distribution(y)

    class_summary.to_csv(

        directories["results"] /
        "NCBI_Class_Distribution.tsv",

        sep="\t",

        index=False

    )

    #E.Create stratified 80:20 split

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df

    ) = create_data_split(

        df,

        X,

        y,

        directories,

        args.test_size

    )

    #F.Save training feature names

    pd.DataFrame({

        "Feature": feature_names

    }).to_csv(

        directories["models"] /
        "NCBI_Training_Feature_Names.tsv",

        sep="\t",

        index=False

    )

    #G.Model development

    results = develop_models(

        X,

        y,

        X_train,

        X_test,

        y_train,

        y_test,

        train_df,

        test_df,

        feature_names,

        directories,

        args

    )

    #H.Generate performance figure

    generate_performance_figure(

        results,

        y_test,

        directories

    )

    #I.Save model metadata

    save_model_metadata(

        X,

        y,

        X_train,

        X_test,

        feature_names,

        results,

        args,

        directories

    )

    #J.Save pipeline summary

    generate_final_summary(

        df,

        X,

        X_train,

        X_test,

        y,

        y_train,

        y_test,

        results,

        directories,

        args

    )

    #K.Final report

    logging.info("=" * 75)

    logging.info(
        "NCBI MODEL DEVELOPMENT COMPLETED SUCCESSFULLY"
    )

    logging.info("=" * 75)

    logging.info(
        "Original genomes: %d",
        len(X)
    )

    logging.info(
        "Training genomes: %d",
        len(X_train)
    )

    logging.info(
        "Internal test genomes: %d",
        len(X_test)
    )

    logging.info(
        "Gene features: %d",
        X.shape[1]
    )

    logging.info(
        "Models developed: %s",
        list(results["model_paths"].keys())
    )

    logging.info(
        "Output directory: %s",
        args.output.resolve()
    )

    for name, paths in results["model_paths"].items():

        logging.info(
            "%s final development model: %s",
            name,
            paths["Final_Development_Model"]
        )

    logging.info("=" * 75)


#Script entry point

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logging.exception(
            "NCBI model development failed: %s",
            error
        )

        sys.exit(1)

#End of script

