#!/usr/bin/env bash

# Project: Bacterial Pangenome Analysis
# Module: Antimicrobial Resistance Database Preparation
# Script: 04_rgi_card.sh
# Software: RGI 6.0.5
# Database: CARD 3.2.7
# Description: Prepare and verify the local CARD reference database for antimicrobial resistance analysis.

set -Eeuo pipefail

#Step 1: Identify project directories

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

#Step 2: Activate RGI environment

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda is not available."
    exit 1
fi

CONDA_BASE="$(conda info --base)" || {
    echo "ERROR: Unable to identify Conda installation."
    exit 1
}

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: Conda initialization script not found."
    exit 1
fi

source "$CONDA_SH"

if ! conda activate rgi_env; then
    echo "ERROR: Unable to activate rgi_env."
    exit 1
fi

#Step 3: Verify RGI installation

if ! command -v rgi >/dev/null 2>&1; then
    echo "ERROR: RGI is not installed in rgi_env."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python3 is not available in rgi_env."
    exit 1
fi

echo ""
echo "ANTIMICROBIAL RESISTANCE DATABASE PREPARATION"

echo ""
echo "RGI version:"

if ! rgi main --version; then
    echo "ERROR: Unable to determine RGI version."
    exit 1
fi

#Step 4: Define CARD database parameters

CARD_VERSION="3.2.7"

CARD_DIR="$PROJECT_ROOT/database/CARD_${CARD_VERSION}"

CARD_JSON="$CARD_DIR/card.json"

#Step 5: Check CARD database directory

if [[ ! -d "$CARD_DIR" ]]; then

    echo "ERROR: CARD database directory not found."
    echo "Expected location: $CARD_DIR"

    exit 1

fi

#Step 6: Verify CARD database file

if [[ ! -f "$CARD_JSON" || ! -s "$CARD_JSON" ]]; then

    echo "ERROR: CARD database file not found or empty."
    echo "Expected location: $CARD_JSON"

    exit 1

fi

echo ""
echo "CARD database file found:"
echo "$CARD_JSON"

#Step 7: Validate CARD JSON format

echo ""
echo "Validating CARD JSON file..."

if ! python3 -m json.tool "$CARD_JSON" > /dev/null; then

    echo "ERROR: CARD JSON file is invalid or corrupted."

    exit 1

fi

echo "CARD JSON validation completed successfully."

#Step 8: Load CARD database locally

echo ""
echo "Loading CARD ${CARD_VERSION} database..."

if ! rgi load \
    --card_json "$CARD_JSON" \
    --local; then

    echo "ERROR: CARD database loading failed."

    exit 1

fi

echo "CARD database loading completed successfully."

#Step 9: Verify local CARD database

echo ""
echo "Verifying local CARD database..."

if ! rgi database --version --local; then

    echo "ERROR: Local CARD database verification failed."

    exit 1

fi

echo "Local CARD database verification completed successfully."

#Step 10: Display database information

echo ""
echo "CARD DATABASE PREPARATION SUMMARY"

echo "RGI version: 6.0.5"

echo "CARD version: $CARD_VERSION"

echo "Database location: $CARD_JSON"

echo "Database mode: Local"

echo "JSON validation: Passed"

echo "Database loading: Successful"

echo "Database verification: Successful"

#Step 11: Final completion message

echo ""
echo "CARD database preparation completed successfully."

echo ""
echo "The local CARD database is ready for subsequent"
echo "antimicrobial resistance analysis using RGI."