#!/usr/bin/env bash

# Project: Bacterial Pangenome Analysis
# Module: Bacterial Pangenome Construction
# Script: 03_panaroo.sh
# Software: Panaroo 1.6.0
# Description: Construct separate bacterial pangenomes using Prokka GFF3 annotation files from NCBI RefSeq and BV-BRC datasets.

set -Eeuo pipefail

#Step 1: Identify project directories

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

#Step 2: Activate Panaroo environment

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

if ! conda activate panaroo_env; then
    echo "ERROR: Unable to activate panaroo_env."
    exit 1
fi

#Step 3: Verify Panaroo installation

if ! command -v panaroo >/dev/null 2>&1; then
    echo "ERROR: Panaroo is not installed in panaroo_env."
    exit 1
fi

echo ""
echo "BACTERIAL PANGENOME CONSTRUCTION"

echo ""
echo "Panaroo version:"

if ! panaroo --version; then
    echo "ERROR: Unable to determine Panaroo version."
    exit 1
fi

#Step 4: Define analysis parameters

THREADS=8

CLEAN_MODE="strict"
CORE_THRESHOLD=0.99

#Step 5: Define input and output directories

INPUT_BASE="$PROJECT_ROOT/02_Prokka/results"

OUTPUT_BASE="$SCRIPT_DIR/results"

if ! mkdir -p "$OUTPUT_BASE"; then
    echo "ERROR: Unable to create output directory."
    exit 1
fi

#Step 6: Define pangenome analysis function

run_panaroo() {

    local dataset="$1"

    local input_dir="$INPUT_BASE/$dataset"
    local output_dir="$OUTPUT_BASE/$dataset"

    local summary_file
    local log_file
    local gff
    local genome_name
    local file
    local version

    local -a gff_files=()

    local -A seen_genomes=()

    echo ""
    echo "Processing dataset: $dataset"

    ##Check input directory

    if [[ ! -d "$input_dir" ]]; then

        echo "ERROR: Prokka input directory not found:"
        echo "$input_dir"

        return 1

    fi

    ##Identify Prokka GFF3 annotation files

    shopt -s nullglob

    gff_files=(
        "$input_dir"/*/*.gff
    )

    shopt -u nullglob

    if [[ ${#gff_files[@]} -eq 0 ]]; then

        echo "ERROR: No Prokka GFF files found in $input_dir"

        return 1

    fi

    ##Validate annotation files

    for gff in "${gff_files[@]}"; do

        if [[ ! -s "$gff" ]]; then

            echo "ERROR: Empty GFF file: $gff"

            return 1

        fi

        genome_name="$(basename "$gff" .gff)"

        ###Check duplicate genome identifiers

        if [[ -n "${seen_genomes[$genome_name]+x}" ]]; then

            echo "ERROR: Duplicate genome identifier: $genome_name"

            return 1

        fi

        seen_genomes["$genome_name"]=1

        ###Validate GFF3 header

        if ! grep -q '^##gff-version 3$' "$gff"; then

            echo "ERROR: Invalid or missing GFF3 header: $gff"

            return 1

        fi

        ###Verify that CDS annotations are present

        if ! awk -F '\t' '
            $3 == "CDS" {
                found = 1
                exit
            }
            END {
                exit !found
            }
        ' "$gff"; then

            echo "ERROR: No CDS annotations found in: $gff"

            return 1

        fi

    done

    echo "Number of genomes: ${#gff_files[@]}"

    ##Prevent accidental overwriting of previous results

    if [[ -d "$output_dir" ]] &&
       [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then

        echo "ERROR: Output directory is not empty:"
        echo "$output_dir"

        echo "Inspect or archive previous results before rerunning."

        return 1

    fi

    ##Create output directory

    if ! mkdir -p "$output_dir"; then

        echo "ERROR: Unable to create output directory."

        return 1

    fi

    ##Define log and summary files

    log_file="$output_dir/panaroo.log"

    summary_file="$output_dir/panaroo_summary.txt"

    ##Record Panaroo version

    version="$(panaroo --version 2>&1)" || {
        echo "ERROR: Unable to determine Panaroo version."
        return 1
    }

    #Step 7: Run Panaroo

    echo ""
    echo "Starting Panaroo analysis for $dataset..."

    echo "Cleaning mode: $CLEAN_MODE"
    echo "Core threshold: $CORE_THRESHOLD"
    echo "Threads: $THREADS"

    if ! panaroo \
        -i "${gff_files[@]}" \
        -o "$output_dir" \
        --clean-mode "$CLEAN_MODE" \
        --remove-invalid-genes \
        --core_threshold "$CORE_THRESHOLD" \
        -t "$THREADS" \
        -a core \
        2>&1 | tee "$log_file"; then

        echo "ERROR: Panaroo analysis failed for $dataset."

        return 1

    fi

    echo ""
    echo "Panaroo analysis completed successfully."

    #Step 8: Verify essential Panaroo outputs

    local required_files=(
        "gene_presence_absence.csv"
        "gene_data.csv"
        "final_graph.gml"
        "struct_presence_absence.Rtab"
        "core_gene_alignment.aln"
    )

    echo ""
    echo "Verifying Panaroo output files..."

    for file in "${required_files[@]}"; do

        if [[ ! -s "$output_dir/$file" ]]; then

            echo "ERROR: Required Panaroo output missing or empty:"
            echo "$output_dir/$file"

            return 1

        fi

        echo "Verified: $file"

    done

    #Step 9: Generate dataset summary

    {
        echo "Panaroo Pangenome Analysis Summary"
        echo ""
        echo "Dataset: $dataset"
        echo "Panaroo version: $version"
        echo "Number of input genomes: ${#gff_files[@]}"
        echo "Cleaning mode: $CLEAN_MODE"
        echo "Core genome threshold: $CORE_THRESHOLD"
        echo "Threads: $THREADS"
        echo "Alignment: Core genome"
        echo "Output directory: $output_dir"
        echo "Status: Completed"
    } > "$summary_file"

    #Step 10: Display dataset results

    echo ""
    echo "Panaroo output files:"

    ls -lh "$output_dir"

    echo ""
    echo "Pangenome analysis completed for $dataset."

    echo "Summary file: $summary_file"

}

#Step 11: Process NCBI RefSeq dataset

if ! run_panaroo "NCBI_RefSeq"; then

    echo "ERROR: NCBI RefSeq pangenome analysis failed."

    exit 1

fi

#Step 12: Process BV-BRC dataset

if ! run_panaroo "BV-BRC"; then

    echo "ERROR: BV-BRC pangenome analysis failed."

    exit 1

fi

#Step 13: Final completion message

echo ""
echo "Panaroo pangenome analysis completed successfully."

echo ""
echo "Datasets processed:"
echo "NCBI RefSeq"
echo "BV-BRC"

echo ""
echo "Cleaning mode: $CLEAN_MODE"
echo "Core genome threshold: $CORE_THRESHOLD"
echo "Threads used: $THREADS"
echo "Alignment type: Core genome"

echo ""
echo "Results directory: $OUTPUT_BASE"

echo ""
echo "All pangenome construction operations completed successfully."