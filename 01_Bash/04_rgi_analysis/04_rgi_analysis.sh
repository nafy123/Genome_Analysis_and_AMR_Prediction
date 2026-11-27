#!/usr/bin/env bash

# Project: Bacterial Pangenome Analysis
# Module: Antimicrobial Resistance Gene Identification
# Script: 04_rgi_analysis.sh
# Software: RGI 6.0.5
# Database: CARD 3.2.7
# Description: Identify antimicrobial resistance genes using Prokka-predicted protein sequences and CARD.
# Datasets: NCBI RefSeq and BV-BRC.

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

echo ""
echo "ANTIMICROBIAL RESISTANCE GENE IDENTIFICATION"

echo ""
echo "RGI version:"
rgi main --version

#Step 4: Define analysis parameters

THREADS=4
CARD_VERSION="3.2.7"

#Step 5: Define input and output directories

INPUT_BASE="$PROJECT_ROOT/02_Prokka/results"
OUTPUT_BASE="$SCRIPT_DIR/results"

if ! mkdir -p "$OUTPUT_BASE"; then
    echo "ERROR: Unable to create output directory."
    exit 1
fi

#Step 6: Verify local CARD database

echo ""
echo "Checking local CARD database..."

if ! rgi database --version --local; then
    echo "ERROR: Local CARD database verification failed."
    echo "Run the CARD database preparation script first."
    exit 1
fi

echo "Local CARD database verification completed."

#Step 7: Define RGI analysis function

run_rgi() {

    local dataset="$1"

    local input_dir="$INPUT_BASE/$dataset"
    local result_dir="$OUTPUT_BASE/$dataset"

    local summary_file="$result_dir/rgi_summary.tsv"

    local faa
    local filename
    local base
    local genome_output
    local output_prefix
    local extension

    local processed=0
    local skipped=0

    local -a faa_files=()

    local -A seen_genomes=()

    echo ""
    echo "Processing dataset: $dataset"

    ##Check input directory

    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: Prokka input directory not found:"
        echo "$input_dir"
        return 1
    fi

    ##Identify Prokka protein FASTA files

    shopt -s nullglob

    faa_files=(
        "$input_dir"/*/*.faa
    )

    shopt -u nullglob

    if [[ ${#faa_files[@]} -eq 0 ]]; then
        echo "ERROR: No Prokka protein files found in $input_dir"
        return 1
    fi

    echo "Number of genomes: ${#faa_files[@]}"

    ##Create output directory

    if ! mkdir -p "$result_dir"; then
        echo "ERROR: Unable to create output directory."
        return 1
    fi

    ##Create dataset summary

    if ! printf "Genome\tStatus\tOutput_directory\n" > "$summary_file"; then
        echo "ERROR: Unable to create RGI summary."
        return 1
    fi

    #Step 8: Analyse individual genomes

    for faa in "${faa_files[@]}"; do

        filename="$(basename "$faa")"
        base="${filename%.faa}"

        ##Check protein input

        if [[ ! -s "$faa" ]]; then
            echo "ERROR: Empty protein file: $faa"
            return 1
        fi

        ##Verify FASTA structure

        if ! awk '
            /^>/ {
                headers++
                next
            }
            /^[[:space:]]*$/ {
                next
            }
            {
                sequences++
            }
            END {
                if (headers == 0 || sequences == 0) {
                    exit 1
                }
            }
        ' "$faa"; then

            echo "ERROR: Invalid or empty protein FASTA file: $faa"
            return 1
        fi

        ##Check duplicate genome identifiers

        if [[ -n "${seen_genomes[$base]+x}" ]]; then
            echo "ERROR: Duplicate genome identifier: $base"
            return 1
        fi

        seen_genomes["$base"]=1

        genome_output="$result_dir/$base"
        output_prefix="$genome_output/${base}_rgi"

        echo ""
        echo "Analysing genome: $base"

        ##Skip previously completed analyses

        if [[ -d "$genome_output" ]]; then

            if [[ -s "${output_prefix}.txt" &&
                  -s "${output_prefix}.json" ]]; then

                echo "RGI analysis already completed. Skipping."

                if ! printf "%s\tSKIPPED_EXISTING\t%s\n" \
                    "$base" "$genome_output" >> "$summary_file"; then

                    echo "ERROR: Unable to update RGI summary."
                    return 1
                fi

                skipped=$((skipped + 1))
                continue

            else

                echo "ERROR: Incomplete previous RGI output:"
                echo "$genome_output"
                echo "Inspect or remove the incomplete directory before rerunning."

                return 1
            fi
        fi

        ##Create genome output directory

        if ! mkdir -p "$genome_output"; then
            echo "ERROR: Unable to create genome output directory."
            return 1
        fi

        ##Run RGI using the local CARD database

        echo "Running RGI with CARD $CARD_VERSION..."

        if ! rgi main \
            --input_sequence "$faa" \
            --output_file "$output_prefix" \
            --input_type protein \
            --local \
            --clean \
            -n "$THREADS" \
            2>&1 | tee "$genome_output/rgi.log"; then

            echo "ERROR: RGI analysis failed for $base."
            return 1
        fi

        #Step 9: Verify RGI output files

        for extension in txt json; do

            if [[ ! -s "${output_prefix}.${extension}" ]]; then
                echo "ERROR: RGI $extension report missing or empty."
                echo "${output_prefix}.${extension}"
                return 1
            fi

        done

        ##Verify execution log

        if [[ ! -s "$genome_output/rgi.log" ]]; then
            echo "ERROR: RGI execution log missing for $base."
            return 1
        fi

        echo "RGI analysis completed successfully: $base"

        if ! printf "%s\tCOMPLETED\t%s\n" \
            "$base" "$genome_output" >> "$summary_file"; then

            echo "ERROR: Unable to update RGI summary."
            return 1
        fi

        processed=$((processed + 1))

    done

    #Step 10: Dataset completion summary

    echo ""
    echo "RGI ANALYSIS SUMMARY: $dataset"

    echo "Total genomes: ${#faa_files[@]}"
    echo "Newly processed: $processed"
    echo "Previously completed: $skipped"
    echo "Output directory: $result_dir"
    echo "Summary file: $summary_file"

    echo "Dataset completed successfully: $dataset"

}

#Step 11: Process NCBI RefSeq genomes

if ! run_rgi "NCBI_RefSeq"; then
    echo "ERROR: NCBI RefSeq RGI analysis failed."
    exit 1
fi

#Step 12: Process BV-BRC genomes

if ! run_rgi "BV-BRC"; then
    echo "ERROR: BV-BRC RGI analysis failed."
    exit 1
fi

#Step 13: Final completion message

echo ""
echo "RGI ANTIMICROBIAL RESISTANCE ANALYSIS COMPLETED"

echo ""
echo "Datasets processed:"
echo "NCBI RefSeq"
echo "BV-BRC"

echo ""
echo "RGI version: 6.0.5"
echo "CARD version: $CARD_VERSION"
echo "Input type: Protein"
echo "Database mode: Local"
echo "Threads per genome: $THREADS"

echo ""
echo "Results directory: $OUTPUT_BASE"

echo ""
echo "Expected outputs:"
echo "RGI tabular reports (.txt)"
echo "RGI JSON reports (.json)"
echo "RGI execution logs (.log)"
echo "Dataset summaries (.tsv)"

echo ""
echo "All antimicrobial resistance analysis operations completed successfully."