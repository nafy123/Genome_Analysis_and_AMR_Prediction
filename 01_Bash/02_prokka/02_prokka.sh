#!/usr/bin/env bash

# Project: Bacterial Pangenome Analysis
# Module: Bacterial Genome Annotation
# Script: 02_prokka.sh
# Software: Prokka 1.15.6
# Organism: Staphylococcus aureus
# Description: Annotate quality-filtered bacterial genome assemblies from NCBI RefSeq and BV-BRC datasets.

set -Eeuo pipefail

#Step 1: Identify project directories

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

#Step 2: Activate Prokka environment

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

if ! conda activate prokka_env; then
    echo "ERROR: Unable to activate prokka_env."
    exit 1
fi

#Step 3: Verify Prokka installation

if ! command -v prokka >/dev/null 2>&1; then
    echo "ERROR: Prokka is not installed in prokka_env."
    exit 1
fi

echo ""
echo "BACTERIAL GENOME ANNOTATION"

echo ""
echo "Prokka version:"

if ! prokka --version; then
    echo "ERROR: Unable to determine Prokka version."
    exit 1
fi

#Step 4: Define annotation parameters

GENUS="Staphylococcus"
SPECIES="aureus"

THREADS=4

#Step 5: Define input and output directories

NCBI_DIR="$PROJECT_ROOT/01_QUAST/results/NCBI_RefSeq/QC/retained_genomes"

BV_BRC_DIR="$PROJECT_ROOT/01_QUAST/results/BV-BRC/QC/retained_genomes"

OUTPUT_DIR="$SCRIPT_DIR/results"

if ! mkdir -p "$OUTPUT_DIR"; then
    echo "ERROR: Unable to create output directory."
    exit 1
fi

#Step 6: Define genome annotation function

run_prokka() {

    local input_dir="$1"
    local result_dir="$2"
    local dataset="$3"

    local summary_file

    local -a files=()

    local -A used_tags=()

    local annotated=0
    local skipped=0

    local fasta
    local filename
    local base
    local prefix
    local locustag
    local genome_output
    local extension

    echo ""
    echo "Processing dataset: $dataset"

    ##Check input directory

    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: Input directory not found: $input_dir"
        return 1
    fi

    ##Identify genome FASTA files

    shopt -s nullglob

    files=(
        "$input_dir"/*.fna
        "$input_dir"/*.fasta
        "$input_dir"/*.fa
    )

    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "ERROR: No genome FASTA files found in $input_dir"
        return 1
    fi

    echo "Input directory: $input_dir"
    echo "Number of genomes: ${#files[@]}"

    ##Create output directory

    if ! mkdir -p "$result_dir"; then
        echo "ERROR: Unable to create $result_dir"
        return 1
    fi

    ##Create annotation summary file

    summary_file="$result_dir/annotation_summary.tsv"

    if ! printf "Genome\tStatus\tOutput_directory\n" > "$summary_file"; then
        echo "ERROR: Unable to create annotation summary."
        return 1
    fi

    #Step 7: Annotate individual genomes

    for fasta in "${files[@]}"; do

        filename="$(basename "$fasta")"

        base="${filename%.fna}"
        base="${base%.fasta}"
        base="${base%.fa}"

        ##Generate a safe annotation prefix

        prefix="$(printf '%s' "$base" | sed 's/[^[:alnum:]_.-]/_/g')"

        ##Generate locus tag

        locustag="$(printf '%s' "$base" | tr -cd '[:alnum:]')"

        if [[ -z "$prefix" || -z "$locustag" ]]; then
            echo "ERROR: Cannot generate annotation identifiers for $filename"
            return 1
        fi

        ##Check locus tag uniqueness

        if [[ -n "${used_tags[$locustag]+x}" ]]; then
            echo "ERROR: Duplicate locus tag: $locustag"
            echo "Please use unique genome identifiers."
            return 1
        fi

        used_tags["$locustag"]=1

        genome_output="$result_dir/$prefix"

        echo ""
        echo "Annotating genome: $filename"

        ##Skip previously completed annotations

        if [[ -d "$genome_output" ]]; then

            if [[ -s "$genome_output/$prefix.gff" &&
                  -s "$genome_output/$prefix.faa" &&
                  -s "$genome_output/$prefix.ffn" &&
                  -s "$genome_output/$prefix.gbk" ]]; then

                echo "Annotation already completed. Skipping."

                if ! printf "%s\tSKIPPED_EXISTING\t%s\n" \
                    "$filename" "$genome_output" >> "$summary_file"; then

                    echo "ERROR: Unable to update annotation summary."
                    return 1
                fi

                skipped=$((skipped + 1))

                continue

            else

                echo "ERROR: Incomplete annotation directory:"
                echo "$genome_output"
                echo "Inspect or remove the incomplete directory before rerunning."

                return 1

            fi

        fi

        ##Run Prokka

        echo "Running Prokka..."

        if ! prokka \
            --outdir "$genome_output" \
            --prefix "$prefix" \
            --locustag "$locustag" \
            --genus "$GENUS" \
            --species "$SPECIES" \
            --kingdom Bacteria \
            --cpus "$THREADS" \
            "$fasta"; then

            echo "ERROR: Prokka failed for $filename."
            echo "Inspect the Prokka output directory for diagnostic information."

            return 1
        fi

        ##Verify essential annotation outputs

        for extension in gff faa ffn gbk; do

            if [[ ! -s "$genome_output/$prefix.$extension" ]]; then

                echo "ERROR: Missing or empty $extension output for $filename"

                return 1

            fi

        done

        echo "Annotation completed successfully: $filename"

        if ! printf "%s\tCOMPLETED\t%s\n" \
            "$filename" "$genome_output" >> "$summary_file"; then

            echo "ERROR: Unable to update annotation summary."
            return 1
        fi

        annotated=$((annotated + 1))

    done

    #Step 8: Display dataset summary

    echo ""
    echo "Dataset annotation summary: $dataset"

    echo "Total input genomes: ${#files[@]}"

    echo "Newly annotated: $annotated"

    echo "Previously completed: $skipped"

    echo "Output directory: $result_dir"

    echo "Annotation summary: $summary_file"

    echo "$dataset annotation completed successfully."

}

#Step 9: Annotate NCBI RefSeq genomes

if ! run_prokka \
    "$NCBI_DIR" \
    "$OUTPUT_DIR/NCBI_RefSeq" \
    "NCBI_RefSeq"; then

    echo "ERROR: NCBI RefSeq annotation failed."
    exit 1

fi

#Step 10: Annotate BV-BRC genomes

if ! run_prokka \
    "$BV_BRC_DIR" \
    "$OUTPUT_DIR/BV-BRC" \
    "BV-BRC"; then

    echo "ERROR: BV-BRC annotation failed."
    exit 1

fi

#Step 11: Final completion message

echo ""
echo "Prokka genome annotation completed successfully."

echo ""
echo "Datasets processed:"
echo "NCBI RefSeq"
echo "BV-BRC"

echo ""
echo "Organism: $GENUS $SPECIES"
echo "Threads per genome: $THREADS"
echo "Results directory: $OUTPUT_DIR"

echo ""
echo "Annotation outputs:"
echo "GFF: Genome annotation"
echo "FAA: Predicted protein sequences"
echo "FFN: Predicted nucleotide sequences"
echo "GBK: GenBank annotation files"

echo ""
echo "All genome annotation operations completed successfully."