
#!/usr/bin/env bash

# Project: Bacterial Pangenome Analysis
# Module: Genome Assembly Quality Assessment
# Script: 01_quast.sh
# Software: QUAST 5.2.0
# Description: Assess bacterial genome assembly quality using NCBI RefSeq and BV-BRC datasets.

set -Eeuo pipefail

#Step 1: Identify project directories

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

#Step 2: Activate QUAST environment

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

if ! conda activate quast_env; then
    echo "ERROR: Unable to activate quast_env."
    exit 1
fi

#Step 3: Verify software installation

if ! command -v quast.py >/dev/null 2>&1; then
    echo "ERROR: QUAST is not installed in quast_env."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python3 is not installed."
    exit 1
fi

echo ""
echo "GENOME ASSEMBLY QUALITY ASSESSMENT"

echo ""
echo "QUAST version:"
quast.py --version

echo ""
echo "Python version:"
python3 --version

#Step 4: Define quality-control parameters

THREADS=4
MIN_CONTIG=1

MIN_GENOME_LENGTH=2600000
MAX_GENOME_LENGTH=3000000

MAX_CONTIGS=200
MIN_N50=50000

#Step 5: Define input and output directories

NCBI_DIR="$PROJECT_ROOT/data/NCBI_RefSeq/FASTA"
BV_BRC_DIR="$PROJECT_ROOT/data/BV-BRC/FASTA"

OUTPUT_DIR="$SCRIPT_DIR/results"

mkdir -p "$OUTPUT_DIR" || {
    echo "ERROR: Unable to create output directory."
    exit 1
}

#Step 6: Define genome quality assessment function

run_quast() {

    local input_dir="$1"
    local dataset="$2"

    local result_dir="$OUTPUT_DIR/$dataset"
    local quast_dir="$result_dir/quast"
    local qc_dir="$result_dir/QC"
    local retained_dir="$qc_dir/retained_genomes"

    local report_file

    echo ""
    echo "Processing dataset: $dataset"

    ##Check input directory

    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: Input directory not found: $input_dir"
        return 1
    fi

    ##Identify genome FASTA files

    local files=()

    shopt -s nullglob

    files=(
        "$input_dir"/*.fna
        "$input_dir"/*.fasta
        "$input_dir"/*.fa
    )

    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "ERROR: No FASTA files found in $input_dir"
        return 1
    fi

    ##Check for duplicate genome filenames

    local -A seen_files=()

    local genome
    local filename

    for genome in "${files[@]}"; do

        filename="$(basename "$genome")"

        if [[ -n "${seen_files[$filename]+x}" ]]; then
            echo "ERROR: Duplicate genome filename: $filename"
            return 1
        fi

        seen_files["$filename"]=1

    done

    echo "Input directory: $input_dir"
    echo "Number of genomes: ${#files[@]}"

    ##Create output directories

    mkdir -p "$qc_dir" "$retained_dir" || {
        echo "ERROR: Unable to create output directories."
        return 1
    }

    ##Remove previous retained genome copies

    if ! find "$retained_dir" -maxdepth 1 -type f -delete; then
        echo "ERROR: Unable to clean retained genome directory."
        return 1
    fi

    ##Remove previous QUAST output to prevent stale reports

    if [[ -d "$quast_dir" ]]; then
        rm -rf -- "$quast_dir" || {
            echo "ERROR: Unable to clean previous QUAST results."
            return 1
        }
    fi

    mkdir -p "$quast_dir" || {
        echo "ERROR: Unable to create QUAST output directory."
        return 1
    }

    ##Run QUAST

    echo ""
    echo "Running QUAST for $dataset..."

    if ! quast.py \
        "${files[@]}" \
        -o "$quast_dir" \
        --threads "$THREADS" \
        --min-contig "$MIN_CONTIG" \
        --force; then

        echo "ERROR: QUAST failed for $dataset."
        return 1
    fi

    ##Verify QUAST report

    report_file="$quast_dir/transposed_report.tsv"

    if [[ ! -s "$report_file" ]]; then
        echo "ERROR: QUAST report was not generated."
        return 1
    fi

    echo "QUAST analysis completed successfully."

    #Step 7: Apply genome quality-control criteria

    echo ""
    echo "Applying genome quality-control criteria..."

    if ! python3 - \
        "$report_file" \
        "$qc_dir" \
        "$retained_dir" \
        "$MIN_GENOME_LENGTH" \
        "$MAX_GENOME_LENGTH" \
        "$MAX_CONTIGS" \
        "$MIN_N50" \
        "${files[@]}" <<'PYTHON'

import csv
import math
import shutil
import sys

from pathlib import Path


##Read input parameters

report_file = Path(sys.argv[1])
qc_dir = Path(sys.argv[2])
retained_dir = Path(sys.argv[3])

min_length = int(sys.argv[4])
max_length = int(sys.argv[5])
max_contigs = int(sys.argv[6])
min_n50 = int(sys.argv[7])

input_files = [Path(p) for p in sys.argv[8:]]

if not input_files:
    raise SystemExit("ERROR: No input genomes supplied.")

if min_length > max_length:
    raise SystemExit("ERROR: Minimum genome length exceeds maximum.")

if max_contigs < 0 or min_n50 < 0:
    raise SystemExit("ERROR: Invalid QC thresholds.")


##Normalize column names

def normalize(value):

    return "".join(
        char.lower()
        for char in value.strip()
        if char.isalnum()
    )


def find_column(headers, candidates):

    normalized = {
        normalize(header): header
        for header in headers
    }

    for candidate in candidates:

        if candidate in normalized:
            return normalized[candidate]

    return None


##Map genome filenames

genome_map = {}
stem_map = {}

for genome in input_files:

    if genome.name in genome_map:

        raise SystemExit(
            f"ERROR: Duplicate genome filename: {genome.name}"
        )

    genome_map[genome.name] = genome

    stem_map.setdefault(genome.stem, []).append(genome)


##Read QUAST report

if not report_file.is_file() or report_file.stat().st_size == 0:

    raise SystemExit("ERROR: QUAST report is missing or empty.")

with report_file.open(
    newline="",
    encoding="utf-8-sig"
) as handle:

    reader = csv.DictReader(handle, delimiter="\t")

    if not reader.fieldnames:

        raise SystemExit("ERROR: Empty QUAST report.")

    headers = reader.fieldnames

    assembly_col = find_column(
        headers, ["assembly", "assemblies"]
    )

    contigs_col = find_column(
        headers, ["#contigs", "contigs", "numberofcontigs"]
    )

    length_col = find_column(
        headers, ["totallength", "length"]
    )

    n50_col = find_column(
        headers, ["n50"]
    )

    gc_col = find_column(
        headers, ["gc", "gcpercent"]
    )

    required = {
        "assembly": assembly_col,
        "contigs": contigs_col,
        "total length": length_col,
        "N50": n50_col,
        "GC content": gc_col
    }

    missing = [
        name
        for name, column in required.items()
        if column is None
    ]

    if missing:

        raise SystemExit(
            "ERROR: Missing QUAST columns: "
            + ", ".join(missing)
        )

    rows = list(reader)


##Assess genome quality

results = []
processed = set()

for row in rows:

    assembly_name = (row[assembly_col] or "").strip()

    if not assembly_name:
        raise SystemExit("ERROR: Empty assembly name in QUAST report.")

    assembly_path = Path(assembly_name)

    genome = genome_map.get(assembly_path.name)

    ###Match by filename stem when necessary

    if genome is None:

        matches = stem_map.get(assembly_path.stem, [])

        if len(matches) > 1:

            raise SystemExit(
                f"ERROR: Ambiguous genome match: {assembly_name}"
            )

        if len(matches) == 1:
            genome = matches[0]

    if genome is None:

        raise SystemExit(
            f"ERROR: Cannot match genome: {assembly_name}"
        )

    if genome in processed:

        raise SystemExit(
            f"ERROR: Duplicate assembly in QUAST report: {genome.name}"
        )

    try:

        contigs = int(
            (row[contigs_col] or "").replace(",", "").strip()
        )

        genome_length = int(
            float(
                (row[length_col] or "").replace(",", "").strip()
            )
        )

        n50 = int(
            float(
                (row[n50_col] or "").replace(",", "").strip()
            )
        )

        gc_content = float(
            (row[gc_col] or "")
            .replace(",", "")
            .replace("%", "")
            .strip()
        )

        if not all(
            math.isfinite(value)
            for value in (genome_length, n50, gc_content)
        ):

            raise ValueError("Non-finite assembly statistic")

        if not 0 <= gc_content <= 100:

            raise ValueError("Invalid GC content")

        if contigs < 0 or genome_length < 0 or n50 < 0:

            raise ValueError("Negative assembly statistic")

    except (ValueError, AttributeError, TypeError) as error:

        raise SystemExit(
            f"ERROR: Invalid statistics for {genome.name}: {error}"
        )

    processed.add(genome)

    ##Apply quality-control thresholds

    reasons = []

    if not min_length <= genome_length <= max_length:

        reasons.append("genome_length_outside_range")

    if contigs > max_contigs:

        reasons.append("too_many_contigs")

    if n50 < min_n50:

        reasons.append("n50_below_threshold")

    status = "RETAINED" if not reasons else "EXCLUDED"

    results.append({

        "assembly": genome.name,

        "genome_length_bp": genome_length,

        "number_of_contigs": contigs,

        "n50_bp": n50,

        "gc_content_percent": gc_content,

        "qc_status": status,

        "exclusion_reason": ";".join(reasons)

    })


##Verify that all input genomes were assessed

missing_genomes = set(input_files) - processed

if missing_genomes:

    raise SystemExit(
        "ERROR: Genomes missing from QUAST report: "
        + ", ".join(
            sorted(g.name for g in missing_genomes)
        )
    )

if len(results) != len(input_files):

    raise SystemExit(
        "ERROR: Input and assessed genome counts differ."
    )


##Generate retained and excluded lists

retained = [
    row["assembly"]
    for row in results
    if row["qc_status"] == "RETAINED"
]

excluded = [
    row["assembly"]
    for row in results
    if row["qc_status"] == "EXCLUDED"
]


##Generate genome QC report

qc_report = qc_dir / "genome_qc_results.csv"

fieldnames = [
    "assembly",
    "genome_length_bp",
    "number_of_contigs",
    "n50_bp",
    "gc_content_percent",
    "qc_status",
    "exclusion_reason"
]

with qc_report.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


##Copy retained genomes

for row in results:

    if row["qc_status"] == "RETAINED":

        genome = genome_map[row["assembly"]]

        shutil.copy2(
            genome,
            retained_dir / genome.name
        )


##Generate retained and excluded text files

(qc_dir / "retained_genomes.txt").write_text(
    "\n".join(retained) + ("\n" if retained else ""),
    encoding="utf-8"
)

(qc_dir / "excluded_genomes.txt").write_text(
    "\n".join(excluded) + ("\n" if excluded else ""),
    encoding="utf-8"
)


##Print quality-control summary

print("\nGENOME QUALITY-CONTROL SUMMARY")
print("--------------------------------")

print(f"Total genomes: {len(results)}")

print(f"Retained genomes: {len(retained)}")

print(f"Excluded genomes: {len(excluded)}")

print(f"QC report: {qc_report}")

print(f"Retained directory: {retained_dir}")

PYTHON
    then
        echo "ERROR: Genome quality control failed for $dataset."
        return 1
    fi

    echo ""
    echo "Quality-control completed successfully for $dataset."

}


#Step 8: Process NCBI RefSeq genomes

if ! run_quast "$NCBI_DIR" "NCBI_RefSeq"; then

    echo "ERROR: NCBI RefSeq processing failed."
    exit 1

fi


#Step 9: Process BV-BRC genomes

if ! run_quast "$BV_BRC_DIR" "BV-BRC"; then

    echo "ERROR: BV-BRC processing failed."
    exit 1

fi


#Step 10: Display final results

echo ""
echo "GENOME QUALITY ASSESSMENT COMPLETED"

echo "Datasets: NCBI RefSeq and BV-BRC"

echo "Results directory: $OUTPUT_DIR"

echo "Threads: $THREADS"

echo "Minimum contig length: $MIN_CONTIG bp"

echo "Genome length: ${MIN_GENOME_LENGTH}-${MAX_GENOME_LENGTH} bp"

echo "Maximum contigs: $MAX_CONTIGS"

echo "Minimum N50: $MIN_N50 bp"

echo ""
echo "NCBI RefSeq QC report:"
echo "$OUTPUT_DIR/NCBI_RefSeq/QC/genome_qc_results.csv"

echo ""
echo "BV-BRC QC report:"
echo "$OUTPUT_DIR/BV-BRC/QC/genome_qc_results.csv"

echo ""
echo "All genome quality-control operations completed successfully."