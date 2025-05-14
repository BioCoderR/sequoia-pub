#!/bin/bash


INPUT_CSV="/projects/conco/gundla/root/uniglacier/model_src/sequoia-pub/examples/ref_file_filtered.csv"
OUTPUT_CSV="/projects/conco/gundla/root/uniglacier/model_src/sequoia-pub/evaluation/GOI_list.csv"

echo "gene" > "$OUTPUT_CSV"

head -n 1 "$INPUT_CSV" | \
    tr ',' '\n' | \
    grep "^rna_" | \
    sed 's/^rna_//' | \
    sort | \
    uniq >> "$OUTPUT_CSV"

echo "Gene list processed and saved to $OUTPUT_CSV"