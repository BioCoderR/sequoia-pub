#!/bin/bash
#SBATCH --job-name=ViS
#SBATCH --output=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/visualize_GOI_out_%j.out
#SBATCH --error=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/visualize_GOI_err_%j.err
#SBATCH --partition=GPUampere
#SBATCH --gres=gpu:2
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=32
#SBATCH --time=25-00:00:00

GENES_IN_USE="all"
FEATURE_USE="uni"
MODEL_TYPE="vis"

declare -A PROJECT_MAP

awk -F',' 'NR > 1 {
    split($1, id_parts, "-");
    wsi_file_name = id_parts[1] "-" id_parts[2] "-" id_parts[3] "-" id_parts[4] "-" id_parts[5];
    project = $NF;
    print wsi_file_name, project;
}' /projects/conco/gundla/root/uniglacier/model_src/sequoia-pub/examples/ref_file_vis.csv | \
while read -r wsi_file_name tcga_project; do
    PROJECT_MAP["$wsi_file_name"]="$tcga_project"
done

SLIDE_PATH="/projects/conco/gundla/root/uniglacier/data/internal/"
SLIDES=($(ls "$SLIDE_PATH"/*.svs))

for slide_path in "${SLIDES[@]}"; do
    slide_filename=$(basename "$slide_path")
    slide_id="${slide_filename%.*}"
    wsi_file_name=$(echo "$slide_id" | cut -d'-' -f1-5)

    TCGA_PROJECT="${PROJECT_MAP[$wsi_file_name]}"
    STUDY=$(echo "$TCGA_PROJECT" | cut -d'-' -f2 | tr '[:upper:]' '[:lower:]')

    if [[ "$TCGA_PROJECT" == "TCGA-GBM" ]]; then
        CHECKPOINT_DIR="/projects/conco/gundla/root/image2st/predictions/sequoia_gbm"
    elif [[ "$TCGA_PROJECT" == "TCGA-LGG" ]]; then
        CHECKPOINT_DIR="/projects/conco/gundla/root/image2st/predictions/sequoia_lgg"
    else
        echo "Unknown project $TCGA_PROJECT for slide $slide_filename"
        continue
    fi

    SAVE_DIR="/projects/conco/gundla/root/uniglacier/models/trained/image2st/vis_all/"

    for fold in {0..4}; do
        echo "Processing slide: $slide_filename (Study: $STUDY)"

        python spatial_vis/visualize.py \
            --study "$STUDY" \
            --checkpoint "/projects/conco/gundla/root/uniglacier/models/trained/image2st/predictions/sequoia_${STUDY}" \
            --model_type "$MODEL_TYPE" \
            --feat_type "$FEATURE_USE" \
            --project "$TCGA_PROJECT" \
            --wsi_file_name "$slide_filename" \
            --slide_path "/projects/conco/gundla/root/uniglacier/models/trained/image2st/Patches_hdf5/" \
            --gene_names "$GENES_IN_USE" \
            --save_folder "/projects/conco/gundla/root/uniglacier/models/trained/image2st/vis_all/" \
            --folds $fold
    done
done