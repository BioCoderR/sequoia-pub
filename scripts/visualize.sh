#!/bin/bash
#SBATCH --job-name=ViS
#SBATCH --output=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/visualize_GOI_out_%j.out
#SBATCH --error=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/visualize_GOI_err_%j.err
#SBATCH --partition=GPUampere
#SBATCH --gres=gpu:2
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=32
#SBATCH --time=25-00:00:00

# PATHs
SLIDE_PATH="/projects/conco/gundla/root/uniglacier/data/internal/"
GENES_IN_USE="all"
FEATURE_USE="uni"
MODEL_TYPE="vis"
FOLDS="0,1,2,3,4"

# Process each project
PROJECTS=("TCGA-GBM" "TCGA-LGG")

for TCGA_PROJECT in "${PROJECTS[@]}"; do
    STUDY=$(echo "$TCGA_PROJECT" | cut -d'-' -f2 | tr '[:upper:]' '[:lower:]')
    CHECKPOINT_DIR="/projects/conco/gundla/root/image2st/predictions/sequoia_${STUDY}"

    echo "Processing project: $TCGA_PROJECT"

    # List of slides for this project (replace with your slide IDs)
    SLIDES=($(find "$SLIDE_PATH" -name "*.svs" -type f -exec basename {} \;))
    
    for slide_filename in "${SLIDES[@]}"; do
        slide_id="${slide_filename%.*}"
        slide_base=$(echo "$slide_id" | cut -d'-' -f1-5)
        SAVE_DIR="/projects/conco/gundla/root/image2st/vis_all/"
        mkdir -p "$SAVE_DIR"

        echo "[$TCGA_PROJECT] Processing slide: $slide_filename"

        python spatial_vis/visualize.py \
            --study "$STUDY" \
            --project "$TCGA_PROJECT" \
            --gene_names "$GENES_IN_USE" \
            --wsi_file_name "$slide_filename" \
            --save_folder "$SAVE_DIR" \
            --model_type "$MODEL_TYPE" \
            --feat_type "$FEATURE_USE" \
            --folds "$FOLDS" \
            --slide_path "$SLIDE_PATH" \
            --mask_path "$SLIDE_PATH" \
            --checkpoint "$CHECKPOINT_DIR"
    done
done
