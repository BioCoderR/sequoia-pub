#!/bin/bash
#SBATCH --job-name=ViSgbm
#SBATCH --output=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/visualize_gbm_GOI_out_%j.out
#SBATCH --error=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/visualize_gbm_GOI_err_%j.err
#SBATCH --partition=GPUampere
#SBATCH --gres=gpu:2
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=32
#SBATCH --time=25-00:00:00

# PATHs
SLIDE_PATH="/projects/conco/gundla/root/uniglacier/data/internal"
GENES_IN_USE="all"
FEATURE_USE="uni"
MODEL_TYPE="vis"
CSV_FILE="mask_status_with_project.csv"
# FOLDS="0,1,2,3,4"
# Read only GBM rows from CSV
mapfile -t SLIDE_IDS < <(awk -F, '$4 == "TCGA-GBM" {print $1}' $CSV_FILE | tail -n +2)
mapfile -t PROJECTS < <(awk -F, '$4 == "TCGA-GBM" {print $4}' $CSV_FILE | tail -n +2)

# Extract values
slide_id=${SLIDE_IDS}
slide_filename="${SLIDE_PATH}/${slide_id}.svs"
SAVE_DIR_BASE="/projects/conco/gundla/root/uniglacier/models/trained/image2st/vis_all/"
STUDY="GBM"
TCGA_PROJECT="TCGA-GBM"
CHECKPOINT_DIR="/projects/conco/gundla/root/uniglacier/models/trained/image2st/predictions/sequoia_gbm"
# SLIDES=($(find "$SLIDE_PATH" -name "*.svs" -type f -exec basename {} \;))
   
for fold in {0..4}; do
    echo "Processing slide: $slide_filename | Fold: $fold"
    python spatial_vis/visualize.py \
        --study "$STUDY" \
        --project "$TCGA_PROJECT" \
        --gene_names "$GENES_IN_USE" \
        --wsi_file_name "$slide_filename" \
        --save_folder "$SAVE_DIR" \
        --model_type "$MODEL_TYPE" \
        --feat_type "$FEATURE_USE" \
        --folds $fold \
        --checkpoint "$CHECKPOINT_DIR"
done
