#!/bin/bash
#SBATCH --job-name=predInd
#SBATCH --output=/projects/conco/gundla/root/uniglacier/logs/sequoia/predict_gbm_out_%j.out
#SBATCH --error=/projects/conco/gundla/root/uniglacier/logs/sequoia/predict_gbm_err_%j.err
#SBATCH --partition=GPUampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Define arguments
REF_FILE="/projects/conco/gundla/root/uniglacier/model_src/sequoia-pub/examples/ref_file_final.csv"
FEATURE_PATH="/projects/conco/gundla/root/image2st/uni_features"  # This is correct
FEATURE_USE="uni_features"
FOLDS=5
SEED=99
BATCH_SIZE=16
DEPTH=6
NUM_HEADS=16
TCGA_PROJECT="TCGA-GBM"  # Changed to LGG features if in TCGA-LGG folder
SAVE_DIR="/projects/conco/gundla/root/image2st/predictions"
EXP_NAME="sequoia_gbm"  # Changed to LGG project if having the features LGG

# Run the script
python ./evaluation/predict_independent_dataset.py \
    --ref_file "$REF_FILE" \
    --feature_path "$FEATURE_PATH" \
    --feature_use "$FEATURE_USE" \
    --folds "$FOLDS" \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE" \
    --depth "$DEPTH" \
    --num-heads "$NUM_HEADS" \
    --tcga_project "$TCGA_PROJECT" \
    --save_dir "$SAVE_DIR" \
    --exp_name "$EXP_NAME"