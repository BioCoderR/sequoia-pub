#!/bin/bash
#SBATCH --job-name=gbm_celltypes
#SBATCH --output=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/gbm_celltypes_tweaked_%j.out
#SBATCH --error=/projects/conco/gundla/root/uniglacier/logs/sequoia/vis/gbm_celltypes_tweaked_%j.err
#SBATCH --time=06:00:00            # 4 hours max
#SBATCH --partition=GPUampere            # or normal or your CPU-only partition
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # 4 cores (good for Pandas / Matplotlib IO)
#SBATCH --mem=32G                   # Safe memory (adjust if you see OOM errors)


# === Run the script ===
python spatial_vis/gbm_celltype_analysis.py
# python spatial_vis/get_emd.py \
#     --slide_nr 1 \
#     --pred_folder ./visualizations/spatial_GBM_pred/vis_all/ \
#     --save_folder ./visualizations/spatial_GBM_pred/vis_all/gbm_celltypes/spatial_maps/ \
#     --gene_names ./spatial_vis/generated_markers/all.npy
