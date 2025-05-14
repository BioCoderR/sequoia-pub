#!/bin/bash
#SBATCH --job-name=unifts
#SBATCH --output=/projects/conco/gundla/root/logs/sequoia/extract_uni_%j.out
#SBATCH --error=/projects/conco/gundla/root/logs/sequoia/extract_uni_%j.err
#SBATCH --partition=GPUampere  # Change based on available partitions
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --cpus-per-task=32     # Request 16 CPU cores
#SBATCH --mem=64G              # Request 64GB memory
#SBATCH --nodes=1              # Run on a single node

echo "Running UNI Feature Extraction on GPU"
echo "Job started on $(hostname) at $(date)"

python3 pre_processing/compute_features_hdf5.py \
        --ref_file /homes/psgudla/PARA/AREAS/Models/sequoia-pub/examples/ref_file_final.csv \
        --patch_data_path /projects/conco/gundla/root/image2st/Patches_hdf5 \
        --feature_path /projects/conco/gundla/root/image2st/uni_features \
        --feat_type uni \
        --max_patch_number 5000

echo "Job completed at $(date)"