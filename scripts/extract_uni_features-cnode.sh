#!/bin/bash

# Set resource limits (modify as needed)
NUM_CPUS=32    # Number of CPU cores to use
MEMORY=128G    # Memory allocation (adjust as necessary)
GPU_ID=0       # Change if using a different GPU

echo "Running UNI Feature Extraction on local machine"
echo "Job started on $(hostname) at $(date)"


# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=$NUM_CPUS  # Control CPU usage

# Run the Python script
CUDA_VISIBLE_DEVICES=0 python3 pre_processing/compute_features_hdf5.py \
        --ref_file /homes/psgudla/PARA/AREAS/Models/sequoia-pub/examples/ref_file.csv \
        --patch_data_path /projects/conco/gundla/root/image2st/Patches_hdf5 \
        --feature_path /projects/conco/gundla/root/image2st/uni_features \
        --feat_type uni \
        --max_patch_number 999999 > /projects/conco/gundla/root/logs/sequoia/unift.log 2>&1

echo "Job completed at $(date)"