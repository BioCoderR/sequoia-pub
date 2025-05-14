#!/bin/bash
#SBATCH --job-name=unifts_tcg
#SBATCH --output=/projects/conco/gundla/root/logs/sequoia/extract_tcga_uni_%j.out
#SBATCH --error=/projects/conco/gundla/root/logs/sequoia/extract_tcga_uni_%j.err
#SBATCH --partition=GPUampere  # Change based on available partitions
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --array=0-4               # Run on a single node
# Get the number of allocated CPUs dynamically
export NUM_CPUS=$SLURM_CPUS_PER_TASK

# Compute start and end indices dynamically for parallel execution
TOTAL_SLIDES=$(wc -l < ./examples/ref_file_final.csv)
SPLITS=5  # Adjust based on the total number of jobs needed
SLIDES_PER_JOB=$((TOTAL_SLIDES / SPLITS))

START_INDEX=$((SLURM_ARRAY_TASK_ID * SLIDES_PER_JOB))
END_INDEX=$((START_INDEX + SLIDES_PER_JOB))

# Ensure END_INDEX does not exceed TOTAL_SLIDES
if [ "$END_INDEX" -gt "$TOTAL_SLIDES" ]; then
    END_INDEX=$TOTAL_SLIDES
fi
echo "Running UNI Feature Extraction on GPU"
echo "Job started on $(hostname) at $(date)"

python3 pre_processing/compute_features_hdf5.py \
        --ref_file /homes/psgudla/PARA/AREAS/Models/sequoia-pub/examples/ref_file_final.csv \
        --patch_data_path /projects/conco/gundla/root/image2st/Patches_hdf5 \
        --feature_path /projects/conco/gundla/root/image2st/uni_features \
        --feat_type uni \
        --tcga_projects TCGA-GBM TCGA-LGG \
        --max_patch_number 6000 \
        --start ${START_INDEX} \
        --end ${END_INDEX}

echo "Job completed at $(date)"


# # Set resource limits (modify as needed)
# NUM_CPUS=32    # Number of CPU cores to use
# MEMORY=128G    # Memory allocation (adjust as necessary)
# GPU_ID=0       # Change if using a different GPU

# echo "Running UNI Feature Extraction on local machine"
# echo "Job started on $(hostname) at $(date)"


# # Set CUDA device
# export CUDA_VISIBLE_DEVICES=$GPU_ID
# export OMP_NUM_THREADS=$NUM_CPUS  # Control CPU usage

# # Run the Python script
# CUDA_VISIBLE_DEVICES=0 python3 pre_processing/compute_features_hdf5.py \
#         --ref_file /homes/psgudla/PARA/AREAS/Models/sequoia-pub/examples/ref_file.csv \
#         --patch_data_path /projects/conco/gundla/root/image2st/Patches_hdf5 \
#         --feature_path /projects/conco/gundla/root/image2st/uni_features \
#         --feat_type uni \
#         --max_patch_number 999999 > /projects/conco/gundla/root/logs/sequoia/unift.log 2>&1

# echo "Job completed at $(date)"
