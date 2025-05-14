#!/bin/bash
#SBATCH --job-name=kmeanft_lgg
#SBATCH --output=/projects/conco/gundla/root/uniglacier/logs/sequoia/extract_kmeans_lgg_%A_%a.out
#SBATCH --error=/projects/conco/gundla/root/uniglacier/logs/sequoia/extract_kmeans_lgg_%A_%a.err
#SBATCH --partition=GPUampere
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --array=0-4

# Create output directories if they don't exist
FEATURE_PATH="/projects/conco/gundla/root/image2st/uni_features"
mkdir -p "${FEATURE_PATH}"

# Get the number of allocated CPUs dynamically
export NUM_CPUS=$SLURM_CPUS_PER_TASK

# Process TCGA-LGG
echo "Processing TCGA-LGG dataset"
TOTAL_SLIDES=$(grep "TCGA-LGG" ./examples/ref_file_missing.csv | wc -l)
SPLITS=5
SLIDES_PER_JOB=$((TOTAL_SLIDES / SPLITS))

START_INDEX=$((SLURM_ARRAY_TASK_ID * SLIDES_PER_JOB))
END_INDEX=$((START_INDEX + SLIDES_PER_JOB))

# Ensure END_INDEX does not exceed TOTAL_SLIDES
if [ "$END_INDEX" -gt "$TOTAL_SLIDES" ]; then
    END_INDEX=$TOTAL_SLIDES
fi

echo "Processing TCGA-LGG slides from index $START_INDEX to $END_INDEX"
echo "Feature output directory: $FEATURE_PATH/TCGA-LGG"

python3 ./pre_processing/kmean_features.py \
        --ref_file ./examples/ref_file_filtered.csv \
        --patch_data_path /projects/conco/gundla/root/image2st/Patches_hdf5 \
        --feature_path "$FEATURE_PATH" \
        --num_clusters 50 \
        --tcga_projects TCGA-LGG \
        --start ${START_INDEX} \
        --end ${END_INDEX} \
        --seed 99

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Python script failed for LGG"
    exit 1
fi