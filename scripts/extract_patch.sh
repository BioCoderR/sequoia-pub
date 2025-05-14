#!/bin/bash
#SBATCH --job-name=patch_job
#SBATCH --output=/projects/conco/gundla/root/logs/sequoia/patch_generation_%j.out
#SBATCH --error=/projects/conco/gundla/root/logs/sequoia/patch_generation_%j.err
#SBATCH --partition=IKIM        # Use IKIM partition
#SBATCH --ntasks=2             # 4 parallel tasks (one per node)
#SBATCH --cpus-per-task=32      # 16 CPUs per task
#SBATCH --mem=96G               # 32GB per task
#SBATCH --nodes=2               # Request 4 nodes


# Get the number of allocated CPUs dynamically
export NUM_CPUS=$SLURM_CPUS_PER_TASK

# Patch Extraction
python3 pre_processing/patch_gen_hdf5.py \
        --ref_file ./examples/ref_file_final.csv \
        --wsi_path /projects/conco/gundla/root/Glioma_CLAM/rawdata/slides \
        --patch_path /projects/conco/gundla/root/image2st/Patches_hdf5 \
        --mask_path /projects/conco/gundla/root/image2st/Patches_hdf5 \
        --patch_size 256 \
        --parallel $NUM_CPUS