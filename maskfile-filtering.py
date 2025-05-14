import os
import glob
import pandas as pd

# Set paths
svs_dir = "/projects/conco/gundla/root/uniglacier/data/internal/"
mask_base_dir = (
    "/projects/conco/gundla/root/uniglacier/models/trained/"
    "image2st/Patches_hdf5/"
)
ref_file_path = (
    "/projects/conco/gundla/root/uniglacier/model_src/"
    "sequoia-pub/examples/ref_file_final.csv"
)

# Read the reference file
ref_df = pd.read_csv(ref_file_path)

# Normalize the wsi filenames in the reference file
ref_df["wsi_file_name"] = ref_df["wsi_file_name"].apply(
    lambda x: os.path.splitext(os.path.basename(str(x)))[0]
)

# Find all .svs files
svs_files = glob.glob(os.path.join(svs_dir, "*.svs"))

# Prepare results
results = []

for svs_path in svs_files:
    filename = os.path.basename(svs_path)
    slide_id = os.path.splitext(filename)[0]

    mask_path = os.path.join(mask_base_dir, slide_id, "mask.npy")
    mask_exists = os.path.isfile(mask_path)

    slide_path = os.path.join(svs_dir, f"{slide_id}.svs")

    results.append({
        "slide_id": slide_id,
        "status": "yes" if mask_exists else "no",
        "slide_path": slide_path
    })

# Create DataFrame and merge with reference data
mask_df = pd.DataFrame(results)
merged_df = mask_df.merge(
    ref_df[["wsi_file_name", "tcga_project"]],
    left_on="slide_id",
    right_on="wsi_file_name",
    how="left"
)

# Drop redundant column
merged_df.drop(columns=["wsi_file_name"], inplace=True)
# Keep only rows where mask exists
filtered_df = merged_df[merged_df["status"] == "yes"].copy()
# Save to CSV
filtered_df.to_csv("mask_status_with_project.csv", index=False)

print("CSV file 'mask_status_with_project.csv' created.")