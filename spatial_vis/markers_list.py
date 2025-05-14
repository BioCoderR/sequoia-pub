import pandas as pd
import numpy as np
import os

# === CONFIG ===
csv_path = '/projects/conco/gundla/root/uniglacier/model_src/sequoia-pub/evaluation/GenesMetaModuleMarkersGBM.xlsx'
output_dir = '/projects/conco/gundla/root/uniglacier/models/trained/image2st/generated_markers/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load marker list
df = pd.read_excel(csv_path)
print(df)
# Check if at least one known cell type column is present
expected_celltypes = ['MES2', 'MES1', 'AC', 'OPC', 'NPC1', 'NPC2', 'G1S', 'G2M']
assert any(celltype in df.columns for celltype in expected_celltypes), \
    "Excel must have columns named MES2, MES1, AC, OPC, NPC1, NPC2, G1S, G2M."

# Save .npy per cell type
for cell_type in expected_celltypes:
    if cell_type in df.columns:
        gene_list = df[cell_type].dropna().unique()
        save_path = os.path.join(output_dir, f"{cell_type}.npy")
        np.save(save_path, gene_list)
        print(f"Saved {cell_type}.npy with {len(gene_list)} genes.")

# Save 'all.npy' (all unique genes across all types)
all_genes = pd.unique(df[expected_celltypes].values.ravel('K'))
all_genes = [gene for gene in all_genes if pd.notna(gene)]
np.save(os.path.join(output_dir, 'all.npy'), all_genes)
print(f"Saved all.npy with {len(all_genes)} total genes.")