"""
check_h5_features.py - Script to verify H5 files for uni and cluster features
and create filtered reference file for missing cluster features
"""

import os
import h5py
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

# Define paths
ROOT_DIR = "/projects/conco/gundla/root/image2st"
H5_DIR = os.path.join(ROOT_DIR, "uni_features")
REF_FILE = "/projects/conco/gundla/root/uniglacier/model_src/sequoia-pub/examples/ref_file_final.csv"
OUTPUT_DIR = os.path.join(ROOT_DIR, "feature_check")

def verify_h5_features(file_path: str) -> Tuple[bool, bool, str]:
    """
    Verify if H5 file contains required feature groups
    Returns: (has_uni_features, has_cluster_features, error_message)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            has_uni = '/uni_features' in f
            has_cluster = '/cluster_features' in f
            return has_uni, has_cluster, ""
    except Exception as e:
        return False, False, str(e)

def check_h5_files() -> List[Dict]:
    """Check all H5 files and return results"""
    results = []
    
    # Read reference file
    ref_df = pd.read_csv(REF_FILE)
    
    for _, row in ref_df.iterrows():
        wsi_file_name = row['wsi_file_name']
        project = row['tcga_project']
        
        h5_path = os.path.join(H5_DIR, project, wsi_file_name, f"{wsi_file_name}.h5")
        
        result = {
            'wsi_file_name': wsi_file_name,
            'project': project,
            'h5_exists': False,
            'has_uni_features': False,
            'has_cluster_features': False,
            'error': ''
        }
        
        if os.path.exists(h5_path):
            result['h5_exists'] = True
            has_uni, has_cluster, error = verify_h5_features(h5_path)
            result['has_uni_features'] = has_uni
            result['has_cluster_features'] = has_cluster
            result['error'] = error
            
        results.append(result)
    
    return results

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check H5 files
    results = check_h5_files()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create filtered reference file for missing cluster features
    ref_df = pd.read_csv(REF_FILE)
    missing_cluster_df = ref_df[ref_df['wsi_file_name'].isin(
        results_df[~results_df['has_cluster_features']]['wsi_file_name']
    )]
    
    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'h5_feature_check.csv'), index=False)
    missing_cluster_df.to_csv(os.path.join(OUTPUT_DIR, 'missing_cluster_features.csv'), index=False)
    
    # Count statistics by project
    project_stats = results_df.groupby('project').agg({
        'wsi_file_name': 'count',
        'h5_exists': 'sum',
        'has_uni_features': 'sum',
        'has_cluster_features': 'sum'
    }).rename(columns={'wsi_file_name': 'total_slides'})
    
    # Print summary
    print("\n H5 Feature Check completed:")
    print("\n Statistics by Project:")
    print(project_stats.to_string())
    
    print("\n Files generated:")
    print(f"    - Full check results: {os.path.join(OUTPUT_DIR, 'h5_feature_check.csv')}")
    print(f"    - Missing cluster features: {os.path.join(OUTPUT_DIR, 'missing_cluster_features.csv')}")
    
    # Print issues
    print("\n  Issues found:")
    issues = results_df[
        (~results_df['h5_exists']) | 
        (~results_df['has_uni_features']) | 
        (~results_df['has_cluster_features']) |
        (results_df['error'] != '')
    ]
    
    for _, issue in issues.iterrows():
        wsi_file_name = issue['wsi_file_name']
        if not issue['h5_exists']:
            print(f"    - {wsi_file_name}: H5 file not found")
        elif not issue['has_uni_features']:
            print(f"    - {wsi_file_name}: Missing uni features")
        elif not issue['has_cluster_features']:
            print(f"    - {wsi_file_name}: Missing cluster features")
        if issue['error']:
            print(f"    - {wsi_file_name}: Error: {issue['error']}")

if __name__ == "__main__":
    main()