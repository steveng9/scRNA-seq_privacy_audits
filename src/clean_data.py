import sys

env = "server" if len(sys.argv) > 1 and sys.argv[1] == "T" else "local"

import os
import anndata as ad
import re
import pandas as pd


if env == "local":
    data_dir = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data"
else:
    data_dir = "/home/golobs/data"

def format_ct_name(cell_name):
    return str(cell_name).replace(" ", "_")


def extract_age(age_str):
    if pd.isna(age_str):
        return age_str
    age_str = str(age_str)
    match = re.search(r'(\d+)[-\s]year[-\s]old', age_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    try:
        return int(float(age_str))
    except (ValueError, TypeError):
        pass
    # Otherwise leave as is (prenatal, post-fertilization, etc.)
    return age_str



def clean_dataset(dataset):
    print(f"\ncleaning {dataset}")

    file_path = os.path.join(data_dir, dataset, "full_dataset.h5ad")
    adata = ad.read_h5ad(file_path)

    if 'self_reported_ethnicity' in adata.obs.columns:
        adata.obs.rename(columns={'self_reported_ethnicity': 'ethnicity'}, inplace=True)
        print("\tupdated column self_reported_ethnicity")

    if 'development_stage' in adata.obs.columns:
        adata.obs.rename(columns={'development_stage': 'age'}, inplace=True)
        print("\tupdated column development_stage")

    if 'donor_id' in adata.obs.columns:
        adata.obs.rename(columns={'donor_id': 'individual'}, inplace=True)
        print("\tupdated column donor_id")

    if 'age' in adata.obs.columns:
        original_values = adata.obs['age'].unique()
        adata.obs['age'] = adata.obs['age'].apply(extract_age)
        cleaned_values = adata.obs['age'].unique()

        print(f"✓ Cleaned age values:")
        print(f"  Original sample: {list(original_values[:5])}")
        print(f"  Cleaned sample: {list(cleaned_values[:5])}")

    if 'cell_type' in adata.obs.columns:
        original_ct = adata.obs['cell_type'].unique()
        adata.obs['cell_type'] = adata.obs['cell_type'].apply(format_ct_name)
        cleaned_ct = adata.obs['cell_type'].unique()

        print(f"✓ Formatted cell type names:")
        print(f"  Original sample: {list(original_ct[:5])}")
        print(f"  Cleaned sample: {list(cleaned_ct[:5])}")

    # Save the cleaned data
    output_path = os.path.join(data_dir, dataset, "full_dataset_cleaned.h5ad")
    adata.write_h5ad(output_path)
    print(f"\n✓ Saved cleaned dataset to {output_path}")

    return adata


# Usage
if __name__ == "__main__":
    for dataset in ["ok", "cg", "aida"]:
        clean_dataset(dataset)
