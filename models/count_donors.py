import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt



train_file = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/onek1k/onek1k_annotated_train_release.h5ad"
train = ad.read_h5ad(train_file)

test_file = "/Users/stevengolob/Documents/school/PhD/Ghent_project_mia_scRNAseq/data/onek1k/onek1k_annotated_test_release.h5ad"
test = ad.read_h5ad(test_file)


all_obs = pd.concat([train.obs, test.obs])


rows_per_individual = all_obs['individual'].value_counts()
rows_per_individual_and_celltype = all_obs[['individual', 'cell_label']].value_counts()


distribution = rows_per_individual.value_counts().sort_index()

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(distribution.index, distribution.values, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Rows per Individual', fontsize=12)
plt.ylabel('Number of Individuals', fontsize=12)
plt.title('Distribution of Row Counts Across Individuals', fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Optional: add value labels on top of bars
for i, (x, y) in enumerate(zip(distribution.index, distribution.values)):
    plt.text(x, y, str(y), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"Total individuals: {len(rows_per_individual)}")
print(f"Total rows: {len(all_obs)}")
print(f"\nDistribution:")
print(distribution)
print(f"\nSummary:")
print(f"  Min rows per individual: {rows_per_individual.min()}")
print(f"  Max rows per individual: {rows_per_individual.max()}")
print(f"  Mean rows per individual: {rows_per_individual.mean():.2f}")
print(f"  Median rows per individual: {rows_per_individual.median():.0f}")