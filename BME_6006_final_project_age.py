"""
Project: BME 6006 Final Project 
Author: Allie Sack
Due Date: 10/14/26
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

#data source: https://www.cbioportal.org/study/summary?id=luad_oncosg_2020

#%% Organize the matrix
# Read the clinical data file
data_clinical_patient = pd.read_excel('data_clinical_patient_6006_project.xlsx')

# Remove patients whose ID starts with "B"
data_clinical_patient = data_clinical_patient[
    ~data_clinical_patient['#Patient identifier'].astype(str).str.startswith('B')
]

# Read the mRNA gene-expression Excel file
# nrows=101 means only the first 101 rows are read (the first row contains the column names so this gives 100 genes)
# I truncated at 100 genes because it was taking too long for python to go through all the genes
data_mrna_seq_zscores = pd.read_excel(
    'data_mrna_seq_v2_rsem_zscores_ref_all_samples_6006_project.xlsx',
    nrows=501
)


## Transpose mRNA dataframe ##
#need to do this because want sample ID as rows not genes for analysis

# First column in original datafram contains the gene/sample identifier
mrna_seq_transposed = data_mrna_seq_zscores.set_index(
    data_mrna_seq_zscores.columns[0]
).T

# Turn the sample ID (currently the index) into a column
mrna_seq_transposed = mrna_seq_transposed.reset_index()

# Rename the sample ID column
mrna_seq_transposed = mrna_seq_transposed.rename(
    columns={'index': 'Patient ID'}
)

## Get sample ID and age from the clinical dataframe ##
age = data_clinical_patient[
    ['#Patient identifier', 'Age']
]

## Match sample IDs and add age for each sample ##
#the samples aren't in the same order across files so need to make sure the correct age is matched with the correct sample and RNA data
mrna_seq_transposed = mrna_seq_transposed.merge(
    age,
    left_on='Patient ID',
    right_on='#Patient identifier',
    how='left'
)

# Remove the #Patient Identifier column because now redundant - same info already in Patient ID
mrna_seq_final = mrna_seq_transposed.drop(
    columns=['#Patient identifier']
)

# Remove patients with no Age information
mrna_seq_final = mrna_seq_final.dropna(subset=['Age'])

# Create age groups
mrna_seq_final['Age Group'] = pd.cut(
    mrna_seq_final['Age'],
    bins=[39, 49, 59, 69, 79],
    labels=['40s', '50s', '60s', '70s']
)

# Remove patients outside of the 40s-70s age groups
mrna_seq_final = mrna_seq_final.dropna(subset=['Age Group'])

# Sort patients by age group and then by age
mrna_seq_final = mrna_seq_final.sort_values(
    by=['Age Group', 'Age']
)

## Move age group to the far left ##

cols = mrna_seq_final.columns.tolist()
cols.insert(0, cols.pop(cols.index('Age Group')))

## Reorder the dataframe using the new column order
mrna_seq_final = mrna_seq_final[cols]


#%% Create clustered heatmap

# Create row labels containing Sample ID and Age Group so each sample can be identified on the heatmap
row_labels = (
    mrna_seq_final['Patient ID'].astype(str)
    + ' , '
    + mrna_seq_final['Age Group'].astype(str)
)

# Remove Age, Age Group, and Patient ID because they are identifiers/clinical variables
# that should not be used for clustering
data_matrix = mrna_seq_final.drop(
    columns=['Age', 'Age Group', 'Patient ID']
)

# Remove any gene that contains a NaN value
data_matrix = data_matrix.dropna(axis=1)

# Create a clustered heatmap using the gene-expression data
cluster1 = sns.clustermap(
    data_matrix,
    cmap='bwr',
    center=0,
    figsize=(20, 20),
    metric='euclidean',
    method='complete',
    row_cluster=False,
    col_cluster=True,
    yticklabels=row_labels
)

# display the heatmap
plt.show()

#%% perform PCA

pca = PCA()

# Perform PCA on the gene-expression data and transform the data
coefficients = pca.fit_transform(data_matrix)

# Calculate the percentage of total variance explained by each PC
explained = pca.explained_variance_ratio_ * 100

# Calculate the cumulative percentage of variance explained
cumulative_explained = np.cumsum(explained)

# Print variance explained by first 3 PCs
print("Variance explained by first 3 PCs:")
print(f"PC1: {explained[0]:.2f}%")
print(f"PC2: {explained[1]:.2f}%")
print(f"PC3: {explained[2]:.2f}%")

print(f"\nTotal variance explained by first 3 PCs: "
      f"{cumulative_explained[2]:.2f}%")

# Find the number of PCs needed to explain at least 85% of the variance
n_pcs_85 = np.argmax(cumulative_explained >= 85) + 1

# Print the number of PCs needed to reach 85% variance
print(f"\nNumber of PCs needed to capture at least 85% "
      f"of the variance: {n_pcs_85}")

## Plot cumulative variance explained ##

plt.figure(figsize=(8, 6))

plt.plot(
    np.arange(1, len(explained) + 1),
    cumulative_explained,
    marker='o',
    linewidth=2
)

# Add a horizontal dashed line at 85%
plt.axhline(
    85,
    linestyle='--',
    linewidth=1.5,
    label='85% variance'
)

# Label axes
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.title('Cumulative Variance Explained by PCA')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Create dataframe containing PCA results ##

pca_df = pd.DataFrame({
    'PC1': coefficients[:, 0],
    'PC2': coefficients[:, 1],
    'PC3': coefficients[:, 2],
    'Age Group': mrna_seq_final['Age Group'].values,
    'Patient ID': mrna_seq_final['Patient ID'].values
})

## PC1 vs PC2 ##

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='Age Group',
    s=70
)

plt.xlabel(f'PC1 ({explained[0]:.2f}% variance)')
plt.ylabel(f'PC2 ({explained[1]:.2f}% variance)')
plt.title('PCA: PC1 vs PC2')

plt.legend(title='Age Group')
plt.grid(True)
plt.tight_layout()
plt.show()

## PC1 vs PC3 ##

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC3',
    hue='Age Group',
    s=70
)

plt.xlabel(f'PC1 ({explained[0]:.2f}% variance)')
plt.ylabel(f'PC3 ({explained[2]:.2f}% variance)')
plt.title('PCA: PC1 vs PC3')

plt.legend(title='Age Group')
plt.grid(True)
plt.tight_layout()
plt.show()

## PC2 vs PC3 ##

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x='PC2',
    y='PC3',
    hue='Age Group',
    s=70
)

plt.xlabel(f'PC2 ({explained[1]:.2f}% variance)')
plt.ylabel(f'PC3 ({explained[2]:.2f}% variance)')
plt.title('PCA: PC2 vs PC3')

plt.legend(title='Age Group')
plt.grid(True)
plt.tight_layout()
plt.show()

### loading scores ###

# Get the PC1 loading score for every gene
pc1_loadings = pd.Series(
    pca.components_[0],
    index=data_matrix.columns
)

# Sort loading scores from highest to lowest loading
pc1_loadings_sorted = pc1_loadings.sort_values(ascending=False)

# Print the PC1 loading scores
print("PC1 loading scores:")
print(pc1_loadings_sorted)