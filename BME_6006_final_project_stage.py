#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
mrna_seq_transposed = data_mrna_seq_zscores.set_index( #tells pandas to use the first column (the gene IDs) as the row index rather than treating it as a variable
    data_mrna_seq_zscores.columns[0] # this gives the name of the first column which is the gene identifiers 
).T #.T transposes the data

# Turn the sample ID (currently the index) into a column
mrna_seq_transposed = mrna_seq_transposed.reset_index()

# Rename the sample ID column
mrna_seq_transposed = mrna_seq_transposed.rename(
    columns={'index': 'Patient ID'}
)

## Get sample ID and detailed cancer type from the clinical dataframe ##
cancer_type = data_clinical_patient[
    ['#Patient identifier', 'Stage']
]

## Match sample IDs and add histology abbreviation for each sample ##
#the samples aren't in the same order across files so need to make sure the correct histological abbreviation is matched with the correct sample and RNA data
mrna_seq_transposed = mrna_seq_transposed.merge(
    cancer_type,
    left_on='Patient ID',  # Column in the mRNA dataframe containing the sample ID
    right_on='#Patient identifier', # Column in the clinical dataframe containing the sample ID
    how='left'     
    # Keep all samples from the mRNA dataframe and add the matching clinical information when possible
)

# Remove the #Sample Identifier column because now redundant - same info already in Sample ID
mrna_seq_final = mrna_seq_transposed.drop(
    columns=['#Patient identifier']
)

# Remove patients with no Stage information
mrna_seq_final = mrna_seq_final.dropna(subset=['Stage'])

## Move detailed cancer type to the far left ##

cols = mrna_seq_final.columns.tolist() # Get a list containing all of the dataframe's column names
cols.insert(0, cols.pop(cols.index('Stage'))) 
# Find the position of "Cancer Type Detailed", remove it from its current position, insert it at position 0 (first column)

## Reorder the dataframe using the new column order where its Cancer Type Detailed then Sample ID then all of the genes
mrna_seq_final = mrna_seq_final[cols]

'''
## filter out the specific cancer types I want to analyze ##
#I did this because having all types was way too much data to reasonably analyze for this
#I thought it'd be interesting to see mRNA changes for the same camcer type across different organs
pancan_mrna_filtered = pancan_mrna_final[
    pancan_mrna_final['Cancer Type Detailed'].isin([
        'Pancreatic Adenocarcinoma',
        'Prostate Adenocarcinoma',
        'Colorectal Adenocarcinoma'
    ])
]
'''


#%% Create clustered heatmap

# Create row labels containing Sample ID and Cancer Type so each sample can be identified on the heatmap
row_labels = (
    mrna_seq_final['Patient ID'] 
    + ' , ' # separate sample ID and cancer type with comma
    + mrna_seq_final['Stage']
)

# Remove Sample ID and Cancer Type because they are identifiers not gene-expression variables that should be used for clustering
data_matrix = mrna_seq_final.drop(
    columns=['Stage', 'Patient ID']
)

# Remove any gene that contains a NaN value
data_matrix = data_matrix.dropna(axis=1)

# Create a clustered heatmap using the gene-expression data
cluster1 = sns.clustermap(
    data_matrix,
    cmap='bwr', # Use a blue-white-red color scale for the gene-expression values
    center=0,  # Center the color scale at 0 because the data are z-scores
    figsize=(20, 20), # Set the size of the heatmap
    metric='euclidean', # Use Euclidean distance to measure differences between samples/genes
    method='complete',  # Use complete linkage to determine how clusters are formed
    row_cluster=True,  # Cluster the samples (rows)
    col_cluster=True, # Cluster the genes (columns)
    yticklabels=row_labels # Label each row with its Sample ID and Cancer Type
)

# display the heatmap
plt.show()

#%% perform PCA

pca = PCA() #create a pca model
# Perform PCA on the gene-expression data and transform the data
coefficients = pca.fit_transform(data_matrix) # "coefficients" contains the PCA scores for each sample

# Calculate the percentage of total variance explained by each PC
explained = pca.explained_variance_ratio_ * 100

# Calculate the cumulative percentage of variance explained
cumulative_explained = np.cumsum(explained) # This adds the variance explained by each PC to the previous PCs

# Print variance explained by first 3 PCs
print("Variance explained by first 3 PCs:")
print(f"PC1: {explained[0]:.2f}%")
print(f"PC2: {explained[1]:.2f}%")
print(f"PC3: {explained[2]:.2f}%")

print(f"\nTotal variance explained by first 3 PCs: "
      f"{cumulative_explained[2]:.2f}%")

# Find the number of PCs needed to explain at least 85% of the variance
# np.argmax finds the first PC where cumulative variance reaches 85%
# +1 is needed because Python starts counting at 0
n_pcs_85 = np.argmax(cumulative_explained >= 85) + 1

# Print the number of PCs needed to reach 85% variance
print(f"\nNumber of PCs needed to capture at least 85% "
      f"of the variance: {n_pcs_85}")

## Plot cumulative variance explained ##

plt.figure(figsize=(8, 6))


plt.plot(
    np.arange(1, len(explained) + 1), # x-axis = number of PCs
    cumulative_explained, # y-axis = cumulative percentage of variance explained
    marker='o',
    linewidth=2
)

# Add a horizontal dashed line at 85%
# This makes it easy to see where the cumulative variance reaches 85%
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
# Create a dataframe containing the PCA scores for each sample
# PC1, PC2, and PC3 are the transformed coordinates of each sample
pca_df = pd.DataFrame({
    'PC1': coefficients[:, 0],  # Get the PC1 score for every sample
    'PC2': coefficients[:, 1],  # Get the PC2 score for every sample
    'PC3': coefficients[:, 2],  # Get the PC3 score for every sample
    'Stage': mrna_seq_final['Stage'].values,  # Add the cancer type for each sample
    # .values converts the pandas column into an array
    'Patient ID': mrna_seq_final['Patient ID'].values # Add the Sample ID for each sample
})

## PC1 vs PC2 ##

# Create a new figure for the PC1 vs PC2 plot
plt.figure(figsize=(8, 6))

# Create a scatter plot using the PCA scores
sns.scatterplot(
    data=pca_df, #using dataframe containing PCA results
    x='PC1', # PC1 scores on the x-axis
    y='PC2', # PC2 scores on the y-axis
    hue='Stage',  # Use different colors for the different cancer types
    s=70  # Set the size of each data point
)

plt.xlabel(f'PC1 ({explained[0]:.2f}% variance)') # Label the x-axis and show how much variance PC1 explains
plt.ylabel(f'PC2 ({explained[1]:.2f}% variance)') # Label the y-axis and show how much variance PC2 explains
plt.title('PCA: PC1 vs PC2')

plt.legend(title='Stage') #add legend for cancer types
plt.grid(True)
plt.tight_layout()
plt.show()

## PC1 vs PC3 ##

# Create a new figure for the PC1 vs PC3 plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x='PC1',  # PC1 scores on the x-axis
    y='PC3',  # PC1 scores on the x-axis
    hue='Stage', # Use different colors for the different cancer types
    s=70
)

plt.xlabel(f'PC1 ({explained[0]:.2f}% variance)') # Label the x-axis and show how much variance PC1 explains
plt.ylabel(f'PC3 ({explained[2]:.2f}% variance)') # Label the y-axis and show how much variance PC3 explains
plt.title('PCA: PC1 vs PC3')

plt.legend(title='Stage')
plt.grid(True)
plt.tight_layout()
plt.show()

## PC2 vs PC3 ##

# Create a new figure for the PC2 vs PC3 plot
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x='PC2', # PC2 scores on the x-axis
    y='PC3', # PC1 scores on the x-axis
    hue='Stage', # Use different colors for the different cancer types
    s=70
)

plt.xlabel(f'PC2 ({explained[1]:.2f}% variance)') # Label the x-axis and show how much variance PC2 explains
plt.ylabel(f'PC3 ({explained[2]:.2f}% variance)') # Label the y-axis and show how much variance PC3 explains
plt.title('PCA: PC2 vs PC3')

plt.legend(title='Stage')
plt.grid(True)
plt.tight_layout()
plt.show()

### loading scores ###
# Get the PC1 loading score for every gene
pc1_loadings = pd.Series(
    pca.components_[0], # pca.components_[0] contains the loading of each gene on PC1
    index=data_matrix.columns # Use the gene names as the index so each loading is associated with its corresponding gene
)

# Sort loading scores from highest to lowest loading
pc1_loadings_sorted = pc1_loadings.sort_values(ascending=False)

# Print the PC1 loading scores
print("PC1 loading scores:")
print(pc1_loadings_sorted)