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

#data source: https://www.cbioportal.org/study/summary?id=luad_oncosg_2020

#%% Organize the matrix

# Read the mRNA gene-expression Excel file
# nrows=101 means only the first 101 rows are read (the first row contains the column names so this gives 100 genes)
# I truncated at 100 genes because it was taking too long for python to go through all the genes
data_mrna_seq_zscores = pd.read_excel(
    'data_mrna_seq_v2_rsem_zscores_ref_all_samples_6006_project.xlsx',
    nrows=102
)


## Transpose mRNA dataframe ##
#need to do this because want sample ID as rows not genes for analysis

# First column in original datafram contains the gene/sample identifier
mrna_seq_transposed = data_mrna_seq_zscores.set_index( #tells pandas to use the first column (the gene IDs) as the row index rather than treating it as a variable
    data_mrna_seq_zscores.columns[0] # this gives the name of the first column which is the gene identifiers 
).T #.T transposes the data

# Remove the Entrez_Gene_Id row
mrna_seq_transposed = mrna_seq_transposed.drop(
    index='Entrez_Gene_Id',
    errors='ignore'
)

# Turn the sample ID (currently the index) into a column
mrna_seq_transposed = mrna_seq_transposed.reset_index()

# Rename the sample ID column
mrna_seq_transposed = mrna_seq_transposed.rename(
    columns={'index': 'Patient ID'}
)

mrna_seq_final = mrna_seq_transposed


#%% Create clustered heatmap

# Create row labels containing Sample ID and Cancer Type so each sample can be identified on the heatmap
row_labels = (
    mrna_seq_final['Patient ID'] 
)

# Remove Sample ID and Cancer Type because they are identifiers not gene-expression variables that should be used for clustering
data_matrix = mrna_seq_final.drop(
    columns=['Patient ID']
)

# Remove any gene that contains a NaN value
data_matrix = data_matrix.dropna(axis=1)

# Create a clustered heatmap using the gene-expression data
cluster1 = sns.clustermap(
    data_matrix,
    cmap='bwr', # Use a blue-white-red color scale for the gene-expression values
    center=0,  # Center the color scale at 0 because the data are z-scores
    figsize=(40, 30), # Set the size of the heatmap
    metric='euclidean', # Use Euclidean distance to measure differences between samples/genes
    method='complete',  # Use complete linkage to determine how clusters are formed
    row_cluster=True,  # Cluster the samples (rows)
    col_cluster=True, # Cluster the genes (columns)
    yticklabels=row_labels # Label each row with its Sample ID and Cancer Type
)

# display the heatmap
plt.show()