import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np


def calculate_kappa_per_label(file1, file2):
    # Load the CSV files into pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure both files have the same length
    if len(df1) != len(df2):
        raise ValueError("The two files must have the same number of rows.")

    # Ensure both files have the 'text' and 'arg_role' columns
    if 'text' not in df1.columns or 'arg_role' not in df1.columns:
        raise ValueError(f"File {file1} is missing required columns.")
    if 'text' not in df2.columns or 'arg_role' not in df2.columns:
        raise ValueError(f"File {file2} is missing required columns.")

    # Extract the 'arg_role' columns
    labels1 = df1['arg_role']
    labels2 = df2['arg_role']

    # Get unique labels
    unique_labels = sorted(labels1.unique())

    # Calculate Cohen's Kappa for each label
    kappa_scores = []
    for label in unique_labels:
        binary_labels1 = (labels1 == label).astype(int)
        binary_labels2 = (labels2 == label).astype(int)
        kappa_score = cohen_kappa_score(binary_labels1, binary_labels2)
        kappa_scores.append(kappa_score)
        print(f"Kappa for label '{label}': {kappa_score}")

    # Calculate mean and median Kappa scores
    mean_kappa = np.mean(kappa_scores)
    median_kappa = np.median(kappa_scores)

    return mean_kappa, median_kappa


file1 = '../Dataset/IRC_labeled_judgments/UKHL_1.43_labeled.csv'
file2 = '../Dataset/meya_labelled/UKHL_1.43_labeled.csv'

mean_kappa, median_kappa = calculate_kappa_per_label(file1, file2)
print(f"Mean Cohen's Kappa Score: {mean_kappa}")
print(f"Median Cohen's Kappa Score: {median_kappa}")
