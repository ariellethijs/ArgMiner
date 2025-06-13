import os
import pandas as pd


def count_arg_roles_in_csv(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)

            issue_count = df[df['arg_role'] == 'ISSUE'].shape[0]
            reason_count = df[df['arg_role'] == 'REASON'].shape[0]
            conclusion_count = df[df['arg_role'] == 'CONCLUSION'].shape[0]

            print(f"File: {filename}")
            print(f"  ISSUE: {issue_count}")
            print(f"  REASON: {reason_count}")
            print(f"  CONCLUSION: {conclusion_count}")
            print("-" + "-" * len(f"File: {filename}"))


directory_path = '/Users/ariellethijssen/areel/msccs/summerproject/ArgMiner/training_analysis/dataset/meya_labelled'
count_arg_roles_in_csv(directory_path)
