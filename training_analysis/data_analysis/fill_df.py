import os
import pandas as pd

directory = '/Users/ariellethijssen/areel/msccs/summerproject/dataset/IRC_labeled_judgments'

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)

        df = pd.read_csv(file_path)

        if 'issue_index' in df.columns:
            df['issue_index'].fillna(-1, inplace=True)

        if 'main_issue' in df.columns:
            df['main_issue'].fillna(0, inplace=True)
        if 'main_conclusion' in df.columns:
            df['main_conclusion'].fillna(0, inplace=True)

        df.to_csv(file_path, index=False)

