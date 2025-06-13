import re

import pandas as pd
import json


casenum = '1.02'
filepath = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments/1.02.json'

# Load the JSON file
with open(filepath, 'r') as file:
    data = json.load(file)

rows = []
for item in data:
    if 'latestLabel' in item and 'jsonResponse' in item['latestLabel']:
        annotations = item['latestLabel']['jsonResponse'].get('NAMED_ENTITIES_RECOGNITION_JOB', {}).get('annotations', [])
        for annotation in annotations:
            text = annotation.get('content')
            role = annotation.get('categories', [{}])[0].get('name')
            if text and role:
                rows.append({'text': text, 'role': role})

df = pd.DataFrame(rows)
#df.to_csv('/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments/1.02.csv', index=False)
# Display the DataFrame
print(df)

segmented_filepath = '/Users/ariellethijssen/areel/MscCS/SummerProject/nolansproject/2024/pipeline/data/UKHL_corpus/UKHL_' + casenum + '.csv'
segmented_df = pd.read_csv(segmented_filepath)


def find_role(text):
    escaped_text = re.escape(text)
    match = df[df['text'].str.contains(escaped_text, na=False)]
    if not match.empty:
        return match.iloc[0]['role']
    return None


segmented_df['role'] = segmented_df['text'].apply(find_role)

print(segmented_df)
segmented_df.to_csv('/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments/1.02.csv', index=False)
