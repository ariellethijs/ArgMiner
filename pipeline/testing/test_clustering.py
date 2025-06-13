import os
import pandas as pd
from collections import defaultdict
from itertools import combinations
import string

def normalize_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.strip().lower().split())

def get_pairs(texts):
    return set(combinations(sorted(texts), 2))

def calculate_accuracy(pipeline_df, dataset_df):
    issue_dataset = dataset_df[dataset_df['arg_role'] == 'ISSUE']

    issue_mapping = defaultdict(set)
    for _, row in issue_dataset.iterrows():
        normalized_text = normalize_text(row['text'])
        issue_mapping[row['issue_index']].add(normalized_text)

    cluster_mapping = defaultdict(set)
    for _, row in pipeline_df.iterrows():
        normalized_text = normalize_text(row['text'])
        cluster_mapping[row['cluster']].add(normalized_text)

    dataset_pairs = set()
    for issue_texts in issue_mapping.values():
        dataset_pairs.update(get_pairs(issue_texts))

    pipeline_pairs = set()
    for cluster_texts in cluster_mapping.values():
        pipeline_pairs.update(get_pairs(cluster_texts))

    correct_pairs = dataset_pairs & pipeline_pairs
    total_pairs = len(dataset_pairs)

    isolated_issues = set(text for texts in issue_mapping.values() if len(texts) == 1 for text in texts)
    isolated_in_pipeline = set(text for texts in cluster_mapping.values() if len(texts) == 1 for text in texts)
    isolated_correct = isolated_issues & isolated_in_pipeline

    accuracy = (len(correct_pairs) + len(isolated_correct)) / (total_pairs + len(isolated_issues)) if total_pairs + len(isolated_issues) > 0 else 0
    print(f'{accuracy}\n\n')

    return accuracy, total_pairs


def process_directories(pipeline_dir, dataset_dir):
    results = []
    overall_correct_pairs = 0
    overall_total_pairs = 0

    dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

    for dataset_file in dataset_files:
        filename = os.path.splitext(dataset_file)[0]
        parts = filename.split('_')
        prefix = parts[0]
        number_part = parts[1].replace('.', '')
        casename = f"{prefix}200{number_part}"

        pipeline_file = f"{casename}_issue_clusters.csv"

        dataset_path = os.path.join(dataset_dir, dataset_file)
        pipeline_path = os.path.join(pipeline_dir, pipeline_file)

        if os.path.exists(pipeline_path):
            print(casename)
            pipeline_df = pd.read_csv(pipeline_path)
            dataset_df = pd.read_csv(dataset_path)

            accuracy, total_pairs = calculate_accuracy(pipeline_df, dataset_df)
            results.append((casename, accuracy))

            overall_correct_pairs += accuracy * total_pairs
            overall_total_pairs += total_pairs

    overall_accuracy = overall_correct_pairs / overall_total_pairs if overall_total_pairs > 0 else 0

    return results, overall_accuracy

pipeline_dir = '/Users/ariellethijssen/areel/MscCS/SummerProject/nolansproject/2024/pipeline/arg_mining/data/issue_clustering'
dataset_dir = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/IRC_labeled_judgments'

accuracies, overall_accuracy = process_directories(pipeline_dir, dataset_dir)
print(f"Overall Accuracy: {overall_accuracy:.2%}")

