import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def extract_casename_from_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.split('_')

    year_part = parts[1].split('.')[0]
    case_number_part = parts[1].split('.')[1]

    full_year = f"200{year_part}"
    casename = f"{parts[0]}{full_year}{case_number_part}"

    return casename


def run_pipeline(casename):



def evaluate_pipeline(annotated_df, pipeline_df):
    merged_df = pd.merge(annotated_df, pipeline_df, on='text', suffixes=('_manual', '_pipeline'))

    metrics = {}
    for column in ['main_issue', 'main_conclusion', 'issue_index']:
        y_true = merged_df[f'{column}_manual']
        y_pred = merged_df[f'{column}_pipeline']

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        confusion = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)

        metrics[column] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'confusion_matrix': confusion,
            'classification_report': class_report
        }

    return metrics


def print_detailed_report(metrics):
    for column, metric in metrics.items():
        print(f"\n--- {column.upper()} ---")
        print(f"Accuracy: {metric['accuracy']:.4f}")
        print(f"Precision: {metric['precision']:.4f}")
        print(f"Recall: {metric['recall']:.4f}")
        print(f"F1-Score: {metric['f1-score']:.4f}")
        print("\nConfusion Matrix:")
        print(metric['confusion_matrix'])
        print("\nClassification Report:")
        print(metric['classification_report'])


csv_directory = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/IRC_labeled_judgments'
all_metrics = []

for csv_file_path in glob.glob(os.path.join(csv_directory, "*.csv")):
    casename = extract_casename_from_filename(csv_file_path)
    annotated_df = pd.read_csv(csv_file_path)

    pipeline_df = run_pipeline(casename)

    metrics = evaluate_pipeline(annotated_df, pipeline_df)
    all_metrics.append(metrics)

overall_metrics = {}
columns = ['main_issue', 'main_conclusion', 'issue_index']

print('\n\n\n\n\n\n\n')

for column in columns:
    overall_accuracy = sum(metric[column]['accuracy'] for metric in all_metrics) / len(all_metrics)
    overall_precision = sum(metric[column]['precision'] for metric in all_metrics) / len(all_metrics)
    overall_recall = sum(metric[column]['recall'] for metric in all_metrics) / len(all_metrics)
    overall_f1 = sum(metric[column]['f1-score'] for metric in all_metrics) / len(all_metrics)

    overall_metrics[column] = {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1-score': overall_f1
    }

print("\n--- OVERALL ACCURACY REPORT ---")
for column, metric in overall_metrics.items():
    print(f"\n--- {column.upper()} ---")
    print(f"Average Accuracy: {metric['accuracy']:.4f}")
    print(f"Average Precision: {metric['precision']:.4f}")
    print(f"Average Recall: {metric['recall']:.4f}")
    print(f"Average F1-Score: {metric['f1-score']:.4f}")



