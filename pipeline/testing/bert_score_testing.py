import os
import bert_score

def load_summary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        summary = file.read().strip()
    return summary

generated_summaries_dir = '../arg_mining/data/IRC_summaries/BERTScore'
reference_summaries_dir = '../arg_mining/data/ICLR_summaries/mj'

generated_summaries = []
reference_summaries = []

for file_name in os.listdir(generated_summaries_dir):
    if file_name.endswith('.txt'):
        print(file_name)
        generated_file_path = os.path.join(generated_summaries_dir, file_name)
        reference_file_path = os.path.join(reference_summaries_dir, file_name)

        if not os.path.exists(reference_file_path):
            raise FileNotFoundError(f"Reference summary not found for {file_name}")

        generated_summaries.append(load_summary(generated_file_path))
        reference_summaries.append(load_summary(reference_file_path))

if len(generated_summaries) != len(reference_summaries):
    raise ValueError("The number of generated summaries must match the number of reference summaries.")

P, R, F1 = bert_score.score(generated_summaries, reference_summaries, lang="en", model_type="bert-base-uncased")

for idx, file_name in enumerate([f for f in os.listdir(generated_summaries_dir) if f.endswith('.txt')]):
    print(f"File: {file_name}")
    print(f"  Precision: {P[idx].item():.4f}")
    print(f"  Recall: {R[idx].item():.4f}")
    print(f"  F1: {F1[idx].item():.4f}")
    print("-" * 30)

print(f"Average Precision: {P.mean().item():.4f}")
print(f"Average Recall: {R.mean().item():.4f}")
print(f"Average F1: {F1.mean().item():.4f}")


