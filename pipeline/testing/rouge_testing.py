
import os
from rouge_score import rouge_scorer


def load_summary(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


generated_summaries_dir = '../arg_mining/data/IRC_summaries/BERTScore'
reference_summaries_dir = '../arg_mining/data/SUMO_summaries/BERTScore'

generated_summaries = []
reference_summaries = []

for file_name in os.listdir(generated_summaries_dir):
    if file_name.endswith('.txt'):
        generated_file_path = os.path.join(generated_summaries_dir, file_name)
        reference_file_path = os.path.join(reference_summaries_dir, file_name)

        if not os.path.exists(reference_file_path):
            raise FileNotFoundError(f"Reference summary not found for {file_name}")

        generated_summaries.append(load_summary(generated_file_path))
        reference_summaries.append(load_summary(reference_file_path))

if len(generated_summaries) != len(reference_summaries):
    raise ValueError("The number of generated summaries must match the number of reference summaries.")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge_1_precisions = []
rouge_1_recalls = []
rouge_1_f1s = []

rouge_2_precisions = []
rouge_2_recalls = []
rouge_2_f1s = []

rouge_l_precisions = []
rouge_l_recalls = []
rouge_l_f1s = []

for generated, reference in zip(generated_summaries, reference_summaries):
    scores = scorer.score(reference, generated)

    rouge_1_scores = scores['rouge1']
    rouge_1_precisions.append(rouge_1_scores.precision)
    rouge_1_recalls.append(rouge_1_scores.recall)
    rouge_1_f1s.append(rouge_1_scores.fmeasure)

    rouge_2_scores = scores['rouge2']
    rouge_2_precisions.append(rouge_2_scores.precision)
    rouge_2_recalls.append(rouge_2_scores.recall)
    rouge_2_f1s.append(rouge_2_scores.fmeasure)

    rouge_l_scores = scores['rougeL']
    rouge_l_precisions.append(rouge_l_scores.precision)
    rouge_l_recalls.append(rouge_l_scores.recall)
    rouge_l_f1s.append(rouge_l_scores.fmeasure)

print(f"Average ROUGE-1 Precision: {sum(rouge_1_precisions) / len(rouge_1_precisions):.4f}")
print(f"Average ROUGE-1 Recall: {sum(rouge_1_recalls) / len(rouge_1_recalls):.4f}")
print(f"Average ROUGE-1 F1: {sum(rouge_1_f1s) / len(rouge_1_f1s):.4f}")

print(f"Average ROUGE-2 Precision: {sum(rouge_2_precisions) / len(rouge_2_precisions):.4f}")
print(f"Average ROUGE-2 Recall: {sum(rouge_2_recalls) / len(rouge_2_recalls):.4f}")
print(f"Average ROUGE-2 F1: {sum(rouge_2_f1s) / len(rouge_2_f1s):.4f}")

print(f"Average ROUGE-L Precision: {sum(rouge_l_precisions) / len(rouge_l_precisions):.4f}")
print(f"Average ROUGE-L Recall: {sum(rouge_l_recalls) / len(rouge_l_recalls):.4f}")
print(f"Average ROUGE-L F1: {sum(rouge_l_f1s) / len(rouge_l_f1s):.4f}")

# import os
# # from IRC_TripleSummaryPipeline import IRC_pipeline
# from rouge_score import rouge_scorer
#
#
# def read_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()
#
#
# def calculate_rouge_scores(summary, reference):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
#     scores = scorer.score(summary, reference)
#
#     rouge_su4_score = scores['rouge2'].fmeasure
#
#     return {
#         'ROUGE-1': scores['rouge1'].fmeasure,
#         'ROUGE-2': scores['rouge2'].fmeasure,
#         'ROUGE-L': scores['rougeL'].fmeasure,
#         'ROUGE-SU4': rouge_su4_score
#     }
#
#
# def calculate_rouge_for_directory(summary_file, reference_dir):
#     summary = read_file(summary_file)
#
#     for ref_file in os.listdir(reference_dir):
#         ref_path = os.path.join(reference_dir, ref_file)
#         reference = read_file(ref_path)
#
#         rouge_scores = calculate_rouge_scores(summary, reference)
#
#         print(f"ROUGE Scores for summary against {ref_file}:")
#         for metric, score in rouge_scores.items():
#             print(f"  {metric}: {score:.4f}")
#         print("-" * 40)
#
#
# if __name__ == "__main__":
#     summary_file = '../arg_mining/data/IRC_summaries/UKHL200223.txt'
#     reference_dir = '../arg_mining/data/reference_summaries'
#
#     calculate_rouge_for_directory(summary_file, reference_dir)


# def process_directory(input_directory):
#     cases = {}
#
#     for file_name in os.listdir(input_directory):
#         if file_name.endswith('.txt'):
#             file_path = os.path.join(input_directory, file_name)
#             url, ICLR_summary = extract_url_and_summary(file_path)
#
#             key = os.path.splitext(file_name)[0]
#             cases[key] = {
#                 'url': url,
#                 'ICLR_summary': ICLR_summary,
#                 'IRC_summary': ''
#             }
#     return cases
#
#
# def extract_url_and_summary(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     url = lines[0].strip()
#     ICLR_summary = ''.join(lines[1:]).strip()
#     return url, ICLR_summary
#
#
# def generate_pipeline_sums(cases):
#     pipeline = IRC_pipeline()
#     for key in cases:
#         print('\n')
#         print(key)
#         print(cases[key]['url'])
#         cases[key]['IRC_summary'] = pipeline.begin(cases[key]['url'])
#     return cases
#
# def calculate_rouge_scores(cases):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', ], use_stemmer=True)
#     rouge_scores = {}
#     total_rouge1, total_rouge2, total_rougeL = 0.0, 0.0, 0.0
#     num_cases = len(cases)
#
#     for key in cases:
#         ICLR_summary = cases[key]['ICLR_summary']
#         IRC_summary = cases[key]['IRC_summary']
#
#         if isinstance(ICLR_summary, list):
#             ICLR_summary = ' '.join(ICLR_summary)
#         if isinstance(IRC_summary, list):
#             IRC_summary = ' '.join(IRC_summary)
#
#         scores = scorer.score(ICLR_summary, IRC_summary)
#         rouge_scores[key] = {
#             'rouge1': scores['rouge1'].fmeasure,
#             'rouge2': scores['rouge2'].fmeasure,
#             'rougeL': scores['rougeL'].fmeasure,
#         }
#
#         # Accumulate the scores
#         total_rouge1 += rouge_scores[key]['rouge1']
#         total_rouge2 += rouge_scores[key]['rouge2']
#         total_rougeL += rouge_scores[key]['rougeL']
#
#         # Print ROUGE scores for each case
#         print(f"\nROUGE scores for {key}:")
#         print(f"ROUGE-1: {rouge_scores[key]['rouge1']:.4f}")
#         print(f"ROUGE-2: {rouge_scores[key]['rouge2']:.4f}")
#         print(f"ROUGE-L: {rouge_scores[key]['rougeL']:.4f}")
#
#     # Calculate and print average ROUGE scores
#     avg_rouge1 = total_rouge1 / num_cases
#     avg_rouge2 = total_rouge2 / num_cases
#     avg_rougeL = total_rougeL / num_cases
#
#     return rouge_scores, {'avg_rouge1': avg_rouge1, 'avg_rouge2': avg_rouge2, 'avg_rougeL': avg_rougeL}
#
#
# input_mj = './arg_mining/data/ICLR_summaries/mj'
# cases_mj = process_directory(input_mj)
# cases_mj = generate_pipeline_sums(cases_mj)
# rouge_scores_mj, avg_rouge_scores_mj = calculate_rouge_scores(cases_mj)
#
# input_no_mj = './arg_mining/data/ICLR_summaries/no_mj'
# cases_no_mj = process_directory(input_no_mj)
# cases_no_mj = generate_pipeline_sums(cases_no_mj)
# rouge_scores_no_mj, avg_rouge_scores_no_mj = calculate_rouge_scores(cases_no_mj)
#
# print("\nResults for Cases with ASMO identified majority:")
# print(f"Average ROUGE-1: {avg_rouge_scores_mj['avg_rouge1']:.4f}")
# print(f"Average ROUGE-2: {avg_rouge_scores_mj['avg_rouge2']:.4f}")
# print(f"Average ROUGE-L: {avg_rouge_scores_mj['avg_rougeL']:.4f}\n")
#
# print("\nResults for Cases without ASMO identified majority:")
# print(f"Average ROUGE-1: {avg_rouge_scores_no_mj['avg_rouge1']:.4f}")
# print(f"Average ROUGE-2: {avg_rouge_scores_no_mj['avg_rouge2']:.4f}")
# print(f"Average ROUGE-L: {avg_rouge_scores_no_mj['avg_rougeL']:.4f}\n")
