import os
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt


def parse_text(text_content):
    texts = []
    labels = []

    pattern_text = r"SENTENCE CLAIM_SCORE:([-\d.]+) EVIDENCE_SCORE:([-\d.]+) TEXT:(.*)$"
    pattern_label_not_arg = r"^NOT_ARGUMENTATIVE (.*)$"
    pattern_label_evidence = r"^EVIDENCE (.*)$"
    pattern_label_claim = r"^CLAIM (.*)$"
    pattern_label_claim_evidence = r"^CLAIM_EVIDENCE (.*)$"

    lines = text_content.strip().split("\n")
    i = 0
    while i < len(lines):
        match_text = re.match(pattern_text, lines[i].strip())
        if match_text:
            # claim_score = float(match_text.group(1))
            # evidence_score = float(match_text.group(2))
            text = match_text.group(3).strip()
            texts.append(text)
            i += 1

            if i < len(lines):
                match_label_not_arg = re.match(pattern_label_not_arg, lines[i].strip())
                match_label_evidence = re.match(pattern_label_evidence, lines[i].strip())
                match_label_claim = re.match(pattern_label_claim, lines[i].strip())
                match_label_claim_evidence = re.match(pattern_label_claim_evidence, lines[i].strip())

                if match_label_not_arg:
                    labels.append("NOT_ARGUMENTATIVE")
                elif match_label_evidence:
                    labels.append("EVIDENCE")
                elif match_label_claim:
                    labels.append("CLAIM")
                elif match_label_claim_evidence:
                    labels.append("CLAIM_EVIDENCE")
                else:
                    labels.append("")

                i += 1
        else:
            i += 1

    return texts, labels

def process_files_in_directory(root_dir):
    all_text = []
    all_labels = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(subdir, file)
                print(f"Processing file: {filepath}")

                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()

                texts, labels = parse_text(text_content)

                all_text.extend(texts)
                all_labels.extend(labels)

    df = pd.DataFrame({'text': all_text, 'label': all_labels})
    return df

def combine_rhetorical_and_argument_roles(casename, rhetorical_df, argument_df, output_path):
    combined_rhetorical_data = []
    combined_text = ""
    current_role = ""

    for index, row in rhetorical_df.iterrows():
        if current_role and current_role != row['role']:
            combined_rhetorical_data.append({'text': combined_text.strip(), 'role': current_role})
            combined_text = row['text']
        else:
            combined_text += " " + row['text']
        current_role = row['role']

    if combined_text:
        combined_rhetorical_data.append({'text': combined_text.strip(), 'role': current_role})

    combined_rhetorical_df = pd.DataFrame(combined_rhetorical_data)

    combined_labeled_data = []

    for arg_index, arg_row in argument_df.iterrows():
        for rhet_index, rhet_row in combined_rhetorical_df.iterrows():
            if arg_row['text'] in rhet_row['text'] or rhet_row['text'] in arg_row['text']:
                combined_labeled_data.append({
                    'text': arg_row['text'],
                    'argument_role': arg_row['label'],
                    'rhetorical_role': rhet_row['role']
                })
                break

    combined_df = pd.DataFrame(combined_labeled_data)

    combined_df.to_csv(output_path + casename + '_combined.csv', index=False)
    print(f"Combined data saved to {output_path + casename + '_combined.csv'}")
    return combined_df


def extract_and_compare(combined_df):
    claims = combined_df[combined_df['argument_role'] == 'CLAIM']
    evidence = combined_df[combined_df['argument_role'] == 'EVIDENCE']
    claim_evidence = combined_df[combined_df['argument_role'] == 'CLAIM_EVIDENCE']

    claims_roles = claims['rhetorical_role'].tolist()
    evidence_roles = evidence['rhetorical_role'].tolist()
    claim_evidence_roles = claim_evidence['rhetorical_role'].tolist()

    return claims_roles, evidence_roles, claim_evidence_roles


def analyze_patterns(claims_roles, evidence_roles, claim_evidence_roles):
    claims_counter = Counter(claims_roles)
    evidence_counter = Counter(evidence_roles)
    claim_evidence_counter = Counter(claim_evidence_roles)

    # Plotting the results
    labels = list(set(claims_roles + evidence_roles + claim_evidence_roles))
    claims_values = [claims_counter[label] for label in labels]
    evidence_values = [evidence_counter[label] for label in labels]
    claim_evidence_values = [claim_evidence_counter[label] for label in labels]

    x = range(len(labels))

    width = 0.2
    plt.bar(x, claims_values, width, label='Claims', align='center')
    plt.bar([p + width for p in x], evidence_values, width, label='Evidence', align='center')
    plt.bar([p + width * 2 for p in x], claim_evidence_values, width, label='Claim_Evidence', align='center')

    plt.xlabel('Rhetorical Roles')
    plt.ylabel('Frequency')
    plt.title('Comparison of Rhetorical Roles in Claims, Evidence, and Claim_Evidence')
    plt.xticks([p + width for p in x], labels, rotation='vertical')
    plt.legend()

    output_path = "/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/rhet_arg_comparison_graphs/" + casename + ".png"
    plt.savefig(output_path)

    # Save the results as a table
    results_df = pd.DataFrame({
        'Rhetorical_Role': labels,
        'Claims_Count': claims_values,
        'Evidence_Count': evidence_values,
        'Claim_Evidence_Count': claim_evidence_values
    })

    results_output_path = "/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/rhet_arg_comparison_graphs/" \
                          + casename + "_results.csv"
    results_df.to_csv(results_output_path, index=False)


print("Which case would you like to create annotated dataset from?")
casename = input()
directory_to_process = "/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/MARGOT/" + casename

argument_df = process_files_in_directory(directory_to_process)

rhet_df = pd.read_csv('/Users/ariellethijssen/areel/MscCS/SummerProject/nolansproject/2024/pipeline/data/UKHL_corpus2/' + casename + '.csv')

output_path = "/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/rhet_arg_roles_combined/"

combined_df = combine_rhetorical_and_argument_roles(casename, rhet_df, argument_df, output_path)

claims_roles, evidence_roles, claim_evidence_roles = extract_and_compare(combined_df)

analyze_patterns(claims_roles, evidence_roles, claim_evidence_roles)
