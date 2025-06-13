import textwrap

import pandas as pd
import os


def wrap_text(text, width=80):
    return '\n'.join(textwrap.wrap(text, width=width))


def label_issues(input_file_path, output_df):
    df = pd.read_csv(input_file_path)
    relevant_columns = ['case_id', 'sentence_id', 'judge', 'role', 'arg_role', 'text']
    df = df[relevant_columns]
    issues_df = df[df['arg_role'] == 'ISSUE'].reset_index(drop=True)
    texts = issues_df['text'].tolist()

    new_rows = []
    labeled_texts = []
    index = 0

    while index < len(issues_df):
        row = issues_df.iloc[index]
        text = row['text']
        wrapped_text = wrap_text(text)

        while True:
            print(f'\nText:\n{wrapped_text}')
            print('\nEnter "1" for MAIN issue, "2" for SUB issue, or "0" to go back to the previous text.')
            user_input = input().strip()

            if user_input == '1':
                issue_type_label = 'MAIN'
                labeled_texts.append(index)
                index += 1
                break
            elif user_input == '2':
                issue_type_label = 'SUB'
                labeled_texts.append(index)
                index += 1
                break
            elif user_input == '0':
                if labeled_texts:
                    index = labeled_texts.pop()
                else:
                    print('No previous text to go back to.')
                break
            else:
                print('Invalid input. Please enter "1" for MAIN, "2" for SUB, or "0" to go back.')

        if user_input in ['1', '2']:
            new_row = row.to_dict()
            new_row['type'] = issue_type_label
            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    output_df = pd.concat([output_df, new_df], ignore_index=True)

    return output_df


def process_directory(directory_path, output_file_path):
    combined_df = pd.DataFrame(columns=['text', 'type'])

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            input_file_path = os.path.join(directory_path, filename)
            print(f'\n --------NEW CASE-------\n')
            print(f'Processing file: {filename}')

            combined_df = label_issues(input_file_path, combined_df)

    combined_df.to_csv(output_file_path, index=False)
    print(f'All labeled issues saved to {output_file_path}')

directory_path = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments/'
output_file_path = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/main_v_sub_issues_analysis/combined_labeled_issues.csv'

process_directory(directory_path, output_file_path)

