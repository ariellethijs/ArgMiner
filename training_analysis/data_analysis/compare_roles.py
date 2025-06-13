import pandas as pd
import os
from tabulate import tabulate


def count_roles_from_directory(directory_path, output_file_path):
    df_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)

    if not {'text', 'arg_role', 'role'}.issubset(combined_df.columns):
        raise ValueError("The input CSV files must contain 'text', 'arg_role', and 'role' columns.")

    arg_roles = ["NON-IRC", "ISSUE", "REASON", "CONCLUSION"]
    roles = ["NONE", "TEXTUAL", "FACT", "PROCEEDINGS", "BACKGROUND", "FRAMING", "DISPOSAL"]

    counts = pd.DataFrame(index=arg_roles, columns=roles).fillna(0)

    for arg_role in arg_roles:
        for role in roles:
            count = len(combined_df[(combined_df['arg_role'] == arg_role) & (combined_df['role'] == role)])
            counts.at[arg_role, role] = count

    table = counts.reset_index()
    table.columns.name = None
    table = table.rename(columns={'index': 'arg_role'})

    print(tabulate(table, headers='keys', tablefmt='grid', showindex=False))

    with open(output_file_path, 'w') as file:
        file.write(tabulate(table, headers='keys', tablefmt='grid', showindex=False))


directory_path = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments'
output_file_path = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/DATASET_roles_comparison_table.txt'
count_roles_from_directory(directory_path, output_file_path)
