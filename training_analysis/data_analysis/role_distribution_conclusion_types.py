import pandas as pd
import os


def read_csv_files(directory):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(directory, f)) for f in all_files]
    return pd.concat(dfs, ignore_index=True)


def filter_and_split_dataframe(df):
    filtered_df = df[df['arg_role'] == 'CONCLUSION']
    df_main_conclusion_1 = filtered_df[filtered_df['main_conclusion'] == 1]
    df_main_conclusion_0 = filtered_df[filtered_df['main_conclusion'] == 0]
    return df_main_conclusion_1, df_main_conclusion_0


def calculate_role_distribution(df):
    if df.empty:
        return pd.Series(dtype=float)
    role_counts = df['role'].value_counts(normalize=True) * 100
    return role_counts


def compare_distributions(df1, df2):
    dist1 = calculate_role_distribution(df1)
    dist2 = calculate_role_distribution(df2)

    all_roles = pd.Index(dist1.index).union(dist2.index)

    dist1 = dist1.reindex(all_roles, fill_value=0)
    dist2 = dist2.reindex(all_roles, fill_value=0)

    comparison_df = pd.DataFrame({
        'Role': all_roles,
        'Main Conclusion = 1 (%)': dist1.values,
        'Main Conclusion = 0 (%)': dist2.values
    }).set_index('Role')

    return comparison_df


def main(directory):
    df = read_csv_files(directory)

    df_main_conclusion_1, df_main_conclusion_0 = filter_and_split_dataframe(df)
    main_output = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/main_conclusions.csv'
    df_main_conclusion_1.to_csv(main_output, index=False)

    comparison_df = compare_distributions(df_main_conclusion_1, df_main_conclusion_0)

    print(comparison_df)


if __name__ == "__main__":
    main('/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/IRC_labeled_judgments')
