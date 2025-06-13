import os
import pandas as pd
import matplotlib.pyplot as plt

def read_result_files(directory):
    all_data = []
    for file in os.listdir(directory):
        if file.endswith("_results.csv"):
            file_path = os.path.join(directory, file)
            data = pd.read_csv(file_path)
            data['casename'] = os.path.splitext(file)[0]
            all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


def analyse_combined_results(directory):
    combined_data = read_result_files(directory)

    aggregated_data = combined_data.groupby('Rhetorical_Role').sum().reset_index()

    plt.figure(figsize=(12, 8))
    x = range(len(aggregated_data))

    plt.bar(x, aggregated_data['Claims_Count'], width=0.2, label='Claims', align='center')
    plt.bar(x, aggregated_data['Evidence_Count'], width=0.2, label='Evidence', align='edge')
    plt.bar(x, aggregated_data['Claim_Evidence_Count'], width=0.2, label='Claim_Evidence', align='edge')

    plt.xlabel('Rhetorical Roles')
    plt.ylabel('Frequency')
    plt.title('Comparison of Rhetorical Roles and Argumentative Roles Across Cases')
    plt.xticks(x, aggregated_data['Rhetorical_Role'], rotation='vertical')
    plt.legend()

    output_graph_path = os.path.join(directory, 'combined_rhet_arg_comparison.png')
    plt.savefig(output_graph_path)
    plt.close()

    output_table_path = os.path.join(directory, 'combined_rhet_arg_comparison.csv')
    aggregated_data.to_csv(output_table_path, index=False)
    print(f"Combined data saved as {output_table_path} and {output_graph_path}")


directory_to_process = "/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/rhet_arg_comparison_graphs"
analyse_combined_results(directory_to_process)
