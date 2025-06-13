import pandas as pd
import textwrap


def wrap_text(text, width=80):
    return textwrap.fill(text, width=width)


def prompt_user_for_clusters(texts):
    clusters = []
    current_cluster = []
    used_texts = set()

    terminal_width = 80

    num_texts = len(texts)
    for i in range(num_texts):
        for j in range(i + 1, num_texts):
            text1 = texts[i]
            text2 = texts[j]

            if text1 in used_texts and text2 in used_texts:
                continue

            wrapped_text1 = wrap_text(text1, width=terminal_width)
            wrapped_text2 = wrap_text(text2, width=terminal_width)

            print("\nText 1:")
            print(wrapped_text1)
            print("\nText 2:")
            print(wrapped_text2)

            response = input("\nShould these two texts be in the same cluster? (y/n): ").strip().lower()
            if response == 'y':
                if not current_cluster or text1 in current_cluster or text2 in current_cluster:
                    if text1 not in current_cluster:
                        current_cluster.append(text1)
                        used_texts.add(text1)
                    if text2 not in current_cluster:
                        current_cluster.append(text2)
                        used_texts.add(text2)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [text1, text2]
                    used_texts.add(text1)
                    used_texts.add(text2)
            elif response == 'n':
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []

    if current_cluster:
        clusters.append(current_cluster)

    all_texts = set(texts)
    clustered_texts = used_texts
    unclustered_texts = all_texts - clustered_texts

    for text in unclustered_texts:
        clusters.append([text])

    return clusters


def review_clusters(clusters):
    print("\nCurrent clusters:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")

    while True:
        try:
            action = int(input("\nEnter 0 to finalize, 1 to split a cluster, or 2 to merge clusters: ").strip())
            if action == 0:
                break
            elif action == 1:
                cluster_index = int(input("Enter the cluster number to split: ").strip()) - 1
                if 0 <= cluster_index < len(clusters):
                    new_cluster = []
                    split_text = input("Enter the text to move to a new cluster: ").strip()
                    if split_text in clusters[cluster_index]:
                        new_cluster.append(split_text)
                        clusters[cluster_index].remove(split_text)
                        clusters.append(new_cluster)
                    else:
                        print("Text not found in the selected cluster.")
                else:
                    print("Invalid cluster number.")
            elif action == 2:
                cluster1_index = int(input("Enter the first cluster number to merge: ").strip()) - 1
                cluster2_index = int(input("Enter the second cluster number to merge: ").strip()) - 1
                if (0 <= cluster1_index < len(clusters)) and (0 <= cluster2_index < len(clusters)):
                    clusters[cluster1_index].extend(clusters.pop(cluster2_index))
                else:
                    print("Invalid cluster numbers.")
            else:
                print("Invalid option.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return clusters


def update_dataframe_with_clusters(df, clusters):
    text_to_cluster_id = {}
    for cluster_id, cluster in enumerate(clusters):
        for text in cluster:
            text_to_cluster_id[text] = cluster_id

    df['cluster_id'] = pd.NA
    df.loc[df['arg_role'] == 'ISSUE', 'cluster_id'] = df.loc[df['arg_role'] == 'ISSUE', 'text'].map(text_to_cluster_id)

    return df


def main():
    print('Which case would you like to label clusters for?')
    casenum = input()

    input_file_path = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments/UKHL_' \
                      + casenum + '_labeled.csv'
    output_file_path = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/clustered_dataset_issues/UKHL_' \
                       + casenum + '_labeled_clusters.csv'
    df = pd.read_csv(input_file_path)
    issues_df = df[df['arg_role'] == 'ISSUE'].reset_index(drop=True)
    texts = issues_df['text'].tolist()
    clusters = prompt_user_for_clusters(texts)
    clusters = review_clusters(clusters)
    updated_df = update_dataframe_with_clusters(df, clusters)
    updated_df.to_csv(output_file_path, index=False)
    print(f"Updated dataset with clusters saved to {output_file_path}")


if __name__ == '__main__':
    main()
