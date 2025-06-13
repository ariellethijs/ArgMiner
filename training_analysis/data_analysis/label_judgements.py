import pandas as pd

# • Issue – Legal question which a court addressed in the case.
# • Conclusion – Court’s decision for the corresponding issue.
# • Reason – Sentences that elaborate on why the court reached the Conclusion.


def label_judgment_lines(input_file, output_file):
    df = pd.read_csv(input_file)
    df.to_csv(output_file, index=False)

    labels = ['ISSUE', 'REASON', 'CONCLUSION', 'NON-IRC']

    roles = ['NON-IRC'] * len(df)

    i = 0
    while i < len(df):
        text = df.loc[i, 'text'].strip()
        if not text:
            i += 1
            continue

        print(f"\nText {i + 1}: {text}")
        print("\nLabels: 1) ISSUE, 2) REASON, 3) CONCLUSION 4) NON-IRC")
        print("Enter 0 to go back to the previous text.")

        while True:
            try:
                user_input = int(input("Enter the label number: "))
                if user_input in [1, 2, 3, 4]:
                    label = labels[user_input - 1]
                    roles[i] = label
                    i += 1
                    break
                elif user_input == 0 and i > 0:
                    i -= 1
                    print("Going back to the previous text.")
                    print(f"Current text: {df.loc[i, 'text'].strip()}")
                    print(f"Current label: {roles[i]}")
                else:
                    print("Invalid input. Please enter 1, 2, 3, 4, or 0.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    df['arg_role'] = roles

    df.to_csv(output_file, index=False)

    print(f"Labeled data saved to {output_file}")


print("Which case would you like to label?")
casenum = input()
input_file = f"../../pipeline/data/UKHL_corpus/UKHL_{casenum}.csv"
output_file = f"../Dataset/IRC_labeled_judgments/UKHL_{casenum}_labeled.csv"

label_judgment_lines(input_file, output_file)

# import json
# import nltk
#
# # Download necessary NLTK data (uncomment if needed)
# # nltk.download('punkt')
#
#
# def split_into_sentences(text):
#     return nltk.sent_tokenize(text)
#
#
# def label_judgment_lines(input_file, output_path):
#     labeled_data = []
#
#     # Read the judgment from the file
#     with open(input_file, 'r') as file:
#         text = file.read()
#
#     # Split the text into sentences
#     sentences = split_into_sentences(text)
#
#     # Define possible labels
#     labels = ['ISSUE', 'REASON', 'CONCLUSION', 'NON-IRC']
#
#     i = 0  # Index for tracking current sentence
#     while i < len(sentences):
#         sentence = sentences[i].strip()
#         if not sentence:
#             i += 1
#             continue
#
#         print(f"\nSentence {i + 1}: {sentence}")
#         print("\nLabels: 1) ISSUE, 2) REASON, 3) CONCLUSION 4) NON-IRC")
#         print("Enter 0 to go back to the previous sentence.")
#
#         while True:
#             try:
#                 user_input = int(input("Enter the label number: "))
#                 if user_input in [1, 2, 3, 4]:
#                     label = labels[user_input - 1]
#                     labeled_data.append({'text': sentence, 'label': label})
#                     i += 1  # Move to the next sentence
#                     break
#                 elif user_input == 0 and i > 0:
#                     # Go back to the previous sentence
#                     i -= 1
#                     print("Going back to the previous sentence.")
#                     print(f"Current sentence: {sentences[i].strip()}")
#                 else:
#                     print("Invalid input. Please enter 1, 2, 3, 4, or 0.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")
#
#     # Save the labeled data to a JSON file
#     with open(output_path, 'w') as json_file:
#         json.dump(labeled_data, json_file, indent=4)
#
#     print(f"Labeled data saved to {output_path}")
#
#
# print("Which case would you like to label?")
# casenum = input()
# file_path = "/Users/ariellethijssen/areel/MscCS/SummerProject/nolansproject/2024/pipeline/data/UKHL_corpus/UKHL_" + casenum + ".txt"
# output_path = "/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments/" + casenum + ".json"
#
# label_judgment_lines(file_path, output_path)


