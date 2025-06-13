import csv
import os

role_styles = {
    "NON-IRC": "color: #34495e; font-weight: bold;",
    "ISSUE": "color: #f39c12; font-weight: bold;",
    "REASON": "color: #2ecc71; font-weight: bold;",
    "CONCLUSION": "color: #e74c3c; font-weight: bold;"
}

def get_styled_text(text, role, issue_index):
    style = role_styles.get(role, "color: #000000;")
    return f'<span style="{style}">[{issue_index}] {text}</span>'

input_directory = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/IRC_labeled_judgments'
# input_directory = '/Users/ariellethijssen/areel/MscCS/SummerProject/NolansProject/2024/pipeline/arg_mining/data/arg_role_labelled'

for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(input_directory, filename.replace('.csv', '.html'))

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write('<html><body>\n')
            with open(input_file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    arg_role = row['arg_role']
                    if arg_role == "NON-IRC":
                        continue

                    text = row['text']
                    issue_index = row['issue_index']
                    styled_text = get_styled_text(text, arg_role, issue_index)
                    output_file.write(styled_text + '<br>\n')
            output_file.write('</body></html>')
