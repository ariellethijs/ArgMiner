import os
import pandas as pd
import torch
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from dataset import ArgRoleDataset

def combine_data_files(specific_files):
    directory = '../Dataset/meya_labelled'
    dataframes = []

    for file in specific_files:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataframes.append(df)
        else:
            print(f"File {file} does not exist in the directory")

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

    print(classification_report(true_labels, predictions, labels=[0, 1, 2, 3]))

    mcm = multilabel_confusion_matrix(true_labels, predictions)
    print("Multilabel Confusion Matrix:")
    print(mcm)

model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert', num_labels=4)
model_save_path = './model/zlucia/legalbert_finetuned_arg_roles.pth'
model.load_state_dict(torch.load(model_save_path))

testing_files = ['UKHL_2.35_labeled.csv', 'UKHL_1.43_labeled.csv']
test_df = combine_data_files(testing_files)

test_dataset = ArgRoleDataset(test_df, 128, 'bert-base-uncased')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_model(model, test_loader, device)

