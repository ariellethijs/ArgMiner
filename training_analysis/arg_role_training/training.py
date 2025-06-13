import os
import re

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from dataset import ArgRoleDataset


def combine_data_files(specific_files):
    directory = '/Users/ariellethijssen/areel/MscCS/SummerProject/Dataset/manually_labeled_judgments'
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


def load_best_model(model_dir='./model/zlucia'):
    model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert', num_labels=4)
    files = os.listdir(model_dir)

    highest_epoch = -1
    best_model_path = None

    pattern = re.compile(r'_(\d+)_epochs\.pth')

    for file in files:
        match = pattern.search(file)
        if match:
            try:
                epoch_count = int(match.group(1))

                if epoch_count > highest_epoch:
                    highest_epoch = epoch_count
                    best_model_path = os.path.join(model_dir, file)
            except ValueError:
                continue

    if best_model_path is None:
        return model, 0

    model.load_state_dict(torch.load(best_model_path))
    return model, highest_epoch


def train_validate_model(model, train_loader, valid_loader, optimizer, scheduler, epochs, device, highest_epoch):
    for epoch in range(0, epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = len(train_loader)
        for batch_count, batch in enumerate(train_loader):
            print(f"training epoch: {epoch} batch: {batch_count}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / num_train_batches

        model.eval()
        total_val_loss = 0.0
        num_val_batches = len(valid_loader)
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / num_val_batches

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_train_loss}, Validation Loss: {average_val_loss}")

    # Save the model
    model_save_path = f'./model/zlucia/legalbert_finetuned_arg_roles_{epochs + highest_epoch}_epochs.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model


model_name = 'zlucia/legalbert'  # LegalBERT model
tokenizer_name = 'bert-base-uncased'
max_seq_length = 128
batch_size = 8
learning_rate = 2e-5
epochs = 3
training_files = ['UKHL_1.04_labeled.csv', 'UKHL_1.05_labeled.csv', 'UKHL_1.11_labeled.csv', 'UKHL_1.15_labeled.csv', 'UKHL_1.26_labeled.csv', 'UKHL_1.28_labeled.csv', 'UKHL_1.35_labeled.csv']
validating_files = ['UKHL_1.38_labeled.csv']
train_df = combine_data_files(training_files)
valid_df = combine_data_files(validating_files)
train_dataset = ArgRoleDataset(train_df, max_seq_length, tokenizer_name)
valid_dataset = ArgRoleDataset(valid_df, max_seq_length, tokenizer_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model initialization
# model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert', num_labels=4)
model, highest_epoch = load_best_model()
print(f'Best model found trained for {highest_epoch} epochs')
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

model = train_validate_model(model, train_loader, valid_loader, optimizer, scheduler, epochs, device, highest_epoch)
