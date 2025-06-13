import os
import torch
# import time
import re
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from dataset import RhetRoleDataset


def save_checkpoint(model, optimizer, epoch, batch_count, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'batch_count': batch_count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"checkpoint saved at {checkpoint_path}")


def load_most_recent_model(device):
    model_directory = './model'
    model_files = os.listdir(model_directory)

    model_files = [f for f in model_files if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]

    model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert', num_labels=7)

    if not model_files:
        return model, 0, 0

    pattern = re.compile(r'checkpoint_epoch_(\d+)_batch_(\d+).pt')
    checkpoints = [(int(m.group(1)), int(m.group(2)), f) for f in model_files if (m := pattern.match(f))]
    checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)

    latest_epoch, latest_batch, latest_checkpoint_file = checkpoints[0]

    checkpoint_path = os.path.join(model_directory, latest_checkpoint_file)
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Correctly load the checkpoint

    model.load_state_dict(checkpoint['model_state_dict'])  # Load model state
    optimizer = AdamW(model.parameters(), lr=learning_rate)  # Initialize optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state

    print(f"Loaded checkpoint from epoch {latest_epoch}, batch {latest_batch}")

    return model, latest_epoch, latest_batch


def train_validate_model(model, train_loader, valid_loader, optimizer, scheduler, epochs, start_epoch, start_batch):
    save_frequency = 500
    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = len(train_loader) - start_batch
        for batch_count, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_count < start_batch:
                continue

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

            # save checkpoint ever 500 batches
            if (batch_count + 1) % save_frequency == 0:
                checkpoint_path = f'./model/checkpoint_epoch_{epoch}_batch_{batch_count + 1}.pt'
                save_checkpoint(model, optimizer, epoch, batch_count + 1, checkpoint_path)

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

        # save checkpoint every epoch after validation
        checkpoint_path = f'./model/checkpoint_epoch_{epoch}.pt'
        save_checkpoint(model, optimizer, epoch + 1, 0, checkpoint_path)

    model_save_path = './model/legalbert_finetuned.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model


model_name = 'zlucia/legalbert'  # LegalBERT model
tokenizer_name = 'bert-base-uncased'
max_seq_length = 128
batch_size = 8
learning_rate = 2e-5
epochs = 4

train_dataset = RhetRoleDataset('./data/train_dataset.csv', max_seq_length, tokenizer_name)
valid_dataset = RhetRoleDataset('./data/valid_dataset.csv', max_seq_length, tokenizer_name)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, start_epoch, start_batch = load_most_recent_model(device)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

model = train_validate_model(model, train_loader, valid_loader, optimizer, scheduler, epochs, start_epoch, start_batch)


# # Run this to estimate how long training will take for current parameters
# num_test_batches = 20  # Number of batches to run for the test
# start_time = time.time()
#
# model.train()
# batch_count = 0d
#
# for epoch in range(1):  # Only one epoch for the test
#     for batch in train_loader:
#         if batch_count >= num_test_batches:
#             break
#
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#
#         optimizer.zero_grad()
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#
#         batch_count += 1
#
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time for {num_test_batches} batches: {elapsed_time} seconds")
#
# num_batches_per_epoch = len(train_loader)
# total_batches = num_batches_per_epoch * epochs
#
# estimated_total_time = (elapsed_time / num_test_batches) * total_batches
# print(f"Estimated total training time: {estimated_total_time / 3600:.2f} hours")


