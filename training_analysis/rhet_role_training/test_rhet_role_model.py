import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from dataset import RhetRoleDataset


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

    print("Classification Report:")
    print(classification_report(true_labels, predictions, labels=[0, 1, 2, 3, 4, 5, 6]))

    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)


model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert', num_labels=7)
model_save_path = './model/legalbert_finetuned.pth'
model.load_state_dict(torch.load(model_save_path))

test_dataset = RhetRoleDataset('./data/test_dataset.csv', 128, 'bert-base-uncased')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_model(model, test_loader, device)


