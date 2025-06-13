import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ArgRoleDataset(Dataset):
    def __init__(self, df, max_seq_length, tokenizer_name):
        self.max_len = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_map = {
            "NON-IRC": 0,
            "ISSUE": 1,
            "REASON": 2,
            "CONCLUSION": 3,
        }
        self.default_label_id = 0
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]

        text = data_row['text']
        label = data_row['arg_role']

        try:
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
        except ValueError as ve:
            print(f"Error processing data at index {idx}: {ve}")
            print(f"Offending text: {text}")
            raise ve

        label_id = self.label_map.get(label, self.default_label_id)
        label_id = torch.tensor(label_id)

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': label_id
        }
