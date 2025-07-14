import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from sklearn.model_selection import train_test_split
import random
import numpy as np
from typing import Tuple, Dict, Any

class NewsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 128
    ) -> None:
        self.data: pd.DataFrame = dataframe
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length: int = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text: str = row['title']
        label: int = row['label']

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids: torch.Tensor = encoding['input_ids'].squeeze(0)
        attention_mask: torch.Tensor = encoding['attention_mask'].squeeze(0)
        label_tensor: torch.Tensor = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_tensor
        }


class NewsDataLoader:
    def __init__(
        self,
        csv_file: str,
        test_size: float = 0.2,
        batch_size: int = 32,
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 128,
        seed: int = 42
    ) -> None:
        dataframe: pd.DataFrame = pd.read_csv(csv_file)

        if 'title' not in dataframe.columns or 'label' not in dataframe.columns:
            raise ValueError("CSV file must contain 'title' and 'label' columns.")

        self.set_seed(seed)

        train_data, test_data = train_test_split(dataframe, test_size=test_size, random_state=42)

        self.train_dataset: NewsDataset = NewsDataset(train_data, tokenizer_name, max_length)
        self.test_dataset: NewsDataset = NewsDataset(test_data, tokenizer_name, max_length)

        self.train_loader: DataLoader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader: DataLoader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        return self.train_loader, self.test_loader

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")
