import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return torch.tensor(data), torch.tensor(target)
