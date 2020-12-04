from torch.utils.data import Dataset
from utils.functions import preprocess

CHARS = list("abcdefghijklmnopqrstuvwxyz'")


class TextDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = preprocess(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]
