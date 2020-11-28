from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]
