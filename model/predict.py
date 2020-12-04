import torch
from utils.functions import *


def predict(model, word, char2id):
    characters = word.split()
    max_len = len(char2id) - 2
    token = token_to_number(characters, char2id, max_len)

    with torch.no_grad():
        output = model(token)

    return output
