import numpy as np
from dataset import TextDataset
from utils import *

char2id = create_chars_dict(CHARS, char2id)


def read_data(path):
    file = open(path, 'r')
    return file.read()


def iterator(data, length=1):
    data_len = len(data)
    gen = 0
    while data_len > gen:
        yield data[gen: gen + length]
        gen += length


def preprocess(data):
    data = tokenize(data)
    data = [token.strip().lower() for token in data.split()]
    tokens = []

    for token in iterator(data):
        err_rate = np.random.random()
        err_token = add_spelling_error(token, err_rate)
        token, err_token = token_to_number(token, char2id), token_to_number(err_token, char2id)
        tokens.append((token, err_token))

    return tokens


if __name__ == '__main__':
    file_path = './data/siddhartha.txt'
    text = read_data(file_path)
    text = preprocess(text)

    td = TextDataset(text)
    print(td[10])
