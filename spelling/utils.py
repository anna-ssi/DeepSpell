import re

import numpy as np
from torch.utils.data import random_split

REMOVE_CHARS = '[’#$%"\+@<=>!&,-.?’:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'
CHARS = list("abcdefghijklmnopqrstuvwxyz'")
SOS = '\t'  # start of sequence.
EOS = '*'  # end of sequence.

char2id = {
    SOS: 0,
    EOS: 1,
}


def tokenize(data):
    return re.sub(REMOVE_CHARS, ' ', data)


def replace(token):
    random_char_index = np.random.randint(len(token))
    token = token[:random_char_index] + np.random.choice(CHARS) + token[random_char_index + 1:]
    return token


def delete(token):
    random_char_index = np.random.randint(len(token))
    token = token[:random_char_index] + token[random_char_index + 1:]
    return token


def add(token):
    random_char_index = np.random.randint(len(token))

    token = token[:random_char_index + 1] + np.random.choice(CHARS) + token[random_char_index + 1:]
    return token


def transpose(token):
    random_char_index = np.random.randint(len(token) - 1)
    token = token[:random_char_index] + token[random_char_index + 1] + token[random_char_index] + \
            token[random_char_index + 2:]
    return token


def add_spelling_error(data, error_rate):
    assert (0.0 <= error_rate < 1.0)

    rand = np.random.rand()
    prob = error_rate / 4.0
    tokens = []

    for token in data:
        if len(token) < 3:
            pass

        elif rand < prob:
            token = replace(token)

        elif prob < rand < prob * 2:
            token = delete(token)

        elif prob * 2 < rand < prob * 3:
            token = add(token)

        elif prob * 3 < rand < prob * 4:
            token = transpose(token)
        else:
            pass

        tokens.append(token)

    return tokens


def create_chars_dict(chars, pre_dict=None):
    if pre_dict is None:
        pre_dict = {}
        last_id = 0
    else:
        last_id = list(pre_dict.values())[-1] + 1

    for char in chars:
        pre_dict[char] = last_id
        last_id += 1

    return pre_dict


def token_to_number(token, char_dict, max_len):
    token = list(token[0])
    token = add_up_to_max_len(token, max_len)
    token = [char_dict[i] for i in token]
    return token


def read_data(path):
    file = open(path, 'r')
    return file.read()


def iterator(data, length=1):
    data_len = len(data)
    gen = 0
    while data_len > gen:
        yield data[gen: gen + length]
        gen += length


def get_max_token_length(data):
    return len(max(data, key=lambda i: len(i)))


def add_up_to_max_len(token, max_len):
    token = [SOS] + token + [EOS] * (max_len - len(token) - 1)
    return token


def transform(token, max_len, err_rate=np.random.random()):
    err_token = add_spelling_error(token, err_rate)
    encoder = token_to_number(err_token, char2id, max_len)
    decoder = token_to_number(token, char2id, max_len)
    target = decoder[1:] + [char2id[EOS]]
    return encoder, decoder, target


def preprocess(data):
    data = tokenize(data)
    data = [token.strip().lower() for token in data.split()]
    max_len_token = get_max_token_length(data)
    tokens = []

    for token in data:
        tokens.append(transform(token, max_len_token))

    return tokens


def split_data(data, ratio=(7, 2, 1)):
    assert sum(ratio) == 10

    length = len(data)
    ratio = [round((length * (i * 10)) / 100) for i in ratio]
    return random_split(data, ratio)


char2id = create_chars_dict(CHARS, char2id)
