import re

import numpy as np

REMOVE_CHARS = '[’#$%"\+@<=>!&,-.?’:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'
CHARS = list("abcdefghijklmnopqrstuvwxyz'")
SOS = '<SOS>'  # start of sequence.
EOS = '<EOS>'  # end of sequence.

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


def token_to_number(token, char_dict):
    token_num = [char_dict[i] for i in token[0]]
    token = [char_dict[SOS]] + token_num + [char_dict[EOS]]
    return token
