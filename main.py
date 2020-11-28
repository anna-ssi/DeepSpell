from model.train import train
from spelling.dataset import *
from spelling.utils import *

if __name__ == '__main__':
    file_path = './data/siddhartha.txt'
    text = read_data(file_path)
    td = TextDataset(text)
    train_data, valid_data, test_data = split_data(td)
    train(train_data, 32)
