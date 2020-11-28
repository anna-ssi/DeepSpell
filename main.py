from model.train import train
from spelling.dataset import *
from spelling.utils import *
if __name__ == '__main__':
    file_path = './data/siddhartha.txt'
    text = read_data(file_path)
    text = preprocess(text)
    td = TextDataset(text)
    train_data, valid_data, test_data = split_data(td)
    # train(td, 32)
