from model.train import train
from model.evaluate import evaluate
from spelling.dataset import *
from spelling.utils import *

from model.decoder import Decoder
from model.encoder import Encoder
from model.seq2seq import Seq2Seq
from spelling.utils import CHARS
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import time

file_path = './data/siddhartha.txt'
text = read_data(file_path)
td = TextDataset(text)
train_data, valid_data, test_data = split_data(td)

INPUT_DIM = len(CHARS)
OUTPUT_DIM = len(CHARS)
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HID_DIM = 216
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 16

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, None)
optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss()

train_loader = loader(train_data, BATCH_SIZE)
valid_loader = loader(valid_data, BATCH_SIZE)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, _ = train(model, train_loader, optimizer, criterion, CLIP)

    print(train_loss, _)
    exit()

    valid_loss = evaluate(model, valid_loader, criterion)

    print(train_loss, valid_loss)
    exit()

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'logs/tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
