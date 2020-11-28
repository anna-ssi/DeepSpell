import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.encoder import Encoder


def collate_fn(batch):
    encoder = [item[0] for item in batch]
    decoder = [item[1] for item in batch]
    target = [item[2] for item in batch]
    return torch.tensor(encoder), torch.tensor(encoder), torch.tensor(target)


def train(train_db, batch_size, device=None, lr=3e-4):
    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )

    model = Encoder(batch_size, 30, 50, 3, 0.2)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    total_loss = 0
    for batch in train_loader:
        encoder, decoder, target = batch
        print(encoder.shape, decoder.shape, target.shape)

        optimizer.zero_grad()

        predictions = model(encoder.to(device))
        # loss = criterion(predictions, target.to(device))
        break

        # loss.backward()
        # optimizer.step()

        # total_loss += loss.item()

    return total_loss
