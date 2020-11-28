# from torch.nn import CrossEntropyLoss
# from torch.optim import Adam
from torch.utils.data import DataLoader


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return data, target


def train(train_db, batch_size, device=None, lr=3e-4):
    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    # model = RewardPredictor(*train_db.sizes)
    # criterion = CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=lr)

    total_loss = 0
    for batch in train_loader:
        data, target = batch
        print(data, target)
        break

        # optimizer.zero_grad()
        #
        # predictions = model(data.to(device))
        # loss = criterion(predictions, target.to(device))
        # loss.backward()
        # optimizer.step()

        # total_loss += loss.item()

    return total_loss
