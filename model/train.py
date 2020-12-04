import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    count = 0
    for encoder, decoder, target in iterator:
        # print(encoder.shape, decoder.shape, target.shape)
        optimizer.zero_grad()
        count += 1

        output = model(encoder, decoder)

        loss = criterion(output.values, target)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), count
