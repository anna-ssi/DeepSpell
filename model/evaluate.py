import torch


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for encoder, decoder, target in iterator:
            output = model(encoder, decoder)
            loss = criterion(output.values, target)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
