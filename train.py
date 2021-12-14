import sys

import torch
from tqdm import tqdm


def train(model, device, train_loader, valid_loader, optimizer, criterion, epoch) -> dict:
    """
    returns history dictionary that contains train_loss, valid_loss as list
    """
    history = {
        'train_loss': [],
        'valid_loss': [],
    }
    for e in range(epoch):
        model.train()
        train_loss = 0
        pbar = tqdm(enumerate(train_loader), file=sys.stdout)
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device),
            output = model(data)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(epoch=f'epoch {e + 1} of {epoch}', loss=f'')
        pbar.close()

        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device),
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
        history['valid_loss'].append(valid_loss)
    return history


def main():
    pass


if __name__ == '__main__':
    main()
