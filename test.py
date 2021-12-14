import sys

import torch
from tqdm import tqdm


def test(model, device, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader, file=sys.stdout):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target)


def main():
    pass


if __name__ == '__main__':
    main()
