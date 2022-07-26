import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from model import AutoEncoder
from utils import Trainer


def main():
    dataset = MNIST(root='./', train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset=dataset, shuffle=True, num_workers=2, batch_size=128)

    params = {
        'ni': [28 * 28],
        'nf': [32, 64],
        'nz': [2],
        'activation': [nn.ReLU],
        'batch_norm': [True, False]
    }
    trainer = Trainer()
    trainer.grid_fit(AutoEncoder, params, dataloader, 0, 4, './', -1, 'bar')


if __name__ == '__main__':
    main()
