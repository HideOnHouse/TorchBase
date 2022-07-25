import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from model import AutoEncoder


def main():
    dataset = MNIST(root='./', train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset=dataset, shuffle=True, num_workers=2, batch_size=128)
    model = AutoEncoder(28 * 28, 128, 4, nn.CELU, batch_norm=True)
    model.fit(dataloader, 0, 16, save_path='./', save_epoch=[2, 4], verbose='bar')


if __name__ == '__main__':
    main()
