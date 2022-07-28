import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from model import LinearClassifier
from utils import Trainer


def main():
    dataset = MNIST(root='./', train=True, transform=ToTensor(), download=True)
    dataloader = DataLoader(dataset=dataset, shuffle=True, num_workers=2, batch_size=128)
    model = LinearClassifier(28 * 28, 1024, 10)
    trainer = Trainer()
    trainer.fit(model, dataloader, 0, 2, './', [4, 8, 16], 'bar')

    total = 0
    correct = 0
    for data, label in dataloader:
        data, label = data.cuda(), label.cuda()
        pred = model.inference_step(data)
        for i in range(pred.shape[0]):
            if pred[i] == label[i]:
                correct += 1
            total += 1
    print(f"acc: {correct / total}")


if __name__ == '__main__':
    main()
