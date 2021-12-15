import torch
from torch.utils.data import DataLoader

from dataset import CustomDataset
from model import CustomModel
from test import test
from train import train
from inference import inference


def main():
    model = CustomModel()
    # train
    train_dataset = CustomDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    history = train(model, device, optimizer, criterion, 16, train_dataloader)

    # Test
    test_dataset = CustomDataset()
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    test(model, device, test_dataloader, criterion)

    # Inference
    infer_dataset = CustomDataset()
    infer_dataloader = DataLoader(infer_dataset, batch_size=1024, shuffle=False)
    inference(model, device, infer_dataloader)


if __name__ == '__main__':
    main()
