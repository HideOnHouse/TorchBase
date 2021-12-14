import torch

from tqdm import tqdm


def inference(model, device, inference_loader):
    model.eval()
    with torch.no_grad():
        for data in tqdm(inference_loader):
            data = data.to(device)
            output = model(data)


def main():
    pass


if __name__ == '__main__':
    main()
