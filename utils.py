import os
from itertools import product
from typing import Union, Optional, List, Iterable, Any, Type

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import CustomModel


class Trainer:
    def __init__(self):
        self.train_history = {
            'epoch_loss': []
        }

    def fit(self,
            model: CustomModel,
            dataloader: DataLoader,
            device: Union[int, str],
            epoch: int,
            save_path: str,
            save_epoch: Union[int, Iterable[int]],
            verbose: Optional[str] = None) -> None:
        if os.path.exists(save_path):
            print(f"Warning. Path {save_path} already exists")
            cmd = ''
            while cmd not in ['y', 'n']:
                cmd = input("Do yoy want to override? (y/n)")
                if cmd != 'y':
                    raise FileExistsError
        os.makedirs(save_path, exist_ok=True)

        if verbose == 'bar':
            pbar = tqdm(range(1, epoch + 1))
        else:
            pbar = range(1, epoch + 1)
        if device < 0:
            device = 'cpu'
        else:
            device = f"cuda:{device}" if torch.cuda.is_available() else 'cpu'

        for e in pbar:
            model.train()
            model.to(device)
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader, 1):
                batch, label = batch
                batch = batch.to(device)
                loss = model.step(batch)
                epoch_loss += loss
                if verbose == 'bar':
                    pbar.set_postfix_str(f"epoch {e} of {epoch}, loss: {epoch_loss / batch_idx:.5f}")
                if verbose == 'str' and ((batch_idx == len(dataloader)) or (batch_idx % (len(dataloader) // 10) == 0)):
                    print(f"epoch {e} of {epoch}, loss: {epoch_loss / batch_idx:.5f}")
            if verbose == 'bar':
                pbar.update()
            if isinstance(save_epoch, int):
                if e % save_epoch == 0:
                    torch.save(model.state_dict(), os.path.join(save_path, f"epoch={e}{os.extsep}pth"))
            elif isinstance(save_epoch, List):
                if e in save_epoch:
                    torch.save(model.state_dict(), os.path.join(save_path, f"epoch={e}{os.extsep}pth"))
            else:
                raise NotImplementedError(f"Not Implemented for type {type(epoch)}")
        torch.save(model.state_dict(), os.path.join(save_path, f"epoch={epoch}{os.extsep}pth"))

    def grid_fit(self,
                 model_class: Type,
                 parameters: dict[str, Iterable[Any]],
                 dataloader: DataLoader,
                 device: Union[int, str],
                 epoch: int,
                 base_path: str,
                 save_epoch: Union[int, Iterable[int]],
                 verbose: Optional[str] = None):

        for params in product(*parameters.values()):
            kwarg = dict(zip(parameters.keys(), params))
            prefix = ""
            for k, v in kwarg.items():
                if 'activation' in str(v):
                    v = str(v).split('.')[-1][:-2]
                prefix += f"{k}={v}_"
            prefix = prefix[:-1]
            save_path = os.path.join(base_path, prefix)
            model = model_class(**kwarg)
            self.fit(model, dataloader, device, epoch, save_path, save_epoch, verbose)
            print(f"model {type(model)} with parameter {kwarg} is done.")
