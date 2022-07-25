import os
from typing import Optional, Union, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def fit(self, dataloader, device, epoch, save_path, save_epoch, verbose: Optional[str] = None):
        raise NotImplementedError

    def step(self, batch):
        raise NotImplementedError

    def inference_step(self, batch):
        raise NotImplementedError

    def _configure_optimizer(self):
        raise NotImplementedError

    def _configure_criterion(self):
        raise NotImplementedError


class AutoEncoder(CustomModel):
    def __init__(self, ni, nf, nz, activation, batch_norm):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            *self._block(ni, nf, activation, batch_norm),
            *self._block(nf, nf // 2, activation, batch_norm),
            *self._block(nf // 2, nf // 4, activation, batch_norm),
            *self._block(nf // 4, nf // 8, activation, batch_norm),
            *self._block(nf // 8, nz)
        )
        self.decoder = nn.Sequential(
            *self._block(nz, nf // 8, activation, batch_norm),
            *self._block(nf // 8, nf // 4, activation, batch_norm),
            *self._block(nf // 4, nf // 2, activation, batch_norm),
            *self._block(nf // 2, nf, activation, batch_norm),
            *self._block(nf, ni)
        )
        self.criterion = self._configure_criterion()
        self.optim = self._configure_optimizer()

    def _block(self, ni, no, activation=None, batch_norm=False, dropout=0):
        ret = [nn.Linear(ni, no)]
        if activation:
            ret.append(activation())
        if batch_norm:
            ret.append(nn.BatchNorm1d(no))
        if dropout != 0:
            ret.append(nn.Dropout())
        return ret

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        x = x.flatten(1)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(-1, 1, 28, 28)
        return out, z

    def fit(self, dataloader: DataLoader, device: Union[int, str], epoch: int, save_path: str,
            save_epoch: Union[int, List[int]],
            verbose: Optional[str] = None):
        if verbose == 'bar':
            pbar = tqdm(range(1, epoch + 1))
        else:
            pbar = range(1, epoch + 1)
        if device < 0:
            device = 'cpu'
        else:
            device = f"cuda:{device}" if torch.cuda.is_available() else 'cpu'
        for e in pbar:
            self.train()
            self.to(device)
            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader, 1):
                batch, label = batch
                batch = batch.to(device)
                loss = self.step(batch)
                epoch_loss += loss
                if verbose == 'bar':
                    pbar.set_postfix_str(f"epoch {e} of {epoch}, loss: {epoch_loss / batch_idx:.5f}")
                if verbose == 'str' and ((batch_idx == len(dataloader)) or (batch_idx % (len(dataloader) // 10) == 0)):
                    print(f"epoch {e} of {epoch}, loss: {epoch_loss / batch_idx:.5f}")
            pbar.update()
            if isinstance(save_epoch, int):
                if e % save_epoch == 0:
                    torch.save(self.state_dict(), os.path.join(save_path, f"epoch={e}{os.extsep}pth"))
            elif isinstance(save_epoch, List):
                if e in save_epoch:
                    torch.save(self.state_dict(), os.path.join(save_path, f"epoch={e}{os.extsep}pth"))
            else:
                raise NotImplementedError(f"Not Implemented for type {epoch.__class__}")
        torch.save(self.state_dict(), os.path.join(save_path, f"epoch={epoch}{os.extsep}pth"))

    def step(self, batch: torch.Tensor) -> float:
        output, _ = self(batch)
        loss = self.criterion(output, batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def inference_step(self, batch: torch.Tensor) -> torch.Tensor:
        output, z = self(batch)
        loss = torch.pow(batch - output, 2)
        loss = torch.sqrt(loss)
        loss = torch.mean(loss, dim=1)
        assert loss.shape[0] == batch.shape[0]
        return loss

    def _configure_optimizer(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.RAdam(self.parameters())
        return optimizer

    def _configure_criterion(self):
        criterion = torch.nn.MSELoss()
        return criterion
