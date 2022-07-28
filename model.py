import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

    def forward(self, x):
        """

        Args:
            x: batch

        Returns:
            basic model forward result
        """
        raise NotImplementedError

    def step(self, batch, device) -> float:
        """

        Args:
            batch:
            device: purposed device to accelerate

        Returns:
            basic model step
        """
        raise NotImplementedError

    def inference_step(self, batch):
        """

        Args:
            batch:

        Returns:
            model prediction that has same length of batch
        """
        raise NotImplementedError

    def _configure_optimizer(self):
        """

        Returns:
            optimizer that model uses
        """
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
        self.criterion = torch.nn.MSELoss()
        self.optim = self._configure_optimizer()

    # noinspection PyMethodMayBeStatic
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

    def step(self, batch: torch.Tensor, device: str) -> float:
        batch = batch.to(device)
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


class LinearClassifier(CustomModel):
    def __init__(self, num_input, num_hidden, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_classes),
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = self._configure_optimizer()

    def forward(self, x):
        output = self.linear(x)
        return output

    def step(self, batch, device):
        data, label = batch
        data = data.to(device)
        label = label.to(device)
        output = self(data)
        loss = self.criterion(output, label)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def inference_step(self, batch):
        output = self(batch)
        pred = torch.argmax(output, dim=1)
        assert batch.shape[0] == pred.shape[0]
        return pred

    def _configure_optimizer(self):
        optimizer = torch.optim.RAdam(self.parameters())
        return optimizer
