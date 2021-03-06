from time import sleep
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl


class DummyDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.rand(10)


class CoolSystem(pl.LightningModule):
    def __init__(self):
        super(CoolSystem, self).__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_nb):
        sleep(1)
        return {'loss': torch.mean(self.forward(batch) ** 2)}

    def validation_step(self, batch, batch_nb):
        sleep(1)
        return {}

    def validation_end(self, outputs):
        return {}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.layer.parameters())]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(DummyDataset(10), batch_size=1)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(DummyDataset(5), batch_size=1)

model = CoolSystem()
trainer = pl.Trainer(weights_summary=None, nb_sanity_val_steps=0, early_stop_callback=False,
                     val_percent_check=1.0, val_check_interval=0.5)
trainer.fit(model)
