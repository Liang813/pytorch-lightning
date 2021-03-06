import torch
import numpy as np
import os
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Make toy dataset
features = torch.from_numpy(np.asarray([[0],[0],[0],[1],[1],[1]])).float()
targets = torch.from_numpy(np.asarray([0,0,0,1,1,1]))
train = torch.utils.data.TensorDataset(features, targets)
train_loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)


#Define lightning model
class CoolSystem(pl.LightningModule):

    def __init__(self):
        super(CoolSystem, self).__init__()
        self.l1 = torch.nn.Linear(1, 10)
        self.l2 = torch.nn.Linear(10, 2)
        for param in self.l2.parameters():
            param.requires_grad = False
        self.loss_func = torch.nn.CrossEntropyLoss()
   
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        return train_loader

# Run the lightning model (check parameter before and after training)

coolsystem = CoolSystem()
print(list(coolsystem.parameters())[3])
trainer = pl.Trainer(min_epochs=10, max_epochs=10, logger=False)    
trainer.fit(coolsystem)
list(coolsystem.parameters())[3]
