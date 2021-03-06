import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import apex.amp as amp

es = EarlyStopping()
es.monitor_op == torch.lt

model = torch.Linear(5, 5)
optimizers = torch.optim.Adam(model.parameters(), lr=1e-3)
amp.initialize(model, optimizers)

es.monitor_op == torch.lt
