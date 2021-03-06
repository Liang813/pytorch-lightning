import torch
import numpy as np
import apex.amp as amp
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
torch_inf = torch.tensor(np.Inf)
es = EarlyStopping(mode='min')


torch_inf if es.monitor_op == torch.lt else -torch_inf 

-torch_inf if es.monitor_op(torch.Tensor(1), torch.Tensor(2))[0].item() else torch_inf

es = EarlyStopping(mode='max')
torch_inf if es.monitor_op == torch.lt else -torch_inf 

-torch_inf if es.monitor_op(torch.Tensor(1), torch.Tensor(2))[0].item() else torch_inf


es = EarlyStopping(mode='min')
model = torch.nn.Linear(5, 5)
optimizers = torch.optim.Adam(model.parameters(), lr=1e-3)
amp.initialize(model, optimizers)
torch_inf if es.monitor_op == torch.gt else -torch_inf


# NOTE!! The above is incorrect, and is what this patch fixed

-torch_inf if es.monitor_op(torch.Tensor(1), torch.Tensor(2))[0].item() else torch_inf
