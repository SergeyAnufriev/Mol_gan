from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import float32,device,manual_seed, cuda
import torch.optim as optim
import numpy as np
import random
from misc.Diagnostic import plot_real_vs_predicted,train_info
from utils import train_test,sweep_to_dict,test


dir_config  = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\ML_TEST\cycle_lr.yaml'
dir_dataset = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'

wandb.init(config=sweep_to_dict(dir_config))
config = wandb.config
n_node_features = 5

'''Fix random seed'''

manual_seed(config.seed)
np.random.seed(config.seed)
#backends.cudnn.deterministic = True
random.seed(config.seed)

'''Initialise model'''

cuda            = device('cuda' if cuda.is_available() else 'cpu')
train_d, test_d = train_test(Mol_dataset(dir_dataset),config.bz)
model           = R(n_node_features,config.h1,config.h2,config.h3,config.h4,config.drop_out,cuda)
model.to(cuda)
criterion = nn.MSELoss()

activation = {}

def get_activation(name):
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def hook(model):
  for item in list(model._modules.items()):
    item[1].register_forward_hook(get_activation(item[0]))

def save_act(activation):
  for key, value in activation.items():
    if 'act' in key:
      wandb.log({key: wandb.Histogram(value.mean(axis=0).flatten().cpu().detach().numpy())})

hook(model)


optimizer = optim.Adam(model.parameters(),lr=config.lr_min)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.lr_min, max_lr=config.lr_max,step_size_up=len(train_d)*config.step_size,mode="triangular",cycle_momentum=False)

train_loss = 0
for _ in range(config.epochs):
    for idx,(A,X,r) in enumerate(train_d):
        r = r.to(float32)
        optimizer.zero_grad()
        outputs = model(A.to(cuda),X.to(cuda))
        loss = criterion(outputs, r.to(cuda))
        loss.backward()
        optimizer.step()
        wandb.log({'lr':optimizer.param_groups[0]["lr"]})
        scheduler.step()
        wandb.log({'Train_Loss':train_loss})
        if idx %10 ==0:
            save_act(activation)
            train_info(model)

    test(model,test_d,cuda)
    plot_real_vs_predicted(model,A.to(cuda),X.to(cuda),r.to(cuda))



