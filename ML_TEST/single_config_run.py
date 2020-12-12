from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import cat,no_grad,float32,flatten,device,save,manual_seed,backends,zeros,tensor,cuda
import torch.optim as optim
import numpy as np
import random
from Diagnostic import plot_real_vs_predicted,train_info,test
from utils import train_test

'''fix random seed'''
seed = 42
manual_seed(seed)
np.random.seed(seed)
backends.cudnn.deterministic = True
random.seed(seed)

cuda = device('cuda' if cuda.is_available() else 'cpu')

'''Experiment set up'''

params = dict(
    h1 = 128,
    h2 = 64,
    h3 = 128,
    h4 = 64,
    learning_rate = 0.001,
    bz =64,
    epoch = 5,
    drop_out=0.1,
    )

wandb.init(config=params,project="ML_TEST") #####
config = wandb.config

n_node_features = 5
h1     = config.h1
h2     = config.h2
h3     = config.h3
h4     = config.h4
b_size = config.bz
lr     = config.learning_rate
dr     = config.drop_out
epoch  = config.epoch

train_d, test_d = train_test(Mol_dataset('/content/gdb9_clean.sdf'),b_size)
model           = R(n_node_features,h1,h2,h3,h4,dr,cuda)
model.to(cuda)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

'''5 batches of data tested for overfitting, since result for one might not be representative for the whole network'''

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

train_loss = 0
for _ in range(epoch):
    for idx,(A,X,r) in train_d:
        A.to(cuda)
        X.to(cuda)
        r.to(cuda)
        r = r.to(float32)
        optimizer.zero_grad()
        outputs = model(A,X)
        loss = criterion(outputs, r)
        loss.backward()
        optimizer.step()
        train_loss += loss
        if idx %10 ==0:
            wandb.log({'Train_Loss':train_loss/10})
            train_loss = 0
            save_act(activation)
            train_info(model)
            test(model,A,X,r)
    plot_real_vs_predicted(model,A,X,r)


