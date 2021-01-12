from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import float32,device,manual_seed,backends,cuda,save,no_grad
import torch.optim as optim
import numpy as np
import random
from Diagnostic import plot_real_vs_predicted,train_info
from utils import train_test
from sklearn.metrics import r2_score
import yaml

'''fix random seed'''
seed = 42
manual_seed(seed)
np.random.seed(seed)
#backends.cudnn.deterministic = True
random.seed(seed)

cuda = device('cuda' if cuda.is_available() else 'cpu')

'''Experiment set up'''
'''Lr rates ordered by backprop order'''

with open(r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\ML_TEST\single_run.yaml') as file:
    params = yaml.safe_load(file)

wandb.init(config=params,project="ML_TEST") #####
config = wandb.config

n_node_features = 5
h1     = config.h1
h2     = config.h2
h3     = config.h3
h4     = config.h4
b_size = config.bz
dr     = config.drop_out
epoch  = config.epochs

train_d, test_d = train_test(Mol_dataset(r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'),b_size)
model           = R(n_node_features,h1,h2,h3,h4,dr,cuda)
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


def test(model, test_loader): #### over full test dataset average loss
  model.eval()
  test_loss = 0
  y_pred = []
  y_real = []
  with no_grad():
    for a,x,r in test_loader:
      outputs    = model(a.to(cuda),x.to(cuda))
      test_loss += criterion(outputs, r.to(cuda))
      y_pred.append(outputs.data.cpu().numpy())
      y_real.append(r)
  y_pred = np.vstack(y_pred)
  y_real = np.vstack(y_real)

  wandb.log({'Test_Loss':test_loss/(len(test_loader.dataset)/test_loader.batch_size)})
  wandb.log({'R2_TEST':r2_score(y_real,y_pred)})


'''Discriminative learning rates'''

lr_layers =['lin5','lin4', 'lin3','agr','conv2','conv1']
lr_rates = [{'params':getattr(model,x).parameters(),'lr':y } for (x,y) in zip(lr_layers,config.lr_rates)]
optimizer = optim.Adam(lr_rates)


train_loss = 0
for _ in range(epoch):
    for idx,(A,X,r) in enumerate(train_d):
        r = r.to(float32)
        optimizer.zero_grad()
        outputs = model(A.to(cuda),X.to(cuda))
        loss = criterion(outputs, r.to(cuda))
        loss.backward()
        optimizer.step()
        train_loss += loss
        if idx %10 ==0:
            wandb.log({'Train_Loss':train_loss/10})
            train_loss = 0
            save_act(activation)
            train_info(model)

    test(model,test_d)
    plot_real_vs_predicted(model,A.to(cuda),X.to(cuda),r.to(cuda))


