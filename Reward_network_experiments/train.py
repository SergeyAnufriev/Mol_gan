from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import cat,no_grad,float32,flatten,device,save,manual_seed,backends
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
import numpy as np
import os
import random



'''
 this script runs reward network training in for the graph regression task 
 reward network takes adjececy tensor and node feature matrix representing graph and outputs scalar value 
'''


'''fix random seed'''
seed = 42
manual_seed(seed)
np.random.seed(seed)
backends.cudnn.deterministic = True
random.seed(seed)



cuda   = device('cuda')

'''default hyperparametrs used to initiate hyperparameters grid search'''

hyperparameter_defaults = dict(
    h1 = 128,
    h2 = 64,
    learning_rate = 0.0001,
    bz =64,
    epochs = 5,
    drop_out=0.1)

wandb.init(config=hyperparameter_defaults, project="rew_test") #####
config = wandb.config
PATH   = os.path.join(wandb.run.dir, "model.pt")

'''split dataset into train and test parts'''

def train_test(dataset,b_size):
  x = int(len(dataset)*0.8)
  y = len(dataset) - x
  train_dataset, test_dataset = random_split(dataset, [x, y])
  train_d, test_d = DataLoader(train_dataset,b_size,True,drop_last=True),\
                  DataLoader(test_dataset,b_size,True,drop_last=True)
  return train_d, test_d


criterion = nn.MSELoss()

'''test function calculates mse error over the whole test part of data'''

def test(model, test_loader): #### over full test dataset average loss
  model.eval()
  test_loss = 0
  with no_grad():
    for a,x,r in test_loader:
      outputs    = model(a.to(cuda),x.to(cuda))
      test_loss += criterion(outputs, r.to(cuda))



    wandb.log({'Test_Loss':test_loss/(len(test_loader.dataset)/test_loader.batch_size)})

'''grad_info function records per train batch gradients for each model parameter'''

def grad_info(model):
  for name, param in model.named_parameters():
    grad = list(flatten(param.grad).cpu().numpy())
    wandb.log({name: wandb.Histogram(grad)})

'''model configuration parameters listed below '''

b_size = config.bz
n_node_features = 5
h1 = config.h1
h2 = config.h2
h3 = 128
h4 = 64
lr = config.learning_rate
dr = config.drop_out
epochs = config.epochs

'''QM9 dataset is used to train reward network model'''

datka = Mol_dataset('/content/gdb9_clean.sdf')


#train_d, test_d = train_test(dataset,b_size)
train_d, test_d = train_test(datka,b_size)
r_net = R(n_node_features,h1,h2,h3,h4,dr,cuda)
r_net.to(cuda)
optimizer = optim.Adam(r_net.parameters(), lr=lr)

'''Training loop '''

def main():
    train_loss = 0
    for _ in range(epochs):
        for idx,(A,X,r) in enumerate(train_d):
            r = r.to(float32)
            optimizer.zero_grad()
            outputs = r_net(A.to(cuda),X.to(cuda))
            loss = criterion(outputs, r.to(cuda))
            loss.backward()
            optimizer.step()
            train_loss += loss
            if idx %10 ==0:
                test(r_net,test_d)  ### Evaluate mean test loss on all batches in test dataset
                wandb.log({'Train_Loss':train_loss/10})  ### Average train loss on 10 last betches
                grad_info(r_net) ### Get Weights gradient info and total grad L2 norm
                train_loss = 0

    save(r_net, PATH)
    wandb.save(PATH)

if __name__ == '__main__':
   main()
