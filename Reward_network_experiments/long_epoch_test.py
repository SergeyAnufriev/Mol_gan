from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import no_grad,float32, device,save,manual_seed, cuda
import torch.optim as optim
import numpy as np
import os
import random
from utils import train_test
from misc.Diagnostic import plot_real_vs_predicted
from sklearn.metrics import r2_score

'''fix random seed'''
seed = 42
manual_seed(seed)
np.random.seed(seed)
#backends.cudnn.deterministic = True
random.seed(seed)

cuda   = device('cuda' if cuda.is_available() else 'cpu')

'''default hyperparametrs used to initiate hyperparameters grid search'''

hyperparameter_defaults = dict(
    h1 = 128,
    h2 = 64,
    learning_rate = 0.001,
    bz =64,
    epochs = 20,
    drop_out=0.1)

wandb.init(config=hyperparameter_defaults, project="ml_test2") #####
config = wandb.config

'''model configuration parameters listed below '''

b_size = config.bz
n_node_features = 5
h1 = config.h1
h2 = config.h2
h3 = config.h3
h4 = config.h4
lr = config.learning_rate
dr = config.drop_out
epochs = config.epochs

'''QM9 dataset is used to train reward network model'''

datka = Mol_dataset(r'/data/gdb9_clean.sdf')
criterion = nn.MSELoss()

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


train_d, test_d = train_test(datka,b_size)
model = R(n_node_features,h1,h2,h3,h4,dr,cuda)
model.to(cuda)
optimizer = optim.Adam(model.parameters(), lr=lr)

'''Training loop '''

def main():
    for epoch in range(epochs):
        for idx,(A,X,r) in enumerate(train_d):
            r = r.to(float32)
            optimizer.zero_grad()
            outputs = model(A.to(cuda),X.to(cuda))
            loss = criterion(outputs, r.to(cuda))
            wandb.log({'Train_loss':loss})
            loss.backward()
            optimizer.step()
        plot_real_vs_predicted(model,A.to(cuda),X.to(cuda),r.to(cuda))

    PATH = os.path.join(wandb.run.dir, str(epoch)+'_'+"model.pt")
    save(model, PATH)
    wandb.save(PATH)
    test(model,test_d)


if __name__ == '__main__':
   main()
