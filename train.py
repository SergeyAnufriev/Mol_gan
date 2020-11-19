from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import cat,no_grad,float32,flatten
from torch.utils.data import DataLoader,random_split
import torch.optim as optim


hyperparameter_defaults = dict(
    h1 = 16,
    h2 = 16,
    learning_rate = 0.0001,
    bz =64)

wandb.init(config=hyperparameter_defaults, project="rew_test") #####
config = wandb.config


def train_test(dataset,b_size):
  x = int(len(dataset)*0.8)
  y = len(dataset) - x
  train_dataset, test_dataset = random_split(dataset, [x, y])
  train_d, test_d = DataLoader(train_dataset,b_size,True,drop_last=True),\
                  DataLoader(test_dataset,b_size,True,drop_last=True)
  return train_d, test_d


criterion = nn.MSELoss()

def test(model, test_loader): #### over full test dataset average loss
  model.eval()
  test_loss = 0
  with no_grad():
    for a,x,r in test_loader:
      outputs    = model(a,x)
      test_loss += criterion(outputs, r)
    wandb.log({'Test_Loss':test_loss/(len(test_loader.dataset)/test_loader.batch_size)})


def grad_info(model):
  total_grad = []
  for name, param in model.named_parameters():
    grad = list(flatten(param.grad).numpy())
    wandb.log({name: wandb.Histogram(grad)})
    total_grad +=grad


b_size = config.bz
n_node_features = 5
h1 = config.h1
h2 = config.h2
h3 = 128
lr = config.learning_rate
datka = Mol_dataset('/content/gdb9_clean.sdf')


#train_d, test_d = train_test(dataset,b_size)
train_d, test_d = train_test(datka,b_size)
r_net = R(n_node_features,h1,h2,h3)
optimizer = optim.Adam(r_net.parameters(), lr=lr)


def main():
    train_loss = 0
    for idx,(A,X,r) in enumerate(train_d):
        r = r.to(float32)
        optimizer.zero_grad()
        outputs = r_net(A,X)
        loss = criterion(outputs, r)
        loss.backward()
        optimizer.step()
        train_loss += loss
        if idx %10 ==0:
            test(r_net,test_d)  ### Evaluate mean test loss on all batches in test dataset
            wandb.log({'Train_Loss':train_loss/10})  ### Average train loss on 10 last betches
            grad_info(r_net) ### Get Weights gradient info and total grad L2 norm
            train_loss = 0
        if idx == 500:
            break

if __name__ == '__main__':
   main()
