from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import cat,no_grad,float32,flatten,device,save,manual_seed,backends,zeros,tensor
from torch.utils.data import Subset
import torch.optim as optim
import numpy as np
import os
import random
from sklearn.metrics import r2_score,mean_absolute_error
import plotly.graph_objects as go

'''
 this script runs overfitting experiment 
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
    n_iter = 5,
    drop_out=0.1,
    dataset=0)

wandb.init(config=hyperparameter_defaults, project="overfit_test") #####
config = wandb.config
PATH   = os.path.join(wandb.run.dir, "model.pt")

criterion = nn.MSELoss()

'''Once the network overfitted a batch of data,
the plot illustrates how predictions vs actual for this batch of data '''

def plot_real_vs_predicted(model,A,X,r):
    model.eval()
    y_pred = model(A,X)
    y_pred = y_pred.detach().cpu().numpy()
    r      = r.detach().cpu().numpy()
    r2  = r2_score(r,y_pred)
    mae = mean_absolute_error(r,y_pred)
    wandb.log({'R2':r2})
    wandb.log({'MAE':mae})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.squeeze(-1).tolist(), y=y_pred.squeeze(-1).tolist(),
                    mode='markers',
                    name='predictions'))
    fig.add_trace(go.Scatter(x=np.linspace(0,0.5,100).tolist(), y=np.linspace(0,0.5,100).tolist(),
                    mode='lines',
                    name='x=y,ideal'))
    fig.update_layout(
       autosize=False,
       width=600,
       height=600,
       xaxis_title="Real",
       yaxis_title="Predicted")
    fig.add_annotation(x=0.4, y=0.0,
            text="R2"+'=' + str(r2),
            showarrow=False,
            yshift=10)
    fig.add_annotation(x=0.4, y=0.05,
            text="MAE"+'=' + str(mae),
            showarrow=False,
            yshift=10)
    wandb.log({'Real_vs_Predicted':fig})

b_size     = config.bz
n_node_features = 5
h1 = config.h1
h2 = config.h2
h3 = 128
h4 = 64
lr = config.learning_rate
dr = config.drop_out

'''QM9 dataset is used to train reward network model'''

datka      = Mol_dataset('/content/gdb9_clean.sdf')

'''data.npy datapoint indecies were created by '''
'''np.random.randint(0,dataset length,size=(number of batches, batch size))'''

all_index  = np.load('/content/data.npy')
idecies    = all_index[config.dataset,:]

A = zeros(b_size,9,9,5)
X = zeros(b_size,9,5)
r = zeros(b_size,1)


'''This code asssebles A,X,r batches of adjecency tensor,node feature matrix and rewards'''
for idx,i in  enumerate(idecies):
    a_,x_,r_ = datka.__getitem__(i)
    A[idx,:,:,:] = a_
    X[idx,:,:]   = x_
    r[idx,:]     = tensor(r,dtype=float32)


'''n_iter the number of iterations network goes over the same batch'''

n_iter = config.n_iter

r_net = R(n_node_features,h1,h2,h3,h4,dr,cuda)
r_net.to(cuda)
optimizer = optim.Adam(r_net.parameters(), lr=lr)

'''5 batches of data tested for overfitting, since result for one might not be representative for the whole network'''

for _ in range(n_iter):
    optimizer.zero_grad()
    outputs = r_net(A.to(cuda),X.to(cuda))
    loss = criterion(outputs, r.to(cuda))
    wandb.log({'MSE':loss})
    loss.backward()
    optimizer.step()
plot_real_vs_predicted(r_net,A.to(cuda),X.to(cuda),r)








