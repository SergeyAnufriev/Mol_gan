
import torch
from torch import autograd
import wandb
from scipy.sparse import linalg
import numpy as np
from torch.autograd import grad
from copy import deepcopy
from scipy.sparse.linalg import eigs,eigsh
import plotly.graph_objects as go
from sklearn.metrics import r2_score,mean_absolute_error
from matplotlib import pyplot as plt

'''This module purpose is to analyse neural network properties and interaction between them in case'''


activation = {}
def get_activation(name):
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def hook(model):
  for item in list(model._modules.items()):
    item[1].register_forward_hook(get_activation(item[0]))


'''Train information: weights, gradients, activations'''

def train_info(model):
  for name, param in model.named_parameters():

    '''weights histogram'''
    weight = list(torch.flatten(param.data).cpu().detach().numpy())
    wandb.log({name+'_'+'weight':wandb.Histogram(weight)})

    '''gradient histogram'''
    grad = list(torch.flatten(param.grad).cpu().detach().numpy())
    wandb.log({name+'_'+'grad': wandb.Histogram(grad)})


def test(model,test_loader,criterion): #### over full test dataset average loss
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for a,x,r in test_loader:
      outputs    = model(a,x)
      test_loss += criterion(outputs, r)


    wandb.log({'Test_Loss':test_loss/(len(test_loader.dataset)/test_loader.batch_size)})


  
'''Find the total number of model parameters'''
  
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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


'''Given vector V, shift model parameters along +-lambda*V
  ,where lambda scalar between -1 and 1'''
  
def model_params_shift(model,vec,lambd):
  
  # shift model params along vec by const lambd
  #model: pytorch module
  #vec  : np.array
  #lambda : scalar in [0,1]
  
  model = deepcopy(model)

  s = 0
  for p in model.parameters():
    p.data += lambd*torch.tensor(vec[s:s+p.numel()],device=p.device).view(p.size())
    s+= p.numel()

  return model.cuda() # model with shifted parameters

'''plot model loss along +-lambda*V'''

def curvature(model,vec,lams,criterion,inputs,targets):
  losses = []
  for lambd in lams:
    m = model_params_shift(model,vec,lambd)
    loss  = criterion(m(inputs),targets)
    losses.append(loss.cpu().detach().numpy())
  
  plt.plot(lams, losses)
  
  
class Gradient:
  def __init__(self,D,G,L1,L2,z_dim,device,data=None,dataloader=None):
    
    if data != None:
      self.data = data
      self.full_dataset = False
    else:
      self.data = dataloader
      self.full_dataset = True

    if not self.full_dataset:
      self.data = data

    self.device = device 

    self.D =  deepcopy(D).to(self.device)
    self.G =  deepcopy(G).to(self.device)
    
    self.L_D = L1
    self.L_G = L2

    self.z_dim = z_dim

    self.theta = list(self.D.parameters())
    self.phi   = list(self.G.parameters())

  def first_diff(self):
    if self.full_dataset == False:

      bz = len(self.data)
      z  = torch.rand((bz,self.z_dim),device=self.device).float()
      x  = self.data.float().to(self.device)

      x_fake1 = self.G(z)
      #x_fake2 = x_fake1.detach().requires_grad_()
      
      grad_D = grad(sum(self.L_D(x_fake1,x,self.D,self.device)),\
                    self.theta,create_graph=True,retain_graph=True)
      grad_G = grad(self.L_G(x_fake1,self.D,self.device),
                    self.phi,create_graph=True,retain_graph=True)

    else:

      grad_dis_epoch ={}
      grad_gen_epoch ={}
      n_data = 0

      for name, param in self.D.named_parameters():
        grad_dis_epoch[name] = torch.zeros_like(param,device=self.device).flatten()

      for name, param in self.G.named_parameters():
        grad_gen_epoch[name] = torch.zeros_like(param,device=self.device).flatten()

      for x_real in self.data:

        bz = len(x_real)
        z  = torch.rand((bz,self.z_dim),device=self.device).float()
        x_fake1 = self.G(z).float()
        #x_fake2 = x_fake1.detach().requires_grad_()
        x_real = x_real.to(self.device)
      
        dL1dtheta = grad(sum(self.L_D(x_real.float(),self.G(z).float(),self.D,self.device)),\
                        self.theta,create_graph=True,retain_graph=True)

        for ((name,_),g) in zip(self.D.named_parameters(),dL1dtheta):
            grad_dis_epoch[name]+=g.flatten()*len(x_real)

        dL2dphi   = grad(self.L_G(x_fake1,self.D,self.device),\
                         self.phi,create_graph=True,retain_graph=True)
        
        for ((name,_),g) in zip(self.G.named_parameters(),dL2dphi):
            grad_gen_epoch[name]+=g.flatten()*len(x_real)

        n_data +=len(x_real)

      grad_D = []
      grad_G = []

      for name,_ in self.D.named_parameters():
        grad_D.append(grad_dis_epoch[name]/n_data)

      for name,_ in self.G.named_parameters():
        grad_G.append(grad_gen_epoch[name]/n_data)

    return grad_D, self.theta, grad_G, self.phi
  
  
  
class Jacobian(linalg.LinearOperator):
  def __init__(self,first_grad,params,device,transpose=False):

    self.first_grad = first_grad
    self.params     = params
    self.transpose  = transpose
    self.device     = device 
    self.dtype      = np.dtype('Float32')
    
    n_params = 0
    for p in self.params:
      n_params+=p.numel()

    self.shape = (n_params,n_params)
  
  @staticmethod
  def vectorize(x):
    return torch.cat([torch.flatten(y) for y in x]).unsqueeze(-1)
  
  def JVP(self,v):
    # operator J(first_grad,params)v --> Hessian(params)v
    dF   = Jacobian.vectorize(self.first_grad)
    u    = torch.ones_like(dF,requires_grad=True)       
    g_v  = grad(dF,self.params,u,create_graph=True,retain_graph=True)
    g_v  = Jacobian.vectorize(g_v)
    ans, = grad(g_v,u,v)
    return ans

  def JTVP(self,v):
    # operator J^T(first_grad,params)v --> Hessian^T(params)v
    return Jacobian.vectorize(grad(Jacobian.vectorize(self.first_grad),self.params,v,retain_graph=True))

  def _matvec(self,v):
    v = np.expand_dims(v,axis=-1)
    v = torch.tensor(v,device=self.device)
    if self.transpose == True:
      return self.JTVP(v).cpu().detach().numpy()
    else:
      return self.JVP(v).cpu().detach().numpy()

  def trace(self,n_iter): ### n_iter for Hutchinson Stochastic Trace Estimators
    trace_list = []
    for _ in range(n_iter):
      v = self.rand_vec()
      trace_list.append(torch.matmul(v.T,self.JTVP(v)))
    return torch.mean(torch.tensor(trace_list))
  
  def eigen_pair(self,n_eigen,which):  #### returns [eig_val, eig_vec]
    
    #### WHICH ################
    #‘LM’ : largest magnitude
    #‘SM’ : smallest magnitude
    #‘LR’ : largest real part
    #‘SR’ : smallest real part
    #‘LI’ : largest imaginary part
    #‘SI’ : smallest imaginary part
    
    if self.transpose == True:
      return eigsh(self,which = which,k=n_eigen)
    else:
      return eigs(self,which = which,k=n_eigen)

  '''random vector from Rademacher distribution'''

  def rand_vec(self):
    v = torch.cat([torch.randint_like(p.data, high=2,device=self.device).flatten()\
          for p in self.params]).unsqueeze(-1)
    v[v==0.]=-1. 
    return v/np.sqrt(len(v))

  '''Orthogonilise: Gram-Schmidt process '''

  @staticmethod
  def ort(w,list1):
    for v in list1:
      w = w - torch.matmul(w.T,v)*v
      w = w/torch.norm(w)
    return w

  '''Lancazos algorithm'''

  def lanczos(self,n_iter):  ### Returns V basis and T tridiagonal matrix
                              
    v       = self.rand_vec()
    w_prime = self.JTVP(v)
    alpha   = torch.matmul(w_prime.T,v)
    w       = w_prime - alpha*v  ## projection of w_prime on v
    v_list      = [v]
    w_list      = [w]
    alpha_list  = [alpha]
    beta_list   = []

    V = torch.zeros((self.shape[0],n_iter))
    T = torch.zeros((n_iter,n_iter))
    T[0,0] = alpha
    V[:,0] = v.squeeze(-1)

    for i in range(n_iter-1):

      betta = torch.norm(w_list[-1])
      beta_list.append(betta)

      T[i,i+1] = betta
      T[i+1,i] = betta

      if betta !=0:
        v = w_list[-1]/betta
        v = Jacobian.ort(v,v_list)
      else:
        v = Jacobian.ort(self.rand_vec(),v_list)

      v_list.append(v)
      V[:,i+1] = v.squeeze(-1)

      w_prime = self.JTVP(v_list[-1])
      alpha = torch.matmul(w_prime.T,v_list[-1])
      alpha_list.append(alpha)
      T[i+1,i+1] = alpha

      w = w_prime-alpha_list[-1]*v_list[-1]-beta_list[-1]*v_list[-2]
      w_list.append(w)
    return V, T
  
def min_max(X): #X 2D array
  x1_gen = X[:,0]
  x2_gen = X[:,1]
  return np.min(x1_gen),np.max(x2_gen),\
         np.min(x2_gen),np.max(x2_gen)

'''vis function plots GAN 2D generated vs real points, with background coloured by discriminator output'''

def vis(G,D,X_train_save,z_fixed,device):
  with torch.no_grad():

    fig = go.Figure()
    G.eval()
    gen = G(z_fixed).cpu().detach().numpy()

    x1_g_min,x1_g_max,x2_g_min,x2_g_max = min_max(gen)
    x1_d_min,x1_d_max,x2_d_min,x2_d_max = min_max(X_train_save)
    
    u,v  = np.meshgrid(np.linspace(min(-5,x1_g_min,x1_d_min),\
                                   max(5,x1_g_max,x1_d_max),20),\
                       np.linspace(min(-5,x2_g_min,x2_d_min),\
                                   max(5,x2_g_max,x2_d_max),20))
    
    vals = np.hstack([u.reshape(400,1),v.reshape(400,1)])
    
    D.eval()
    pred = D(torch.tensor(vals,device=device).float()).cpu().detach().numpy().reshape(20,20)

    fig.add_trace(go.Contour(
        z=[list(x) for x in list(pred)],
        x=list(np.linspace(min(-5,x1_g_min,x1_d_min),max(5,x1_g_max,x1_d_max),20)),
        y=list(np.linspace(min(-5,x2_g_min,x2_d_min),\
                    max(5,x2_g_max,x2_d_max),20)),\
        contours=dict(
            coloring ='heatmap',
            showlabels = True, 
            labelfont = dict(size = 12,color = 'white'))
    ))


    fig.add_trace(go.Scatter(x=X_train_save[:, 0], y=X_train_save[:, 1],
                    mode='markers',
                    name='Real Data',marker_color='Green',marker=dict(size=7,color='Green')))

    fig.add_trace(go.Scatter(x=gen[:, 0], y=gen[:, 1],
                    mode='markers',
                    name='Fake data',marker_color='Red',marker=dict(size=10,color='Red',symbol = 'cross')))
    
    fig.update_layout(
        autosize=True,
        width=700,
        height=700)

    return fig

