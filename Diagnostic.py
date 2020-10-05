import torch
from torch import autograd
import wandb
from scipy.sparse import linalg
import numpy as np
from torch.autograd import grad
from copy import deepcopy


def grad_info(model,t):
  total_grad = []
  for name, param in model.named_parameters():
    grad = list(torch.flatten(param.grad).numpy())
    wandb.log({t+name: wandb.Histogram(grad)})
    total_grad +=grad 
  wandb.log({t+'L2':np.linalg.norm(total_grad)})
  
  
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
  
  
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
      
      grad_D = grad(self.L_D(x_fake1,x,self.D,self.device),\
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
      
        dL1dtheta = grad(self.L_D(x_real.float(),self.G(z).float(),self.D,self.device),\
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
    v = torch.tensor(v)
    if self.transpose == True:
      return self.JTVP(v).detach().numpy()
    else:
      return self.JVP(v).detach().numpy()

  def trace(self,n_iter): ### n_iter for Hutchinson Stochastic Trace Estimators
    trace_list = []
    for _ in range(n_iter):
      v = torch.cat([torch.randint_like(p.data, high=2,device=self.device).flatten()\
          for p in self.params if p.requires_grad == True]).unsqueeze(-1)
      v[v==0.]=-1. ### sample from Rademacher distribution
      trace_list.append(torch.matmul(v.T,self.JTVP(v)))
    return torch.mean(torch.tensor(trace_list))

      
def vis(G,D):
  with torch.no_grad():

    fig = go.Figure()
    G.eval()
    gen = G(z_fixed)
    gen = gen.cpu().data.numpy()

    x1_g_min,x1_g_max,x2_g_min,x2_g_max = min_max(gen)
    x1_d_min,x1_d_max,x2_d_min,x2_d_max = min_max(X_train_save)
    
    u,v  = np.meshgrid(np.linspace(min(-5,x1_g_min,x1_d_min),\
                                   max(5,x1_g_max,x1_d_max),20),\
                       np.linspace(min(-5,x2_g_min,x2_d_min),\
                                   max(5,x2_g_max,x2_d_max),20))
    
    vals = np.hstack([u.reshape(400,1),v.reshape(400,1)])
    
    D.eval()
    pred = D(torch.tensor(vals).float()).detach().numpy().reshape(20,20)

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
