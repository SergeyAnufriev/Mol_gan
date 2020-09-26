import torch
from torch import autograd
import wandb
from scipy.sparse import linalg
import numpy as np

def grad_info(model,t):
  total_grad = []
  for name, param in model.named_parameters():
    grad = list(torch.flatten(param.grad).numpy())
    wandb.log({t+name: wandb.Histogram(grad)})
    total_grad +=grad 
  wandb.log({t+'L2':np.linalg.norm(total_grad)})

class JacobianVectorProduct(linalg.LinearOperator):
    def __init__(self, grad, params):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)
        self.shape = (self.grad.size(0), self.grad.size(0))
        self.dtype = np.dtype('Float32')
        self.params = params

    def _matvec(self, v):
        v = torch.Tensor(v)
        if self.grad.is_cuda:
            v = v.cuda()
        grad_vector_product = torch.dot(self.grad, v)
        hv = autograd.grad(grad_vector_product, self.params, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        hv = torch.cat(_hv)
        return hv.cpu()
      
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
