import torch
from torch import autograd
from numpy import linalg 
import wandb

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
