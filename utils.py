import torch
from torch.autograd import grad
from torch.autograd import Variable
from torch.utils.data import DataLoader,random_split
import yaml
import wandb
import numpy as np
from sklearn.metrics import r2_score
import torch.nn as nn


def JTVP(X,Y,V):

    '''Jacobian vector product where X is function
    to take derivative with respect to Y
    and V is the vector'''
    '''returns <(dx/dy)T,V>'''

    HTVP =torch.autograd.grad(
            X, Y,
            grad_outputs=V,allow_unused=True,retain_graph=True)[0]
    if HTVP is None:
        return torch.zeros_like(Y)
    else:
        return HTVP


def weight_init(type_):

    '''Initialise linear layers weights'''
    '''Returns weight function for model.apply(weight function)'''

    def weight(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if type_ == 'normal':
                m.weight.data.normal_(0.0, 0.02)
            elif type_ == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif type_ == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight,gain=(2**0.5))
            elif type_ == 'kaiming_normal':
                nn.init.kaiming_normal(m.weight)
            else:
                m.weight.data.fill_(0)
            if m.bias is not None:
                m.bias.data.fill_(0)
    return weight

def sweep_to_dict(dir):

    '''Sets parameters values given experiment.yaml file'''

    with open(dir) as file:
        params = yaml.safe_load(file)
    default_dict ={}
    pr = params['parameters']
    for k,v in pr.items():
        if 'value' in v:
            m = v['value']
        else:
            m = v['values'][0]
        default_dict[k] = m
    return default_dict


def test(model, test_loader,cuda):

  '''Evaluate model on regression preformance with R2
      calculated over test dataset'''

  model.eval()
  test_loss = 0
  y_pred = []
  y_real = []
  with torch.no_grad():
    for a,x,r in test_loader:
      outputs    = model(a.to(cuda),x.to(cuda))
      test_loss += criterion(outputs, r.to(cuda))
      y_pred.append(outputs.data.cpu().numpy())
      y_real.append(r)
  y_pred = np.vstack(y_pred)
  y_real = np.vstack(y_real)

  wandb.log({'Test_Loss':test_loss/(len(test_loader.dataset)/test_loader.batch_size)})
  wandb.log({'R2_TEST':r2_score(y_real,y_pred)})


def train_test(dataset,b_size):

  ''' Split dataset into train/test'''

  x = int(len(dataset)*0.8)
  y = len(dataset) - x
  train_dataset, test_dataset = random_split(dataset, [x, y])
  train_d, test_d = DataLoader(train_dataset,b_size,True,drop_last=True),\
                  DataLoader(test_dataset,b_size,True,drop_last=True)
  return train_d, test_d


real_label = 1.
fake_label = 0.

criterion = torch.nn.BCEWithLogitsLoss()

def gan_loss_dis(A_r,x_r,A_f,x_f,netD):
    '''DCGAN DISCRIMINATOR LOSS'''

    b_size   = x_f.size()[0]
    label_r  = torch.full((b_size,), real_label, dtype=torch.float)
    label_f  = torch.full((b_size,), fake_label, dtype=torch.float)

    output_r  = netD(A_r,x_r).view(-1)
    Loss_real = criterion(output_r,label_r)

    output_f  = netD(A_f,x_f).view(-1)
    Loss_fake = criterion(output_f,label_f)

    return -Loss_real+Loss_fake

def gan_loss_gen(A_f,x_f,netD):

    '''DCGAN GENERATOR LOSS'''

    b_size   = x_f.size()[0]
    label_r  = torch.full((b_size,), real_label, dtype=torch.float)
    output_f = netD(A_f,x_f).view(-1)
    Loss_gen = criterion(output_f,label_r)

    return -Loss_gen


def wgan_dis(A_r,x_r,A_f,x_f,netD):

  '''WGAN LOSS DISCRIMINATOR'''

  Loss_real = netD(A_r,x_r).mean()
  Loss_fake = netD(A_f.detach(),x_f.detach()).mean()

  return Loss_real, Loss_fake


def wgan_gen(A_f,x_f,netD):

  '''WGAN LOSS GENERATOR'''

  return -netD(A_f,x_f).mean()


def grad_penalty(A_r,x_r,A_f,x_f,netD,device):

    '''GRADIENT PENALTY DISCRIMINATOR'''
    
    eps   = torch.rand(A_r.size()[0],device=device)
    eps_x = torch.unsqueeze(torch.unsqueeze(eps,-1),-1)
    eps_A = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(eps,-1),-1),-1)
    
    x_hat = Variable(eps_x*x_r + (1-eps_x)*x_f,requires_grad=True).to(device)
    A_hat = Variable(eps_A*A_r + (1-eps_A)*A_f,requires_grad=True).to(device)
    
    d_hat = netD(A_hat,x_hat)
    
    gradients_x = grad(outputs=d_hat, inputs=x_hat,
                              grad_outputs=torch.ones(d_hat.size(),device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty_x = ((gradients_x.norm(2, dim=(1,2))-1) ** 2).mean()


    gradients_a = grad(outputs=d_hat, inputs=A_hat,
                              grad_outputs=torch.ones(d_hat.size(),device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]


    gradient_penalty_a = ((gradients_a.norm(2, dim=(1,2,3))-1) ** 2).mean()


    return gradient_penalty_x+gradient_penalty_a


def clip_weight(model,clip_value):

    '''Clip model weights by [-clip_value,clip_value]'''

    for p in model.parameters():
        p.data.clamp_(-clip_value,clip_value)


def L2_norm(model):

    '''Model gradients L2 norm'''

    total_norm =0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children
