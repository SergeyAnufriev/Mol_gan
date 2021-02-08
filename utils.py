import torch
from torch.autograd import grad
from torch.autograd import Variable
from torch.utils.data import DataLoader,random_split
import yaml
import wandb
import numpy as np
from sklearn.metrics import r2_score
from rdkit import Chem

def array_to_atom(x,atom_set=['C','O','N','F']):
  max_atoms = len(atom_set)
  idx  = np.dot(x.numpy(),np.array(range(0,max_atoms+1))).astype(int)
  if idx == max_atoms:
    return 0
  else:
    atom = Chem.rdchem.Atom(atom_set[idx])
    return atom.GetAtomicNum()

def array_to_bond(x):
  if torch.sum(x).numpy() == 0:
    return None
  else:
    idx = np.dot(x.numpy(),np.array(range(0,5))).astype(int)
  return [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,\
            Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC,None][idx]


def A_x_to_mol(A,x):

  '''Converst Adjecency tensor (A) and node feature matrix (X) to molecule'''

  mol = Chem.RWMol()
  n_atoms = x.size()[0]

  non_empty_atoms = []
  for i in range(n_atoms):
    if x[i,:][-1].numpy() != 1:
      mol.AddAtom(Chem.Atom(0))
      non_empty_atoms.append(i)

  for i in non_empty_atoms:
    for j in non_empty_atoms:
      bond = array_to_bond(A[i,j,:])
      if i<j and bond != None:
        mol.AddBond(i,j,bond)
  for i in non_empty_atoms:
    mol.GetAtomWithIdx(i).SetAtomicNum(array_to_atom(x[i,:]))

  try:
      Chem.SanitizeMol(mol)
      return mol
  except:
    return None






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


def test(model, test_loader,cuda): #### over full test dataset average loss
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
  x = int(len(dataset)*0.8)
  y = len(dataset) - x
  train_dataset, test_dataset = random_split(dataset, [x, y])
  train_d, test_d = DataLoader(train_dataset,b_size,True,drop_last=True),\
                  DataLoader(test_dataset,b_size,True,drop_last=True)
  return train_d, test_d


real_label = 1.
fake_label = 0.

criterion = torch.nn.BCEWithLogitsLoss()

'''this file collects GAN losses functions'''



'''DCGAN DISCRIMINATOR LOSS'''
def gan_loss_dis(A_r,x_r,A_f,x_f,netD):

    b_size   = x_f.size()[0]
    label_r  = torch.full((b_size,), real_label, dtype=torch.float)
    label_f  = torch.full((b_size,), fake_label, dtype=torch.float)

    output_r  = netD(A_r,x_r).view(-1)
    Loss_real = criterion(output_r,label_r)

    output_f  = netD(A_f.detach(),x_f.detach()).view(-1)
    Loss_fake = criterion(output_f,label_f)

    return -Loss_real+Loss_fake

'''DCGAN GENERATOR LOSS'''
def gan_loss_gen(A_f,x_f,netD):

    b_size   = x_f.size()[0]
    label_r  = torch.full((b_size,), real_label, dtype=torch.float)
    output_f = netD(A_f,x_f).view(-1)
    Loss_gen = criterion(output_f,label_r)

    return -Loss_gen

'''WGAN LOSS DISCRIMINATOR'''
def wgan_dis(A_r,x_r,A_f,x_f,netD):

  Loss_real = netD(A_r,x_r).mean()
  Loss_fake = netD(A_f.detach(),x_f.detach()).mean()

  return Loss_real, Loss_fake

'''WGAN LOSS GENERATOR'''
def wgan_gen(A_f,x_f,netD):
  return -netD(A_f,x_f).mean()

'''GRADIENT PENALTY DISCRIMINATOR'''
def grad_penalty(A_r,x_r,A_f,x_f,netD,device):
    
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
    total_norm =0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
                        
'''
def gradient_penalty(self, y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(self.device)
    dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


x = torch.rand((32,4,5))

print(x.norm(2, dim=(1,2))-1)

y = x.view(x.size(0),-1)

print(y.size())
print(torch.sqrt(torch.sum(y**2, dim=1)))

X = torch.rand(5,10,30)

X1 = X[:2,:,:]

X2 = X[2:,:,:]

print(X.norm())

print((X1.norm()**2+X2.norm()**2)**(1./2))
print(X.size(),X1.size(),X2.size())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(Variable(torch.tensor([3.,4.])).to(device))
'''





