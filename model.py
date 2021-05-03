import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch_geometric.nn import GlobalAttention

activation = {'tanh':nn.Tanh(),'relu':nn.LeakyReLU(),'sigmoid':nn.Sigmoid()}

def permute3D(A):

  A = A.permute(2,0,1)
  A_new = torch.triu(A,diagonal=1) + torch.transpose(torch.triu(A,diagonal=1),1,2)

  return A_new.permute(1,2,0)

def permute4D(A):

  bz,n,m,k = A.size()

  A_new = A.permute(1,2,3,0)
  A_new = A_new.reshape(n,m,-1)
  A_new = permute3D(A_new)
  A_new = A_new.reshape(n,m,k,bz)
  A_new = A_new.permute(3,0,1,2)

  return A_new


def to_list(bz,n_nodes):
  '''Returns assigns each batch instance nodes '''
  final_list = [0]*n_nodes
  for i in range(1,bz):
    final_list += [i]*n_nodes
  return final_list


class Generator(nn.Module):

    '''Class to create generator model'''
    '''Generator z: --> A, X'''
    '''A - graph adjecency tensor  size =Batch_size X NumberNodes X NumberNodes X Number of unique connections types'''

    # N - maximum number of atoms
    # T - number of atom types
    # Y - number of bond types

    #### Candidates to enforce A_sym = LL^T

    def __init__(self,config,N,T,Y):
      super(Generator, self).__init__()

      self.N = N
      self.T = T
      self.Y = Y

      self.temp = config.temp ### GambelSoftmax activation - temperature

      self.lin1 = nn.Linear(config.z_dim,config.h1_g)
      self.lin2 = nn.Linear(config.h1_g,config.h2_g)
      self.lin3 = nn.Linear(config.h2_g,config.h3_g)

      self.edges = nn.Linear(config.h3_g,self.N*self.N*self.Y)
      self.nodes = nn.Linear(config.h3_g,self.N*self.T)

      self.drop_out   = nn.Dropout(config.drop_out)

      self.act_       = activation[config.nonlinearity_G]

      self.act_nodes  = nn.functional.gumbel_softmax
      self.act_edges  = nn.functional.gumbel_softmax


    def forward(self,x):

        x           = self.act_(self.drop_out(self.lin1(x)))
        x           = self.act_(self.drop_out(self.lin2(x)))
        output      = self.act_(self.drop_out(self.lin3(x)))

        '''Compute Nodes feature matrix X'''
        nodes_logit = self.drop_out(self.nodes(output)).view(-1,self.N,self.T)
        nodes       = self.act_nodes(nodes_logit,dim=-1,tau=self.temp,hard=True)

        '''Compute adjecency tensor A, where A must be symmetric'''
        edges_logit      = self.drop_out(self.edges(output)).view(-1,self.N,self.N,self.Y)
        edges_logit_T    = torch.transpose(edges_logit,1,2)
        edges_logit      = 0.5*(edges_logit+edges_logit_T)
        edges_logit      = self.act_edges(edges_logit,dim=-1,tau=self.temp,hard=True)
        edges            = permute4D(edges_logit)

        return  nodes, edges

class Convolve(nn.Module):

    '''Graph convolution layer based on Relational Graph Convolution paper'''

    def __init__(self,in_channels,out_channels,device):
        super(Convolve,self).__init__()

        self.root         = nn.Linear(in_channels,out_channels,bias=True)

        self.single_      = nn.Linear(in_channels,out_channels,bias=True)
        self.double_      = nn.Linear(in_channels,out_channels,bias=True)
        self.triple_      = nn.Linear(in_channels,out_channels,bias=True)
        self.aromat_      = nn.Linear(in_channels,out_channels,bias=True)

        self.device       = device

    def forward(self,A,x):

        A        = A[:,:,:,:-1]
        sum_     = torch.sum(A,dim=2)
        norm     = 1./(sum_ + torch.full(sum_.size(),1e-7,device=self.device))  ## look for analytical solution
        A_new    = torch.einsum('sijcd,sicd->sijcd',A.unsqueeze(4),norm.unsqueeze(3))

        X_new    = torch.cat([self.single_(x).unsqueeze(dim=1),\
                              self.double_(x).unsqueeze(dim=1),\
                              self.triple_(x).unsqueeze(dim=1),\
                              self.aromat_(x).unsqueeze(dim=1)],dim=1)

        X_new    = torch.einsum('scja,sijcd->sia',X_new,A_new)   + self.root(x)

        return X_new

class gate_nn(torch.nn.Module):

  '''Neural network used to compute attention scores for each graph node'''
  '''Used in aggregation layer'''

  def __init__(self,in_channels,drop_out):
    super(gate_nn,self).__init__()
    self.lin1     = nn.Linear(in_channels,1,bias=True)
    self.drop_out = nn.Dropout(drop_out)
  def forward(self,x):
    return self.drop_out(self.lin1(x))


class nn_(torch.nn.Module):

  '''Graph nodes transformation network for Global aggregation layer'''
  '''Transforms graph nopdes embeddings into shape of final graph vector represenation'''

  def __init__(self,in_channels,out_channels,drop_out):
    super(nn_,self).__init__()
    self.lin2 = nn.Linear(in_channels,out_channels)
    self.drop_out = nn.Dropout(drop_out)
    self.act = nn.LeakyReLU()

  def forward(self,x):
    return self.act(self.drop_out(self.lin2(x)))



class Aggregate(torch.nn.Module):

    '''Aggregation layer, returns weighted sum (found by attention) of graph nodes represenations'''

    def __init__(self,gate_nn,nn,device):
        super(Aggregate, self).__init__()
        self.agg = GlobalAttention(gate_nn, nn)
        self.device = device

    def forward(self,x):

        bz,n,f = x.size()
        x = x.reshape(bz*n,f)
        batch = torch.tensor(to_list(bz,n)).type(torch.LongTensor).to(self.device)

        return self.agg(x,batch)


def spec_norm(m,type):

    '''Function to apply/remove spectral normalisation to layer m'''
    '''type: bool for apply/remove argument'''

    if m.__class__.__name__ == 'Linear':
        if type == True:
            nn.utils.spectral_norm(m,n_power_iterations=2)
        else:
            nn.utils.remove_spectral_norm(m)


class R(torch.nn.Module):

  '''Class to create discriminator model'''

  def __init__(self,config,device):
    super(R,self).__init__()

    self.spectral_norm_mode = False
    self.loss_type          = config.loss
    self.drop_out           = nn.Dropout(config.drop_out)

    self.conv1  = Convolve(5,config.h1_d,device)
    self.conv2  = Convolve(config.h1_d+5,config.h2_d,device)

    self.agr    = Aggregate(gate_nn(config.h2_d+5,config.drop_out),nn_(config.h2_d+5,config.h3_d,config.drop_out),device)

    self.linear = nn.Sequential(nn.Linear(config.h3_d,config.h3_d,bias=True),
                                activation[config.nonlinearity_D],
                                nn.Linear(config.h3_d,config.h4_d,bias=True),
                                activation[config.nonlinearity_D],
                                nn.Linear(config.h4_d,1,bias=True))

    self.act_ = activation[config.nonlinearity_D]

  def forward(self,A,x):

    '''Convolution layers'''
    h_1    = self.act_(self.drop_out(self.conv1(A,x)))
    h_2    = self.act_(self.drop_out(self.conv2(A,torch.cat((h_1,x),-1))))

    '''Aggregate layer'''
    h_3    = self.act_(self.agr.forward(torch.cat((h_2,x),-1)))

    '''Dense layers'''
    scalar = self.linear(h_3)

    if self.loss_type == 'GAN':
        scalar = activation['sigmoid'](scalar)

    return scalar


  def turn_on_spectral_norm(self):

      '''apply spectral norm to all discriminator weights '''

      if self.spectral_norm_mode is not None:
          assert self.spectral_norm_mode is False, "can't apply spectral_norm. It is already applied"

      for m in self.conv1.children():
          spec_norm(m,True)

      for m in self.conv2.children():
          spec_norm(m,True)

      for m in self.agr.agg.gate_nn.children():
          spec_norm(m,True)

      for m in self.agr.agg.nn.children():
          spec_norm(m,True)

      for m in self.linear.children():
          spec_norm(m,True)

      self.spectral_norm_mode = True


  def turn_off_spectral_norm(self):

      '''remove spectral norm from all discriminator weights'''

      if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

      for m in self.conv1.children():
          spec_norm(m,False)

      for m in self.conv2.children():
          spec_norm(m,False)

      for m in self.agr.agg.gate_nn.children():
          spec_norm(m,False)

      for m in self.agr.agg.nn.children():
          spec_norm(m,False)

      for m in self.linear.children():
          spec_norm(m,False)

      self.spectral_norm_mode = False
