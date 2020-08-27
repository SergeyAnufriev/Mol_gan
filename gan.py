import torch
import torch.nn as nn
from torch.nn.functional import one_hot


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

def A_mat(data):   ### data to one-hot-encode adjacency tensor
  edge_type  = data.edge_attr
  edge_type_ = one_hot(edge_type)
  n_nodes    = data.num_nodes
  adj_mat    = torch.zeros((n_nodes,n_nodes,len(torch.unique(edge_type))))

  edge_index = data.edge_index
  for i in range(data.num_edges):
    k, m = edge_index[0,i],edge_index[1,i]
    adj_mat[k,m,:] = edge_type_[i,:]

  return adj_mat.to(dtype = torch.float32)


def to_list(bz,n_nodes):
  final_list = [0]*n_nodes
  for i in range(1,bz):
    final_list += [i]*n_nodes
  return final_list


class Generator(nn.Module):

    # N - maximum number of atoms
    # T - number of atom types
    # Y - number of bond types

    #### Candidates to enforce A_sym = LL^T

    def __init__(self,z_dim,N,T,Y,temp):
      super(Generator, self).__init__()

      self.N = N
      self.T = T
      self.Y = Y

      self.temp = temp ### GambelSoftmax activation - temperature

      self.lin1 = nn.Linear(z_dim,128)
      self.lin2 = nn.Linear(128,256)
      self.lin3 = nn.Linear(256,512)

      self.edges = nn.Linear(512,self.N*self.N*self.Y)
      self.nodes = nn.Linear(512,self.N*self.T)

      self.act   = nn.functional.gumbel_softmax

    def forward(self,x):

        output      = self.lin3(self.lin2(self.lin1(x)))

        nodes_logit = self.nodes(output).view(-1,self.N,self.T)
        nodes       = self.act(nodes_logit,dim=-1,tau=self.temp,hard=True)

        edges_logit      = self.edges(output).view(-1,self.N,self.N,self.Y)
        edges_logit_T    = torch.transpose(edges_logit,1,2)
        edges_logit      = 0.5*(edges_logit+edges_logit_T)

        edges_logit      = self.act(edges_logit,dim=-1,tau=self.temp,hard=True)

        edges = permute4D(edges_logit)

        return  nodes, edges
    
    
    
 

class Aggregate(torch.nn.Module):
    def __init__(self,gate_nn,nn):
        super(Aggregate, self).__init__()
        self.agg = GlobalAttention(gate_nn, nn)
        self.act = torch.nn.Sigmoid()
    def forward(self,x,batch=None):
        bz,n,f = x.size()
        x = x.reshape(bz*n,f)
        batch = torch.tensor(to_list(bz,n)).type(torch.LongTensor)
        return self.agg(x,batch)
      
  
  
  
class Convolve(nn.Module):
    def __init__(self,in_channels,out_channels,n_relations):
      super(Convolve,self).__init__()

      self.weight     = Parameter(torch.zeros(in_channels,out_channels,n_relations))  ### Look at initializations 
      self.theta_root = Parameter(torch.zeros(in_channels,out_channels))  ### Look at initializations 

    def forward(self,A,x):

      A        = A[:,:,:,:-1]
      sum_     = torch.sum(A,dim=2)
      norm     = 1./(sum_ + torch.full(sum_.size(),1e-7))  ## look for analytical solution
      A_new    = torch.einsum('sijcd,sicd->sijcd',A.unsqueeze(4),norm.unsqueeze(3))
      Theta_ij = torch.einsum('abc,sijcd->sijabd',self.weight,A_new).squeeze(-1)
      x_new    = torch.einsum('sja,sijab->sib',x,Theta_ij) + torch.matmul(x,self.theta_root)


      return self.act(x_new)
