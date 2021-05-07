from random import choice
import torch
from rdkit import Chem
import numpy as np


class Queue:

    '''BFS queue object '''

    def __init__(self):
        self.items  = []
    def enq(self,x):
        self.items.insert(0,x)
    def __len__(self):
        return len(self.items)
    def deq(self):
        if len(self.items ) == 0:
            return None
        else:
            return self.items.pop()


def distance(adj_list,start):

    '''BFS ALGORITHM TO COMPUTE SHORTEST DISTANCE in graph'''

    q = Queue()
    colour   = []
    dist = []
    parent   = []
    for i in range(len(adj_list)):
        if i == start:
           c = 'GRAY'
           d = 0
           p = None
        else:
            c = 'WHITE'
            d = float('inf')
            p = None
        colour.append(c)
        dist.append(d)
        parent.append(p)
    q.enq(start)
    while len(q)!=0:
        u = q.deq()
        for v in adj_list[u]:
            if colour[v] == 'WHITE':
               colour[v] = 'GRAY'
               dist[v] = dist[u] + 1
               parent[v] = u
               q.enq(v)
        colour[u] = 'BLACK'
    return dist


def subgraphs(adj_list):

  '''find all disconnetced subgaphs in graph
     Input: graph adjecency list
     Output: list of isolated graphs adjecency lists'''

  nodes     = list(range(len(adj_list)))
  subgraphs = []

  while len(nodes)!=0:

    start = choice(nodes)
    dist  = distance(adj_list,start)
    subG  = [i for i,x in enumerate(dist) if x != float('inf')]
    subgraphs.append(subG)
    nodes = list(set(nodes)-set(subG))

  return subgraphs


def node_to_dict(mat,device):

  '''
  Input: one hot node feature matrix N_node X N_node_types
  Output: dictionary where key = node number and value = {0,1}
  0 - empty node
  1 - non empty node
  '''

  node_to_val = {}
  n,m = mat.size()
  pad = torch.tensor([0]*(m-1) + [1],dtype=torch.float32,device=device)
  for i in range(n):
    if all(mat[i,:] == pad):
      node_to_val[i] = 0
    else:
      node_to_val[i] = 1
  return node_to_val


def tens_to_adj_list(tens,device):

  '''Input: Adjecency tensor size = N_nodes x N_nodes x Type_of_connections
     Output: Adjecency list graph reperesenation'''

  n,_,m = tens.size()
  adj_l = [[] for _ in range(n)]
  pad1 = torch.zeros(m,dtype=torch.float32,device=device)
  pad2 = torch.tensor([0]*(m-1)+[1],dtype=torch.float32,device=device)
  for i in range(n):
    for j in range(n):
      con_ = tens[i,j,:]
      data = [torch.equal(con_,pad1),torch.equal(con_,pad2)]
      if not any(data):
        adj_l[i].append(j)
  return adj_l


def valid(subgraphs,dict_):

  '''Input: list of adjecency lists and dictionary which tells if a node is an empty node
   Output: {0,1}
   0 - Invalid graph
   1 - Valid graph

   Graph is valid if all other disconnected graphs are self isolated empty nodes'''

  n_valid = 0  ### number of graphs with real atoms
  for G in subgraphs:
    if len(G)>1 and any(dict_[x] == 0 for x in G):  ## check if empty nodes are self isolated or not
      return 0
    elif any(dict_[x] == 1 for x in G):  ### check if real atom present in subgraph
      n_valid+=1
  if n_valid == 1:
    return 1  ### only one subgraph must have real atoms
  else:
    return 0


def valid_graph(A,x,device):

  '''Function returns valid sub Graph, if all other disconnected graphs
   are empty nodes'''

  adj_list   = tens_to_adj_list(A,device)  ## From Adjeccency tensor to adjecency list
  dict_      = node_to_dict(x,device)      ## 1 if atom real, 0 empty node
  subgraphs_ = subgraphs(adj_list)  ## find all subgraphs by BFS algo

  if valid(subgraphs_,dict_) == 0:
    return 0,None,None
  else:
    for G in subgraphs_:
      if all(dict_[x] != 0 for x in G): 
        return 1,A[G,:,:][:,G,:],x[G,:]


def array_to_atom(x,atom_set=['C','O','N','F']):

  '''Input: one-hot atom represenation
     Output rdkit-atom specific atom number'''

  max_atoms = len(atom_set)
  idx  = np.dot(x.cpu().numpy(),np.array(range(0,max_atoms+1))).astype(int)
  if idx == max_atoms:
    return 0
  else:
    atom = Chem.rdchem.Atom(atom_set[idx])
    return atom.GetAtomicNum()

def array_to_bond(x):

  '''Input: one-hot bond represenation
     Output rdkit bond object'''

  if torch.sum(x).cpu().numpy() == 0:
    return None
  else:
    idx = np.dot(x.cpu().numpy(),np.array(range(0,5))).astype(int)
  return [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,\
            Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC,None][idx]


def A_x_to_mol(A,x,device):

  '''Convert Adjecency tensor (A) and node feature matrix (x)
  to molecule, i.e rdkit mol object
  if invalid molecule returns None'''

  _,A,x = valid_graph(A,x,device)

  mol = Chem.RWMol()
  n_atoms = x.size()[0]

  non_empty_atoms = []
  for i in range(n_atoms):
    if x[i,:][-1].cpu().numpy() != 1:
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


def A_X_to_mols(A,X,device):

    '''Returns list of mol objects if molecule is a valid
    otherwise returns None's'''

    bz = A.size()[0]
    mols = []
    A,X  = A.detach(),X.detach()

    for i in range(bz):
        a,x = A[i,:,:,:],X[i,:,:]
        v,_,_ = valid_graph(a,x,device)
        if v == 0:
            mols+=[None]
        else:
            mols+=[A_x_to_mol(a,x,device)]

    return mols
