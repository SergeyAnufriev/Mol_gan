from random import choice
import torch

#### BFS #########################

class Queue:
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

'''BFS ALGORITHM TO COMPUTE SHORTEST DISTANCE'''

def distance(adj_list,start): ### BFS

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

'''find all disconnetced subgaphs in graph'''
def subgraphs(adj_list):

  nodes     = list(range(len(adj_list)))
  subgraphs = []

  while len(nodes)!=0:

    start = choice(nodes)
    dist  = distance(adj_list,start)
    subG  = [i for i,x in enumerate(dist) if x != float('inf')]
    subgraphs.append(subG)
    nodes = list(set(nodes)-set(subG))

  return subgraphs


def node_to_dict(mat):
  node_to_val = {}
  n,m = mat.size()
  pad = torch.tensor([0]*(m-1) + [1]).type(torch.LongTensor)
  for i in range(n):
    if all(mat[i,:] == pad):
      node_to_val[i] = 0
    else:
      node_to_val[i] = 1
  return node_to_val


def tens_to_adj_list(tens):
  n,_,m = tens.size()
  adj_l = [[] for _ in range(n)]
  pad1 = torch.zeros(m).type(torch.float32)
  pad2 = torch.tensor([0]*(m-1)+[1]).type(torch.float32)
  for i in range(n):
    for j in range(n):
      con_ = tens[i,j,:]
      data = [torch.equal(con_,pad1),torch.equal(con_,pad2)]
      if not any(data):
        adj_l[i].append(j)
  return adj_l


def valid(subgraphs,dict_):
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


def valid_graph(A,x):

  adj_list   = tens_to_adj_list(A)  ## From Adjeccency tensor to adjecency list
  dict_      = node_to_dict(x)      ## 1 if atom real, 0 empty node
  subgraphs_ = subgraphs(adj_list)  ## find all subgraphs by BFS algo

  if valid(subgraphs_,dict_) == 0:
    return 0,None,None
  else:
    for G in subgraphs_:
      if all(dict_[x] != 0 for x in G): 
        return 1,A[G,:,:][:,G,:],x[G,:]
