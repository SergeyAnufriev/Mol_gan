from model import Convolve,A_mat
import torch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


edge_index = torch.tensor([[0, 1, 1, 2,3,4],
                           [1, 0, 2, 1,4,3]], dtype=torch.long)
x = torch.tensor([[1.,2.,3.,4.,5.], [5.,4.,3.,2.,1.],\
                  [1.,1.,1.,1.,2.],[7.,6.,8.,9.,0.],\
                  [4.,3.,3.5,7.,1.2]], dtype=torch.float)

attr = torch.tensor([0,0,1,1,2,2],dtype=torch.long)

data = Data(x=x, edge_index=edge_index,edge_attr=attr)

C  = Convolve(5,2,3)
C2 = RGCNConv(in_channels=5,out_channels=2,num_relations=3)

list1 = []

for p in C2.parameters():
  list1.append(p.data)

for i in range(3):
  C.weight.data[:,:,i] = list1[0][i,:,:]

C.theta_root.data = list1[1]

adj = torch.cat([A_mat(data),torch.zeros((5,5,1))],dim=-1).unsqueeze(0) ## PAD NONE RELATION


print(C.forward(adj,data.x.unsqueeze(0)),C2.forward(data.x,data.edge_index,data.edge_attr))
