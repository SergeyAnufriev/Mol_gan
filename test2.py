import torch.nn as nn
import torch


''''Llalalalalalala '''
def permute3D(A):

  A = A.permute(2,0,1)
  A_new = torch.triu(A,diagonal=1) + torch.transpose(torch.triu(A,diagonal=1),1,2)

  return A_new.permute(1,2,0)


f = nn.functional.gumbel_softmax


x = torch.rand((5,5,4))

#x = 0.5*(x+torch.transpose(x,0,1))
y = f(x,hard=True)
y = permute3D(y)

for i in range(5):
    for j in range(5):
        if i !=j:
            print('symmetry',torch.all(y[i,j]==y[j,i]))
            print('sum',torch.sum(y[i,j])==1)



