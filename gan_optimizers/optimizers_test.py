from torch import optim
import torch

'''RMS prop '''

x = torch.tensor([1.],requires_grad=True)

y = x**2
opt = optim.SGD([x],0.01,0.2,nesterov=True)
y.backward()
print(x,x.grad)
opt.step()

opt.zero_grad()

y = x**2
y.backward()
print(x,x.grad)
opt.step()


print(x)

'''Finished calc with no grad'''
