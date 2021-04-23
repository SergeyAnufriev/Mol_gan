import torch


z = torch.tensor([1.],requires_grad=True)

a = torch.tensor([2.],requires_grad=True)
b = torch.tensor([2.],requires_grad=True)

L = a*(b*z)

L.backward()

print(b.grad)

print(torch.autograd.grad(b.grad,a))

