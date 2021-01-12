import numpy as np
import torch
import math
from torch.autograd import grad
import matplotlib.pyplot as plt
import os
from scipy import linalg

os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''Initialize agents params'''

#x = torch.tensor([0.1],requires_grad=True)
#y = torch.tensor([1.1],requires_grad=True)

'''Define loss functions'''

ro_       = lambda x:   0.1*x**2 + 0.3*np.sin(x)
g1_       = lambda x,y: 0.1*x**2 + 0.2*(y-ro_(x))**2
f1_       = lambda x,y: 1+ 0.2*(x-0.8)**2 + 0.2*np.cos((3*(x-0.8)-np.pi/2)**2)

df_dx     = lambda x: 0.2*x -1.2*np.sin((3*(x-0.8)-np.pi/2)**2)*(3*(x-0.8)-np.pi/2)
dg_dy     = lambda x,y : 0.4*(y-ro_(x))

def Jacobian(x):

    dg_dydx = -0.08*x - 0.12*np.cos(x)
    df_dxdy = 0
    dg_dydy = 0.4
    df_dxdx = (-3.6*x + 0.6*np.pi + 2.88)*(18*x - 14.4 - 3*np.pi)*np.cos\
        (9*(x - 0.8 - np.pi/6)**2) - 3.6*np.sin(9*(x - 0.8 - np.pi/6)**2) + 0.2

    J = np.zeros((2,2))

    J[0,0] = -0.1*df_dxdx
    J[1,1] = -1*dg_dydy

    J[0,1] = -0.1*df_dxdy
    J[1,0] = -1*dg_dydx

    return np.identity(2)+J



'''Naive SGD algorithm
   start = (x_0,y_0),where x_0 and y_0 are initialised agents params repectively
   lr1, lr2 - the learning rates,n_steps - number of SGD steps '''
mu =0.5
def naive_sgd(start,lr1,lr2,n_steps):
    x_0,y_0 = start
    x_list = [x_0]
    y_list = [y_0]
    v = 0
    for _ in range(n_steps):
       # v = mu * v -lr1*df_dx(x_0)
       # x = x_0 +v
        x = x_0 -lr1*df_dx(x_0)
        y = y_0 -lr2*dg_dy(x_0,y_0)
        x_list.append(x)
        y_list.append(y)
        x_0 = x
        y_0 = y
    return x_list,y_list

x_list,y_list = naive_sgd((-2,0),0.1,1,100)

g1_vector = np.vectorize(g1_)
f1_vector = np.vectorize(f1_)

'''Plot losses for range -3<x<3 and 3<y<3 '''

x_range = np.linspace(-3,3,100)
y_range = np.linspace(-3,3,100)

y_ridge = [ro_(x) for x in list(x_range)]

grid_x, grid_y     = np.meshgrid(x_range, y_range)

'''(1) Grad_y[g1_x(x,0)] = 0 --> y = ro_(x) : the ridge'''

grid_y_grad_zero = np.full_like(y_range,0.)

g1 = g1_vector(grid_x, grid_y)
f1 = f1_vector(grid_x, grid_y)



plt.figure(figsize=(20, 10))
ax = plt.axes(projection='3d')

'''G(x,y) loss plot'''
ax.plot_surface(grid_x, grid_y, g1, rstride=1, cstride=1,
              cmap='terrain', edgecolor=None)

'''Ridge at G(x,y) where y = ro_(x) and dG(x,y)/dy = 0'''
ax.plot(x_range,y_ridge,[g1_(x,y) for x,y in zip(list(x_range),y_ridge)],color='r',linewidth=5)


'''f(x,y) loss plot'''
#ax.plot_surface(grid_x, grid_y, f1+6, rstride=1, cstride=1,
   #            cmap='terrain', edgecolor=None)

'''f(x) where y = ro_(x) '''
ax.plot(x_range,y_ridge,[f1_(x,y)+6 for x,y in zip(list(x_range),y_ridge)],color='r',linewidth=5)

'''Stakelberg Equilibrium for agaent with loss g(x,y)'''
ax.scatter(0.7,ro_(0.7),g1_(0.7,ro_(0.7)),color='black',linewidth=10)
ax.scatter(x_range,y_ridge,color='r')

'''Stakelberg Equilibrium for agaent with loss f(x)'''
ax.scatter(0.7,ro_(0.7),f1_(0.7,ro_(0.7))+6,color='black',linewidth=10)


ax.scatter(x_list,y_list,[g1_(x,y) for x,y in zip(x_list,y_list)])

ax.scatter(x_list,y_list,[f1_(x,y) +6 for x,y in zip(x_list,y_list)])

plt.show()


plt.figure(figsize=(20, 10))

plt.contour(grid_x, grid_y, g1)
plt.plot(x_list,y_list)
plt.scatter(0.7,ro_(0.7),color='r',linewidth=10)
plt.scatter(x_range,y_ridge,color='r')

plt.show()


plt.figure(figsize=(20, 10))

plt.contour(grid_x, grid_y, f1)
plt.scatter(x_range,y_ridge,color='r')
plt.plot(x_list,y_list)
plt.scatter(0.7,ro_(0.7),color='r',linewidth=10)

plt.show()

plt.figure(figsize=(20, 10))
plt.title('x-first agent params, y-second agent params, (x*,y*)-stalkelberg equilibrium point')
plt.plot(range(101),[abs(x-0.7) for x in x_list],label='|x-x*|')
plt.plot(range(101),[abs(y-ro_(0.7)) for y in y_list],label='|y-y*|')
plt.legend()

plt.show()


plt.figure(figsize=(20, 10))

plt.title('agent losses vs optimizer iteration steps')
plt.plot(range(101),[g1_(x,y) for x,y in zip(x_list,y_list)],label='G1_loss')
plt.plot(range(101),[f1_(x,y) for x,y in zip(x_list,y_list)],label='F1_loss')
plt.legend()

plt.show()


plt.plot(x_range,[f1_(x,0) for x in x_range])
plt.show()

print('eiegenvalues',linalg.eigvals(Jacobian(x_list[-1])))
print('df_dx',df_dx(x_list[-1]))
print('df_dy',dg_dy(x_list[-1],y_list[-1]))
