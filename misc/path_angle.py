import torch
import wandb
import torch.nn as nn
import os
from model import Generator, R
from utils import sweep_to_dict, wgan_dis, wgan_gen, grad_penalty
from data_loader import Mol_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

dir_config = r'/config_files/GAN_param_grid.yaml'
PATH = r'/data'
dir_dataset = r'/data/gdb9_clean.sdf'

#wandb.init(config=sweep_to_dict(dir_config))
#config = wandb.config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net_D = R(5, 128, 64, 128, 64, 0.1, device)
net_G = Generator(32, 128, 256, 512, 9, 5, 5, 0.1, 0.1)

data = DataLoader(Mol_dataset(dir_dataset), 64, shuffle=True)

torch.save({'epoch': 1,
            'gen_state_dict': net_G.state_dict(),
            'dis_state_dict': net_D.state_dict()}, os.path.join(PATH, 'start.pth'))

torch.save({'epoch': 1,
            'gen_state_dict': net_G.state_dict(),
            'dis_state_dict': net_D.state_dict()}, os.path.join(PATH, 'end.pth'))

checkpoint1 = torch.load(os.path.join(PATH, 'start.pth'))
checkpoint2 = torch.load(os.path.join(PATH, 'end.pth'))


def path_angle(checkpoint1, checpoint2, G, D, dataset,device,ratio=0.1,n_points=2):


    L1                   = int(len(dataset)*ratio)
    lengths              = [L1, len(dataset) - L1]
    subsetA, _           = random_split(dataset, lengths)
    dataloader           = DataLoader(subsetA,batch_size=64,drop_last=True,shuffle=False)


    diff = []

    '''Caclulate difference'''

    for name, _ in D.named_parameters():
        delta = checpoint2['dis_state_dict'][name] - checkpoint1['dis_state_dict'][name]
        diff.append(delta)

    for name, _ in G.named_parameters():
        delta = checpoint2['gen_state_dict'][name] - checkpoint1['gen_state_dict'][name]
        diff.append(delta)

    dW = torch.cat([p.flatten() for p in diff]).to(device)

    cosine = []
    norm_v = []

    '''torch.linspace(-0.1, 1.1, n_points, device=device)'''

    D_state_ = D.state_dict()
    G_state_ = G.state_dict()

    for alpha in torch.tensor([0.5,0.5]):

        '''Gradinet accumulation'''

        D_state, G_state = D_state_,G_state_

        grad_gen_epoch = {}
        grad_dis_epoch = {}

        '''interpolate with alpha moving average'''

        for name, param in D.named_parameters():
            D_state[name] = alpha * checpoint2['dis_state_dict'][name] + (1 - alpha) * checkpoint1['dis_state_dict'][
                name]
            grad_dis_epoch[name] = torch.zeros_like(param)

        for name, param in G.named_parameters():
            G_state[name] = alpha * checpoint2['gen_state_dict'][name] + (1 - alpha) * checkpoint1['gen_state_dict'][
                name]
            grad_gen_epoch[name] = torch.zeros_like(param)

        D.load_state_dict(D_state)
        G.load_state_dict(G_state)

        '''compute gradient'''

        for (A, X, _) in dataloader:

            z = torch.randn(64, 32).to(device)
            X_fake, A_fake = G(z)

            '''zero accumulated gradient'''

            for p in G.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            for p in D.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            '''calculate losses'''

            D_real, D_fake = wgan_dis(A.to(device), X.to(device), A_fake.to(device), X_fake.to(device), D)
            GP     = grad_penalty(A.to(device), X.to(device), A_fake.to(device), X_fake.to(device), D, device)
            D_loss = -D_real + D_fake + 10 * GP
            G_loss = wgan_gen(A_fake.to(device), X_fake.to(device), D)

            '''calculate gradients'''

            for p in D.parameters():
                p.requires_grad = True

            for p in G.parameters():
                p.requires_grad = False

            D_loss.backward()

            for p in G.parameters():
                p.requires_grad = True

            for p in D.parameters():
                p.requires_grad = False

            G_loss.backward()

            for name, p in D.named_parameters():
                grad_dis_epoch[name] += p.grad * 64

            for name, p in G.named_parameters():
                grad_gen_epoch[name] += p.grad * 64

        grad_D = torch.cat([p.flatten() for _, p in grad_dis_epoch.items()]) / (len(dataloader)*64)
        grad_G = torch.cat([p.flatten() for _, p in grad_gen_epoch.items()]) / (len(dataloader)*64)

        total_grad  = torch.cat([grad_D, grad_G])

        dot_product = torch.dot(total_grad,dW)
        '''
        cosine.append(dot_product/(torch.norm(total_grad)*torch.norm(dW)))
        '''
        cosine.append(dot_product)

        norm_v.append(torch.norm(total_grad))


    return cosine, norm_v


gr, w = path_angle(checkpoint1, checkpoint2, net_G, net_D,Mol_dataset(dir_dataset),device)

print('gr', gr)
print('w', w)
