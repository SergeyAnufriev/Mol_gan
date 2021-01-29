from model import R,Generator
from data_loader import Mol_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb
from utils import sweep_to_dict,L2_norm
from utils import wgan_dis,wgan_gen,grad_penalty
import os
from rdkit import Chem
from vizulise import plot2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Initialise parameters and data path'''

dir_config  = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\config_files\train_no_reward.yaml'
dir_dataset = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'

wandb.init(config=sweep_to_dict(dir_config))
config  = wandb.config
run_loc = wandb.run.dir


'''Fix random seed'''

torch.manual_seed(config.seed)
np.random.seed(config.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

'''torch.random.seed(config.seed)'''

'''Initialise Models, Data and Optimizers'''

data = DataLoader(Mol_dataset(dir_dataset),config.bz,shuffle=True)

D = R(5,config.h1_d,config.h2_d,config.h3_d,config.h4_d,config.drop_out,device)
G = Generator(config.z_dim,config.h1_g,config.h2_g,config.h3_g,9,5,5,config.temp,config.drop_out)

D.to(device)
G.to(device)

opt_D = torch.optim.Adam(D.parameters(), lr=config.lr_d,betas=(0.0,0.9))
opt_G = torch.optim.Adam(G.parameters(), lr=config.lr_g,betas=(0.0,0.9))

'''Training loop'''



for epoch in range(config.epochs):
    for i,(A,X,_) in enumerate(data):

        '''Train discriminator'''

        opt_D.zero_grad()
        z = torch.randn(config.bz,config.z_dim).to(device)
        X_fake,A_fake = G(z)

        D_real, D_fake =  wgan_dis(A.to(device),X.to(device),A_fake.to(device),X_fake.to(device),D)

        wandb.log({'D(real)':D_real})
        wandb.log({'D(fake)':D_fake})
        wandb.log({'D_Wloss':-D_real+D_fake})

        GP      =  grad_penalty(A.to(device),X.to(device),A_fake.to(device),X_fake.to(device),D,device)
        D_loss  =  -D_real+ D_fake + config.LAMBDA*GP

        wandb.log({'GP':GP})
        wandb.log({'D_Wloss_GP':D_loss})
        D_loss.backward()

        D_norm = L2_norm(D)
        wandb.log({'D_grad_L2':D_norm})

        opt_D.step()

        '''n_crtitic default = 5'''

        if i%config.n_critic == 0 and i !=0:

            '''Train generator (later number critic iterations: for now 1 to 1'''

            opt_G.zero_grad()
            z             = torch.randn(config.bz,config.z_dim).to(device)
            X_fake,A_fake = G(z)
            G_loss        = wgan_gen(A_fake.to(device),X_fake.to(device),D)
            wandb.log({'G_loss':G_loss})
            G_loss.backward()
            G_norm = L2_norm(G)
            wandb.log({'G_grad_L2':G_norm})
            total_norm = (G_norm**2+D_norm**2)** (1. / 2)
            wandb.log({'Total_L2':total_norm})
            opt_G.step()

            plot2(A_fake,X_fake)


    ''' save generator model at each epoch, to check its performance later'''
    PATH = os.path.join(run_loc,'G'+'_'+'epoch-{}.pt'.format(epoch))
    torch.save(G, PATH)
    wandb.save('*.pt')


