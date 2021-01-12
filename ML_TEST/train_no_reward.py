from model import R,Generator
from data_loader import Mol_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb
from utils import sweep_to_dict
from utils import wgan_dis,wgan_gen,grad_penalty

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Initialise parameters and data path'''

dir_config  = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\ML_TEST\train_no_reward.yaml'
dir_dataset = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'

wandb.init(config=sweep_to_dict(dir_config))
config = wandb.config

'''Fix random seed'''

torch.manual_seed(config.seed)
np.random.seed(config.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
#torch.random.seed(config.seed)

'''Initialise Models, Data and Optimizers'''

data = DataLoader(Mol_dataset(dir_dataset),config.bz,shuffle=True)

D = R(5,config.h1_d,config.h2_d,config.h3_d,config.h4_d,config.drop_out,device)
G = Generator(config.z_dim,config.h1_g,config.h2_g,config.h3_g,9,5,5,config.temp,config.drop_out)

D.to(device)
G.to(device)

opt_D = torch.optim.Adam(D.parameters(), lr=config.lr_d,betas=(0.0,0.9))
opt_G = torch.optim.Adam(G.parameters(), lr=config.lr_g,betas=(0.0,0.9))

'''Training loop'''

for _ in range(config.epochs):
    for i,(A,X,_) in enumerate(data):

        '''Train discriminator'''

        opt_D.zero_grad()
        z = torch.randn(config.bz,config.z_dim).to(device)
        X_fake,A_fake = G(z)

        Wloss_d =  wgan_dis(A,X,A_fake,X_fake,D)
        wandb.log({'Wloss_d':Wloss_d})

        GP      =  grad_penalty(A,X,A_fake,X_fake,D)
        D_loss  =  Wloss_d + config.LAMBDA*GP

        wandb.log({'D_loss':D_loss})
        D_loss.backward()
        opt_D.step()

        '''Train generator (later number critic iterations: for now 1 to 1'''
        '''n_crtitic default = 5'''

        if (i+1)%5 == 0:

            opt_G.zero_grad()
            z             = torch.randn(config.bz,config.z_dim).to(device)
            X_fake,A_fake = G(z)
            G_loss        = wgan_gen(A_fake,X_fake,D)
            wandb.log({'G_loss':G_loss})
            G_loss.backward()
            opt_G.step()




