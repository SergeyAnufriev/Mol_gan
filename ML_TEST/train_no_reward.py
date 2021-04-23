from model import R,Generator
from data_loader import Mol_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb
from utils import sweep_to_dict,L2_norm,wgan_dis,wgan_gen,grad_penalty,weight_init
import os
from vizulise import plot2
from valid import valid_compounds

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

data = DataLoader(Mol_dataset(dir_dataset),config.bz,shuffle=True,drop_last=True)

D = R(config,device)
G = Generator(config,9,5,5)

D.apply(weight_init(config.weight_init_D))
G.apply(weight_init(config.weight_init_G))

D.to(device)
G.to(device)

'''Turn on discriminator spectral normalisation'''
if config.Spectral_Norm_D == True:
    D.turn_on_spectral_norm()

opt_D = torch.optim.Adam(D.parameters(), lr=config.lr_d,betas=(0.0,0.9))
opt_G = torch.optim.Adam(G.parameters(), lr=config.lr_g,betas=(0.0,0.9))

'''Training loop'''

z_test             = torch.randn(5000,config.z_dim,device=device)

for epoch in range(config.epochs):
    for i,(A,X,_) in enumerate(data):

        '''Train discriminator'''

        opt_D.zero_grad()
        z = torch.randn(config.bz,config.z_dim).to(device)
        X_fake,A_fake  = G(z)
        D_real, D_fake = wgan_dis(A.to(device),X.to(device),A_fake.to(device),X_fake.to(device),D)
        D_loss = D_fake - D_real

        '''apply gradient penalty to loss if WGAN-GP'''

        if config.LAMBDA !=0:
            GP      =  grad_penalty(A.to(device),X.to(device),A_fake.to(device),X_fake.to(device),D,device)
            wandb.log({'GP':GP})
            D_loss  += config.LAMBDA*GP

        D_loss.backward()

        '''Log discriminator statistics'''

        wandb.log({'D_Wloss':D_loss})
        wandb.log({'D(real)':D_real})
        wandb.log({'D(fake)':D_fake})
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
            G_loss.backward()

            '''Log geneerator statistics'''

            wandb.log({'G_loss':G_loss})
            G_norm = L2_norm(G)
            wandb.log({'G_grad_L2':G_norm})
            total_norm = (G_norm**2+D_norm**2)** (1. / 2)
            wandb.log({'Total_L2':total_norm})


            opt_G.step()

    '''Log chemical performance per epoch'''

    z = torch.randn(config.bz,config.z_dim,device=device)
    X_fake,A_fake = G(z)
    plot2(A_fake,X_fake)
    x,a = G(z_test)
    wandb.log({'valid':valid_compounds(a,x,device)})

    PATH = os.path.join(run_loc,'G'+'_'+'epoch-{}.pt'.format(epoch))
    torch.save(G, PATH)
    wandb.save('*.pt')


