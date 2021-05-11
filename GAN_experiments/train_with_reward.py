from model import R,Generator
from data_loader import Mol_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import wandb
from utils import sweep_to_dict,L2_norm,wgan_dis,wgan_gen,grad_penalty,weight_init,clip_weight
import os
from vizulise import plot2
from valid import A_X_to_mols
from molecular_metrics import MolecularMetrics

device_type = 'GPU'

'''GPU/CPU calculations depending on if GPU is available'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device_type == 'GPU':
    dir_config  = r'/home/zcemg08/Scratch/Mol_gan2/config_files/GAN_param_grid.yaml'
    dir_dataset = r'/home/zcemg08/Scratch/data/gdb9_clean.sdf'
else:
    dir_config  = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\config_files\GAN_param_grid.yaml'
    dir_dataset = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'


'''Initialise parameters in config object'''
wandb.init(config=sweep_to_dict(dir_config))
config  = wandb.config
run_loc = wandb.run.dir

'''Fix random seed'''
torch.manual_seed(config.seed)
np.random.seed(config.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

'''Initialise data loader object'''
data = DataLoader(Mol_dataset(dir_dataset,device),config.bz,shuffle=True,drop_last=True)

'''Initialise generator G and discriminator D models based on config object'''
D = R(config,device)
V = R(config,device,'sigmoid')
G = Generator(config,9,5,5)

'''Apply weight initialisation'''
D.apply(weight_init(config.weight_init_D))
V.apply(weight_init(config.weight_init_D))
G.apply(weight_init(config.weight_init_G))


D.to(device)
G.to(device)
V.to(device)

'''Turn on/off discriminator spectral normalisation'''
if config.Spectral_Norm_D == True:
    D.turn_on_spectral_norm()

'''Optimizers set up'''
opt_D = torch.optim.RMSprop(D.parameters(), lr=config.lr_d,alpha=config.alpha)
opt_V = torch.optim.RMSprop(V.parameters(), lr=config.lr_d,alpha=config.alpha)
opt_G = torch.optim.RMSprop(G.parameters(), lr=config.lr_g,alpha=config.alpha)

def reward(mols):
    return torch.tensor([Mol_dataset.reward(mol)[0] if mol is not None else 0 for mol in mols],device=device,dtype=torch.float32)

mse = torch.nn.MSELoss()

'''Fixed random number to evaluate Generator model molecules valid score'''
z_test             = torch.randn(5000,config.z_dim,device=device)

'''Model chemical validity evaluation frequency = epoch/k'''
l = len(data)
k = int(l/5)
counter = 0

'''Training loop'''
for epoch in range(config.epochs):
    for i,(A,X,real_true_reward) in enumerate(data):

        '''Train discriminator'''
        counter +=1
        opt_D.zero_grad()
        z = torch.randn(config.bz,config.z_dim,device=device)
        X_fake,A_fake  = G(z)

        opt_V.zero_grad()
        '''Calculate actual reward for generated molecules'''
        fake_mols        = A_X_to_mols(A_fake,X_fake,device)
        fake_true_reward = reward(fake_mols).unsqueeze(1)

        '''Calculate rewards by reward network'''
        value_real       = V(A,X)
        value_fake       = V(A_fake,X_fake)

        '''Find Value function loss'''
        V_loss = mse(fake_true_reward,value_fake) + mse(real_true_reward,value_real)
        wandb.log({'Value_loss':V_loss,'epoch':counter/l})
        V_loss.backward()
        opt_V.step()

        '''Calculate disriminator loss'''
        D_real, D_fake = wgan_dis(A,X,A_fake,X_fake,D)
        D_loss = D_fake - D_real

        '''apply gradient penalty to discriminator loss if WGAN-GP'''

        if config.Lambda is not None:
            GP      =  grad_penalty(A,X,A_fake,X_fake,D,device)
            wandb.log({'GP':GP,'epoch':counter/l})
            D_loss  += config.Lambda*GP

        D_loss.backward()

        '''Log discriminator statistics'''

        wandb.log({'D_Wloss':D_loss,'epoch':counter/l})
        wandb.log({'D(real)':D_real,'epoch':counter/l})
        wandb.log({'D(fake)':D_fake,'epoch':counter/l})
        D_norm = L2_norm(D)
        wandb.log({'D_grad_L2':D_norm,'epoch':counter/l})

        opt_D.step()

        '''Clip discriminator weights after optimizer step if weight clipping is applied'''
        if config.clip_value is not None:
            clip_weight(D,config.clip_value)

        '''Train discriminator n_critic times more then generator'''
        if i%config.n_critic == 0 and i !=0:

            '''Train generator'''

            opt_G.zero_grad()
            z             = torch.randn(config.bz,config.z_dim,device=device)
            X_fake,A_fake = G(z)
            '''Calculate generator loss'''
            G_loss        = wgan_gen(A_fake,X_fake,D)
            Rl_loss       = V(A_fake,X_fake).mean()

            alpha = G_loss/Rl_loss

            Generator_loss = G_loss*0.8 - 0.2*alpha.detach()*Rl_loss
            Generator_loss.backward()

            '''Log generator statistics'''
            wandb.log({'RL_LOSS':Rl_loss,'epoch':counter/l})
            wandb.log({'G_loss':G_loss,'epoch':counter/l})
            wandb.log({'Generator_loss':Generator_loss,'epoch':counter/l})
            G_norm = L2_norm(G)
            wandb.log({'G_grad_L2':G_norm,'epoch':counter/l})
            total_norm = (G_norm**2+D_norm**2)** (1. / 2)
            wandb.log({'Total_L2':total_norm,'epoch':counter/l})
            opt_G.step()

        if i%k==0:
           '''Log model valid score and plot generated molecules per 'epoch/k' discriminator steps'''

           z = torch.randn(config.bz,config.z_dim,device=device)
           X_fake,A_fake = G(z)
           '''Visual inspection'''
           plot2(A_fake,X_fake)

           '''Valid compounds calculation'''
           x,a       = G(z_test)
           mols_fake = A_X_to_mols(a,x,device)
           wandb.log({'valid':MolecularMetrics.valid_total_score(mols_fake),'epoch':counter/l})

           '''Save generator weights'''
           PATH = os.path.join(run_loc,'G'+'_'+'epoch-{}.pt'.format(epoch))
           torch.save(G, PATH)
           wandb.save('*.pt')
