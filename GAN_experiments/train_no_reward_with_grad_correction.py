from model import R,Generator,to_list
from data_loader import Mol_dataset
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
import torch
import numpy as np
import wandb
from utils import sweep_to_dict,L2_norm,JTVP
from utils import wgan_dis,wgan_gen,grad_weight_info,gan_loss_dis,gan_loss_gen
import os
from vizulise import plot2
from valid import valid_compounds
from optimizers import LSD_Adam,parameters_grad_to_vector


device_type = 'GPU'

'''GPU/CPU calculations depending on if GPU is available'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device_type == 'GPU':
    dir_config  = r'/home/zcemg08/Scratch/Mol_gan2/config_files/GAN_param_grid.yaml'
    dir_dataset = r'/home/zcemg08/Scratch/data/gdb9_clean.sdf'
else:
    dir_config  = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\config_files\GAN_param_grid.yam'
    dir_dataset = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'

print(sweep_to_dict(dir_config))

wandb.init(config=sweep_to_dict(dir_config))
config  = wandb.config
run_loc = wandb.run.dir

print(config)
'''Fix random seed'''

torch.manual_seed(config.seed)
np.random.seed(config.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

'''torch.random.seed(config.seed)'''

'''Initialise Models, Data and Optimizers'''

data = DataLoader(Mol_dataset(dir_dataset),config.bz,shuffle=True,drop_last=True)

'''batch vector for intance normalization'''


D = R(5,config.h1_d,config.h2_d,config.h3_d,config.h4_d,config.drop_out,device)
G = Generator(config.z_dim,config.h1_g,config.h2_g,config.h3_g,9,5,5,config.temp,config.drop_out)

D.to(device)
G.to(device)

activation_D = {}
activation_G = {}

def get_activation(name,activation):

  def hook(model, input, output):
    activation[name] = output.detach()
  return hook


def hook(model,activation):
  for item in list(model._modules.items()):
    item[1].register_forward_hook(get_activation(item[0],activation))

def save_act(activation,name):
  for key, value in activation.items():
      if 'drop' not in key:
          wandb.log({name+'_'+key: wandb.Histogram(value.mean(axis=0).flatten().cpu().detach().numpy())})


#PATH = os.path.join(run_loc,'Generator.pt')
#torch.save(G, PATH)
#wandb.save('*.pt')

hook(G,activation_G)
hook(D,activation_D)

'''
opt_D = OptMirrorAdam(D.parameters(), lr=config.lr_d,betas=(0.0,0.99))
opt_G = OptMirrorAdam(G.parameters(), lr=config.lr_g,betas=(0.0,0.99))
'''

opt_D = LSD_Adam(D.parameters(),lr=config.lr_d,betas=(0.55,0.9))
opt_G = LSD_Adam(G.parameters(),lr=config.lr_g,betas=(0.5,0.9))


'''Discriminator and Generator weights as vectors'''
dis_params_flatten = parameters_to_vector(D.parameters())
gen_params_flatten = parameters_to_vector(G.parameters())
print('dis_params_flatten',dis_params_flatten)

z_test             = torch.randn(10,config.z_dim,device=device)

'''K - frequency loading valid calculations'''
k = len(data)/20

'''100-102 Fill empty d_grad and g_grad buffers'''
noise = torch.autograd.Variable(z_test)
X1,A1 = G(noise.to(device))
'''
if config.loss == 'WGAN':
    Loss = wgan_gen(A1.to(device),X1.to(device),D)
else:
'''

Loss = gan_loss_gen(A1.to(device),X1.to(device),D)
print('loss',Loss)
(0.0*Loss).backward(create_graph=True)


for epoch in range(config.epochs):
    for i,(A,X,_) in enumerate(data):

        '''Train discriminator'''

        z = torch.randn(config.bz,config.z_dim).to(device)
        X_fake,A_fake = G(z)

        '''
        if config.loss == 'WGAN':

            D_real, D_fake =  wgan_dis(A.to(device),X.to(device),A_fake.to(device),X_fake.to(device),D)
            wandb.log({'D(real)':D_real})
            wandb.log({'D(fake)':D_fake})
            wandb.log({'D_Wloss':-D_real+D_fake})
            #GP      =  grad_penalty(A.to(device),X.to(device),A_fake.to(device),X_fake.to(device),D,device)
            #wandb.log({'GP':GP})
            D_loss  =  -D_real+ D_fake
            wandb.log({'D_Wloss_GP':D_loss})
        else:
        
        '''

        D_loss = gan_loss_dis(A.to(device),X.to(device),A_fake.to(device),X_fake.to(device),D)
        wandb.log({'D_REAL':D(A.to(device),X.to(device))})
        wandb.log({'D_FAKE':D(A_fake.to(device),X_fake.to(device))})
        wandb.log({'D_loss':D_loss})

        '''123-127 Create differentiable dLossD_dX'''
        grad_d = torch.autograd.grad(D_loss, inputs=(D.parameters()), create_graph=True)

        for p, g in zip(D.parameters(), grad_d):
            p.grad = g

        ''''144-146 calculate delta_Y'''
        gen_params_flatten_prev = gen_params_flatten + 0.0
        gen_params_flatten = parameters_to_vector(G.parameters()) + 0.0
        delta_gen_params_flatten = gen_params_flatten - gen_params_flatten_prev

        '''d_LossG_dY'''
        grad_gen_params_flatten = parameters_grad_to_vector(G.parameters())

        '''Jacobian vector product <H_xy,delta_Y>'''

        vjp_dis = tuple([JTVP(grad_gen_params_flatten,param,delta_gen_params_flatten) \
                         for param in D.parameters()])

        D_norm = L2_norm(D)
        wandb.log({'D_grad_L2':D_norm})
        grad_weight_info(D,'D')
        save_act(activation_D,'D_act')

        '''Apply gradient correction to optimizer'''
        opt_D.step(vjps=vjp_dis)

        '''n_crtitic default = 5'''

        if i%config.n_critic == 0 and i !=0:

            '''Train generator (later number critic iterations: for now 1 to 1'''

            z             = torch.randn(config.bz,config.z_dim).to(device)
            X_fake,A_fake = G(z)

            '''
            if config.loss == 'WGAN':
                G_loss        = wgan_gen(A_fake.to(device),X_fake.to(device),D)
            else:
            '''
            G_loss        = gan_loss_gen(A_fake.to(device),X_fake.to(device),D)

            wandb.log({'G_loss':G_loss})


            '''162-165 Create differentiable dLossG_dY'''
            grad_g = torch.autograd.grad(G_loss, inputs=(G.parameters()), create_graph=True)

            for p, g in zip(G.parameters(), grad_g):
                p.grad = g

            ''''161-163 calculate delta_X'''
            dis_params_flatten_prev = dis_params_flatten + 0.0
            dis_params_flatten = parameters_to_vector(D.parameters())
            delta_dis_params_flatten = dis_params_flatten - dis_params_flatten_prev

            '''d_LossD_dX'''
            grad_dis_params_flatten = parameters_grad_to_vector(D.parameters())

            '''Jacobian vector product <H_yx,delta_X>'''

            vjp_gen = tuple([JTVP(grad_dis_params_flatten,param,delta_dis_params_flatten) \
                         for param in G.parameters()])

            G_norm = L2_norm(G)
            wandb.log({'G_grad_L2':G_norm})
            total_norm = (G_norm**2+D_norm**2)** (1. / 2)
            wandb.log({'Total_L2':total_norm})
            grad_weight_info(G,'G')
            save_act(activation_G,'G_act')

            '''Apply gradient correction to optimizer'''
            opt_G.step(vjps=vjp_gen)

        if i%k==0:
            z = torch.randn(config.bz,config.z_dim,device=device)
            X_fake,A_fake = G(z)
            plot2(A_fake,X_fake)
            x,a = G(z_test)
            wandb.log({'valid':valid_compounds(a,x,device)})

    #PATH = os.path.join(run_loc,'G'+'_'+'epoch-{}.pt'.format(epoch))
    #torch.save(G.state_dict(), PATH)
    #wandb.save('*.pt')
