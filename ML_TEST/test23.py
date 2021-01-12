from model import R
from data_loader import Mol_dataset
import wandb
import torch.nn as nn
from torch import float32,device,manual_seed,backends,cuda,save,no_grad
import torch.optim as optim
import numpy as np
import random
from Diagnostic import plot_real_vs_predicted,train_info
from utils import train_test,sweep_to_dict,test


dir_config  = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\ML_TEST\cycle_lr.yaml'
dir_dataset = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'


print(sweep_to_dict(dir_config))
