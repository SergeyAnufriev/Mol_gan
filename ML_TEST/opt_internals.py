from model import R,Generator
from torch.optim import RMSprop
import torch
import pickle

'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = R(5,128,64,128,64,0.1,device)
#G = Generator(config.z_dim,config.h1_g,config.h2_g,config.h3_g,9,5,5,config.temp,config.drop_out)


opt_D = RMSprop(D.parameters(),lr=0.001)



print(checkpoint['opt_state_dict'])
'''

v = torch.tensor([[0.0401, 0.0073, 0.0241, 0.0092, 0.0364, 0.0450, 0.0402, 0.0010, 0.0309,
         0.0307, 0.0607, 0.0807, 0.0042, 0.0005, 0.0267, 0.0271, 0.0790, 0.0155,
         0.0106, 0.0644, 0.1033, 0.0768, 0.0090, 0.1657, 0.1143, 0.2328, 0.0357,
         0.0171, 0.0984, 0.1209, 0.0009, 0.2618, 0.0557, 0.1918, 0.1114, 0.1435,
         0.0386, 0.0006, 0.0364, 0.0740, 0.0572, 0.1598, 0.0330, 0.1191, 0.0106,
         0.0359, 0.0118, 0.0224, 0.0251, 0.0346, 0.0020, 0.0342, 0.0051, 0.0080,
         0.0112, 0.0016, 0.0094, 0.0182, 0.0156, 0.0044, 0.0030, 0.0308, 0.0488,
         0.0591]])

print(v.size())
