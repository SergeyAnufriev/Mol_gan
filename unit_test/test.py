import unittest
import torch
from model import R,Generator
from utils import dotdict,sweep_to_dict,get_children
from torch.nn.init import _calculate_fan_in_and_fan_out,_calculate_correct_fan,calculate_gain
import math

'''The following functions get tested'''
from utils import grad_penalty,weight_init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = sweep_to_dict(r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\config_files\GAN_param_grid.yaml')
config = dotdict(config)

Dis    = R(config,device)
Gen    = Generator(config,9,5,5)


class TestSum(unittest.TestCase):

    '''22-33 Test spectral norm discriminator on/off'''

    def test_spectral_norm_on(self):
        Dis.turn_on_spectral_norm()
        for c in get_children(Dis):
            if c.__class__.__name__ =='Linear':
                self.assertGreater(len(c._forward_pre_hooks),0,'No Hook applied --> No Spectral Norm applied')
                for _, hook in c._forward_pre_hooks.items():
                    hook_name = str(hook)
                    self.assertNotEqual(hook_name.find('SpectralNorm'),-1,'Spectral Norm is not included in the hook')

    def test_spectral_norm_off(self):
        Dis.turn_on_spectral_norm()
        Dis.turn_off_spectral_norm()
        for c in get_children(Dis):
            if c.__class__.__name__ =='Linear':
                for _, hook in c._forward_pre_hooks.items():
                    hook_name = str(hook)
                    self.assertEqual(hook_name.find('SpectralNorm'),-1,'Spectral Norm is applied')

    '''40-84 Test weight initialisation'''

    def test_weight_normal(self):
        Dis.apply(weight_init('normal'))
        Gen.apply(weight_init('normal'))
        for model in [Dis,Gen]:
            for c in get_children(model):
                if c.__class__.__name__ =='Linear':
                    data      = c.weight.data
                    mu, sigma = torch.mean(data), torch.std(data)
                    self.assertAlmostEqual(mu,0,delta=0.01,msg='mean is not close to zero')
                    self.assertAlmostEqual(sigma,0.02,delta=0.003,msg='std is not close to 0.2')
                    
    def test_weight_xavier_normal(self):
        Dis.apply(weight_init('xavier_normal'))
        Gen.apply(weight_init('xavier_normal'))
        for model in [Dis,Gen]:
            for c in get_children(model):
                if c.__class__.__name__ =='Linear':
                    data         = c.weight.data
                    mu, sigma    = torch.mean(data), torch.std(data)
                    f_in, f_out  = _calculate_fan_in_and_fan_out(data)
                    sigma_xavier = 2/torch.sqrt(torch.tensor([f_in+f_out],dtype=torch.float32))
                    self.assertAlmostEqual(mu,0,delta=0.01,msg='mean is not close to zero')
                    self.assertAlmostEqual(sigma,sigma_xavier,delta=0.0015,msg='std is not close to sigma_xavier')

    def test_weight_kaiming_normal(self):
        Dis.apply(weight_init('kaiming_normal'))
        Gen.apply(weight_init('kaiming_normal'))
        '''Kaiming normal default configs, which are arguments to weight initialisation'''
        nonlinearity = 'leaky_relu'
        mode = 'fan_in'
        a = 0
        for model in [Dis,Gen]:
            for c in get_children(model):
                if c.__class__.__name__ =='Linear':
                    data         = c.weight.data
                    mu, sigma    = torch.mean(data), torch.std(data)
                    fan = _calculate_correct_fan(data, mode)
                    gain = calculate_gain(nonlinearity, a)
                    std_kaiming = gain / math.sqrt(fan)
                    self.assertAlmostEqual(mu,0,delta=0.01,msg='mean is not close to zero')
                    self.assertAlmostEqual(sigma,std_kaiming,delta=0.0015,msg='std is not close to sigma_xavier')


    def test_weight_kaiming_uniform(self):
        Dis.apply(weight_init('kaiming_normal'))
        Gen.apply(weight_init('kaiming_normal'))
        '''Kaiming uniform default configs, which are arguments to weight initialisation'''
        nonlinearity = 'leaky_relu'
        mode = 'fan_in'
        a = 0
        for model in [Dis,Gen]:
            for c in get_children(model):
                if c.__class__.__name__ =='Linear':
                    data           = c.weight.data
                    '''calculate expected unifrom distr bound'''
                    fan = _calculate_correct_fan(data, mode)
                    gain = calculate_gain(nonlinearity, a)
                    std = gain / math.sqrt(fan)
                    bound = math.sqrt(3.0) * std
                    self.assertTrue(torch.abs(torch.max(data))<bound)



if __name__ == '__main__':
    unittest.main()




