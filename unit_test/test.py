import unittest
import torch
from model import R,Generator
from utils import dotdict,sweep_to_dict,get_children
from torch.nn.init import _calculate_fan_in_and_fan_out,_calculate_correct_fan,calculate_gain
import math
from rdkit import Chem

'''The following functions get tested'''
from utils import grad_penalty,weight_init
from valid import A_x_to_mol
from data_loader import Mol_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_dataset = r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\data\gdb9_clean.sdf'
suppl       = Chem.SDMolSupplier(dir_dataset)
dataset = Mol_dataset(dir_dataset,device)

config = sweep_to_dict(r'C:\Users\zcemg08\Documents\GitHub\Mol_gan\config_files\GAN_param_grid.yaml')
config = dotdict(config)

Dis    = R(config,device)
Gen    = Generator(config,9,5,5)


class TestSum(unittest.TestCase):


    def test_data_loader_and_AX_to_mol(self):
        '''convert mols to A,X then to mol,
        check if moles in smiles match'''

        fail_convert     = 0
        non_equal_mols   = 0
        non_equal_smiles = 0
        n_compounds = len(suppl)

        for mol in suppl:
            '''mol to graph'''
            A = dataset.adj_mat(mol)
            X = dataset.atom_features(mol)
            smiles_original = Chem.MolToSmiles(mol)
            try:
                '''graph to mol'''
                mol_back    = A_x_to_mol(A,X,device)
                smiles_back = Chem.MolToSmiles(mol_back)
                if smiles_back != smiles_original:
                    non_equal_smiles+=1
                if mol_back != mol:
                    non_equal_mols+=1
            except:
                fail_convert+=1

        mls  = non_equal_mols/n_compounds
        smi  = non_equal_smiles/n_compounds
        conv = fail_convert/n_compounds

        self.assertFalse(any(x>0 for x in [mls,smi,conv]),'the fraction encode-decode not equal mols={},'
                                                          'smiles={},fails to decode={}'.format(mls,smi,conv))

    def test_get_children(self):
        '''Test for true number of layers in generator and discriminator'''
        self.assertEqual(sum([1 if c.__class__.__name__ =='Linear' else 0 for c in get_children(Dis)]),15,'Discriminator has 15 linear layers')
        self.assertEqual(sum([1 if c.__class__.__name__ =='Linear' else 0 for c in get_children(Gen)]),5,'Generator has 5 linear layers')
        

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

    #40-84 Test weight initialisation

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
        #Kaiming normal default configs, which are arguments to weight initialisation
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
        #Kaiming uniform default configs, which are arguments to weight initialisation
        nonlinearity = 'leaky_relu'
        mode = 'fan_in'
        a = 0
        for model in [Dis,Gen]:
            for c in get_children(model):
                if c.__class__.__name__ =='Linear':
                    data           = c.weight.data
                    #calculate expected unifrom distr bound
                    fan = _calculate_correct_fan(data, mode)
                    gain = calculate_gain(nonlinearity, a)
                    std = gain / math.sqrt(fan)
                    bound = math.sqrt(3.0) * std
                    self.assertTrue(torch.abs(torch.max(data))<bound,'Weight values outside (-bound,bound) range')



if __name__ == '__main__':
    unittest.main()




