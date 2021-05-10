import torch
from rdkit import Chem
from molecular_metrics import MolecularMetrics
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


bond_dict = {Chem.rdchem.BondType.SINGLE:0,Chem.rdchem.BondType.DOUBLE:1,
             Chem.rdchem.BondType.TRIPLE:2,Chem.rdchem.BondType.AROMATIC:3}


class Mol_dataset(Dataset,MolecularMetrics):

  '''Pytorch dataset class to sample molecules from dataset
  in form of A,X,r
  A - adjecency tensor
  X - node feature matrix
  r - scalar, reward'''

  def __init__(self,sdf_file,device,atom_set=['C','O','N','F','*'],N=9):
    self.suppl    = Chem.SDMolSupplier(sdf_file)
    self.atom_set = atom_set
    self.N = N 
    self.atom_to_num  = dict({x:y for x,y in zip(atom_set,range(len(atom_set)))})
    self.device = device

  def __len__(self):
    return len(self.suppl)

  def atom_features(self,mol):

    '''Input: rdkit mol object
      Output: node feature matrix size max_num_atoms x n_atom_types including no atom'''
    
    a_list = [self.atom_to_num[atom.GetSymbol()] for atom in mol.GetAtoms()]
    delta  =  self.N - len(a_list)

    '''add empty atoms if n_atoms < max_atoms number'''
    if delta>0:
      a_list = a_list + [self.atom_to_num['*']]*delta

    '''One hot encode list of atom numbers to node feature matrix'''
    X = one_hot(torch.tensor(a_list,device=self.device),num_classes=5)

    return X.type(torch.float32)

  def adj_mat(self,mol):

    '''Input: rdkit mol object
      Output: adjecency tensor size max_num_atoms x max_num_atoms x n_bond_types including no bond'''

    adjecency_mat = torch.zeros((self.N,self.N,5),device=self.device,dtype=torch.float32)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjecency_mat[start,end,:][bond_dict[bond.GetBondType()]] = 1
        adjecency_mat[end,start,:][bond_dict[bond.GetBondType()]] = 1

    return adjecency_mat


  @staticmethod
  def reward(mol):

    '''Input: rdkit mol object
       Output: reward'''

    LogP_ = Mol_dataset.water_octanol_partition_coefficient_scores([mol],norm=True)
    QED_  = Mol_dataset.quantitative_estimation_druglikeness_scores([mol],norm=True)
    SAS_  = Mol_dataset.synthetic_accessibility_score_scores([mol],norm=True)

    return LogP_*QED_*SAS_.astype('float32')


  def __getitem__(self,idx):

    '''Overwrite pytorch get item method'''

    if torch.is_tensor(idx):
      idx = idx.tolist() 

    mol = self.suppl[idx]

    while mol is None:
      idx+=1
      mol  = self.suppl[idx]

    r = torch.tensor(Mol_dataset.reward(mol),device=self.device,dtype=torch.float32)

    return self.adj_mat(mol), self.atom_features(mol), r

