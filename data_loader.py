import torch
from rdkit import Chem
from molecular_metrics import MolecularMetrics
from torch.utils.data import Dataset
import numpy as np
from torch.nn.functional import one_hot


class Mol_dataset(Dataset,MolecularMetrics):

  '''Pytorch dataset class to sample molecules from dataset
  in form of A,X,r
  A - adjecency tensor
  X - node feature matrix
  r - scalar, reward'''

  def __init__(self,sdf_file,atom_set=['C','O','N','F'],N=9):
    self.suppl    = Chem.SDMolSupplier(sdf_file)
    self.atom_set = atom_set
    self.N = N 
    self.atom_to_num  = dict({x:y for x,y in zip(atom_set,range(len(atom_set)))})

  
  def __len__(self):
    return len(self.suppl)

  @staticmethod
  def mol_with_atom_index(mol):

    '''Assign each atom atom index attribute,
    each index is a 0<unique value<number of atoms'''

    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol
  
  @staticmethod
  def bond_features(bond):

    '''Input: bond object
      Output: one-hot vector represenation'''

    bt = bond.GetBondType()
    return (torch.Tensor([bt == Chem.rdchem.BondType.SINGLE,\
                          bt == Chem.rdchem.BondType.DOUBLE,\
                          bt == Chem.rdchem.BondType.TRIPLE,\
                          bt == Chem.rdchem.BondType.AROMATIC]))

  def atom_features(self,mol):

    '''Input: rdkit mol object
      Output: node feature matrix size max_num_atoms x n_atom_types including no atom'''
    
    a_list = []
    max_atoms_types = len(self.atom_set)

    for atom in mol.GetAtoms():
      a_list.append(self.atom_to_num[atom.GetSymbol()])     
    
    a_list  = torch.tensor(a_list).type(torch.LongTensor)
    
    N_f_mat = one_hot(a_list,num_classes=max_atoms_types)
    n_atoms = N_f_mat.size()[0]
    N_f_mat = torch.cat([N_f_mat,torch.zeros((n_atoms,1))],dim=1)
    delta   = self.N - n_atoms

    if delta>0:
      pad = [[0]*max_atoms_types + [1] for _ in range(delta)]
      pad = torch.tensor(pad).type(torch.LongTensor)
      N_f_mat = torch.cat([N_f_mat,pad],dim=0)
    return N_f_mat

  def adj_mat(self,mol):

    '''Input: rdkit mol object
      Output: adjecency tensor size max_num_atoms x max_num_atoms x n_bond_types including no bond'''

    adjecency_mat = torch.zeros((self.N,self.N,4))
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjecency_mat[start,end,:] = Mol_dataset.bond_features(bond)
        adjecency_mat[end,start,:] = Mol_dataset.bond_features(bond)
    return torch.cat([adjecency_mat,torch.zeros((self.N,self.N,1))],dim=-1)


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
      if mol is None:
        idx+=1
        mol  = self.suppl[idx]

    r = Mol_dataset.reward(mol)

    return self.adj_mat(mol), self.atom_features(mol), r

