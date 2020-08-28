import torch
from rdkit import Chem
from molecular_metrics import MolecularMetrics
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit.Chem import RWMol
 


class Mol_dataset(Dataset,MolecularMetrics):
  def __init__(self,sdf_file,atom_set=['C','O','N','F'],N=9):
    self.suppl    = Chem.SDMolSupplier(sdf_file)
    self.atom_set = atom_set
    self.N = N 
    self.atom_to_num  = dict({x:y for x,y in zip(atom_set,range(len(atom_set)))})

  
  def __len__(self):
    return len(self.suppl)

  @staticmethod
  def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol
  
  @staticmethod
  def bond_features(bond):
    bt = bond.GetBondType()
    return (torch.Tensor([bt == Chem.rdchem.BondType.SINGLE,\
                          bt == Chem.rdchem.BondType.DOUBLE,\
                          bt == Chem.rdchem.BondType.TRIPLE,\
                          bt == Chem.rdchem.BondType.AROMATIC]))

  def array_to_atom(self,x):
    max_atoms = len(self.atom_set)
    idx  = np.dot(x.numpy(),np.array(range(0,max_atoms+1))).astype(int)
    if idx == max_atoms:
      return None
    else:

      atom = Chem.rdchem.Atom(self.atom_set[idx])
      return atom.GetAtomicNum()
  
  @staticmethod
  def array_to_bond(x):

    if torch.sum(x).numpy() == 0:
      return None

    else:

      idx = np.dot(x.numpy(),np.array(range(0,5))).astype(int)

      return [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,\
            Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC,None][idx]

  
  def A_x_to_mol(self,A,x):

    mol = RWMol()
    n_atoms = x.size()[0]

    for i in range(n_atoms):
      mol.AddAtom(Chem.Atom(0))

    for i in range(n_atoms):
      for j in range(n_atoms):
        bond = Mol_dataset.array_to_bond(A[i,j])
        if i>j and bond != None:
          mol.AddBond(i,j,bond)

    for i in range(n_atoms):
      mol.GetAtomWithIdx(i).SetAtomicNum(self.array_to_atom(x[i,:]))

    return mol


  def atom_features(self,mol):
    
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
    adjecency_mat = torch.zeros((self.N,self.N,4))
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjecency_mat[start,end,:] = Mol_dataset.bond_features(bond)
        adjecency_mat[end,start,:] = Mol_dataset.bond_features(bond)
    return torch.cat([adjecency_mat,torch.zeros((self.N,self.N,1))],dim=-1)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist() 

    mol = self.suppl[idx]

    LogP_ = Mol_dataset.water_octanol_partition_coefficient_scores([mol],norm=True) 
    QED_  = Mol_dataset.quantitative_estimation_druglikeness_scores([mol],norm=True)
    SAS_  = Mol_dataset.synthetic_accessibility_score_scores([mol],norm=True)

    reward  = LogP_*QED_*SAS_ 
    
    return self.adj_mat(mol), self.atom_features(mol), LogP_ ,QED_, SAS_, reward

