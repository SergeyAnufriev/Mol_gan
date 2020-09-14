from rdkit import Chem

m = Chem.MolFromSmiles('Cc1ccccc1')

print(dir(m))
