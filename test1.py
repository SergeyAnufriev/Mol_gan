from rdkit import Chem
import numpy as np

m = Chem.MolFromSmiles('Cc1ccccc1')

print(dir(m))


from scipy.sparse import csr_matrix
csr_matrix((3, 4), dtype=np.int8).toarray()



