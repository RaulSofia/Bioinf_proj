import pandas as pd
import numpy as np
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import Chem
from vocabulary import Vocabulary

def MolWithoutIsotopesToSmiles(mol):
  atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
  for atom, isotope in atom_data:
      if isotope:
          atom.SetIsotope(0)
  smiles = Chem.MolToSmiles(mol)
  for atom, isotope in atom_data:
      if isotope:
          atom.SetIsotope(isotope)
  return smiles

def keep_largest_fragment(smiles):
    mol = Chem.MolFromSmiles(smiles)
    frags = Chem.GetMolFrags(mol, asMols=True)
    frags = sorted(frags, key=lambda x: x.GetNumAtoms(), reverse=True)
    return Chem.MolToSmiles(frags[0])

data = pd.read_csv("./data_clean_a2d.csv")
remover = SaltRemover()
data['SMILES'] = data['SMILES'].map(lambda x: Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(x))))
data['SMILES'] = data['SMILES'].map(lambda x: keep_largest_fragment(x))
data['SMILES'] = data['SMILES'].map(lambda x: MolWithoutIsotopesToSmiles(Chem.MolFromSmiles(x)))
data['SMILES'] = data['SMILES'].map(lambda x: Chem.CanonSmiles(x))

data.reset_index(drop=True, inplace=True)
#setting max_len to the max len of the smiles in the data
vocab = Vocabulary("vocab.csv")

data_to_drop = []
# print(len(tokenized), len(data))
for i, tok_smile in enumerate(data['SMILES']):
    try:
        tokenized = vocab.tokenize([tok_smile])
        encoded = vocab.encode(tokenized)
    except Exception as e:
        print(i)
        data_to_drop.append(i)
        continue

data = data.drop(data_to_drop)


#center and reduce
# data['pCHEMBL_norm'] = (data['pCHEMBL'] - data['pCHEMBL'].mean()) / data['pCHEMBL'].std()
data['pCHEMBL_norm'] = (data['pCHEMBL'] - data['pCHEMBL'].min()) / (data['pCHEMBL'].max() - data['pCHEMBL'].min())

data = data.drop_duplicates(subset=['SMILES'])
data.to_csv("./data_clean_nosalt_canon_a2d.csv", index=False)