import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import sys
sys.path.append('../DLEPS')
import molecule_vae
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw

import numpy as np  
import pandas as pd

df = pd.read_csv("../data/BindindDB_IC50_human.csv")
print(df.head())
