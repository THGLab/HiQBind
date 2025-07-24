import os
import re
from copy import deepcopy
import json
import itertools
from typing import List, Optional, Union, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from rcsb import download_pdb_cif
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import openmm.app as app
from pdbfixer import PDBFixer
import networkx as nx


building_blocks = pd.read_csv(os.path.join(os.path.dirname(__file__), 'building_blocks.csv'), index_col=0)


def mol_from_seq(seq):
    rxn_link = AllChem.ReactionFromSmarts('[N:1]-[C:2]-[C:3](=[O:4])[O:5].[N:6]-[C:7]-[C:8]=[O:9]>>[N:1]-[C:2]-[C:3](=[O:4])-[N:6]-[C:7]-[C:8]=[O:9].[O:5]')
    rxn_ncap = AllChem.ReactionFromSmarts('[C:1](=[O:2])[O:3].[N:4]-[C:5]-[C:6]=[O:7]>>[C:1](=[O:2])-[N:4]-[C:5]-[C:6]=[O:7].[O:3]')
    rxn_ccap = AllChem.ReactionFromSmarts('[N:1]-[C:2]-[C:3](=[O:4])[O:5].[#7H1,#7H2,#7H3:6]>>[N:1]-[C:2]-[C:3](=[O:4])-[N:6].[O:5]')
    # Generic amide rxn
    rxn_general = AllChem.ReactionFromSmarts('[C:1](=[O:2])[O:3].[#7H1,#7H2,#7H3:4]>>[C:1](=[O:2])[N:4].[O:3]')

    for i in range(len(seq)):
        assert seq[i] in building_blocks.index, f'Unsupported residue: {seq[i]}'
        assert building_blocks.loc[seq[i], 'Category'] != 'other', f'Not amino-acid: {seq[i]}'
        if i == 0:
            product = Chem.MolFromSmiles(building_blocks.loc[seq[i], 'Smiles'])
            continue
        
        if i == 1 and building_blocks.loc[seq[0], 'Category'] == 'ncap':
            if building_blocks.loc[seq[i], 'Category'] == 'ccap':
                # ACE-NME cases
                rxn = rxn_general
            else:
                rxn = rxn_ncap
        elif i == len(seq) - 1 and building_blocks.loc[seq[i], 'Category'] == 'ccap':
            rxn = rxn_ccap
        else:
            rxn = rxn_link

        rxn_res = rxn.RunReactants((product, Chem.MolFromSmiles(building_blocks.loc[seq[i], 'Smiles'])))
        product = rxn_res[0][0]
        Chem.SanitizeMol(product)
    return product


def form_disulfide_bond(m):
    mnoh = Chem.RemoveHs(m)
    thiols = [match[0] for match in mnoh.GetSubstructMatches(Chem.MolFromSmarts('[SH:1]-[C:2]'))]
    results = []
    for i, j in itertools.combinations(thiols, 2):
        new_mol = Chem.RWMol(mnoh)
        new_mol.AddBond(i, j, Chem.BondType.SINGLE)
        new_mol.GetAtomWithIdx(i).SetNumExplicitHs(0)
        new_mol.GetAtomWithIdx(j).SetNumExplicitHs(0)
        mol = new_mol.GetMol()
        Chem.SanitizeMol(mol)
        results.append(mol)
    return results