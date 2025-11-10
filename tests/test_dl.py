from rdkit import Chem
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../utils')
from hiqbind.fix_ligand import run_dl


def test_protonation(in_smi, out_smi):
    out_mol = run_dl(Chem.MolFromSmiles(in_smi))
    out_smi_dl = Chem.MolToSmiles(out_mol) 
    out_smi = Chem.MolToSmiles(Chem.MolFromSmiles(out_smi))
    assert out_smi_dl == out_smi, f'{out_smi_dl} != {out_smi}'
    

if __name__ == '__main__':
    test_cases = [
        ('CO', 'CO'),
        ('C(=O)O', 'C(=O)[O-]'),
        ('CN', 'C[NH3+]'),
        ('P(=O)(O)(O)O', 'P(=O)([O-])([O-])[O-]'),
        ('S(=O)(=O)(O)O', 'S(=O)(=O)([O-])([O-])'),
        ('c1ccccc1S', 'c1ccccc1[S-]'),
        ('c1ccccc1O', 'c1ccccc1O'),
        ('C(=N)N', 'C(=[NH2+])N'),
        ('CNO', 'CNO'),
        ('CC=N', 'CC=[NH2+]'),
        ('CC=NO', 'CC=NO'),
        ('C=NC=O', 'C=NC=O'),
        ('CNNC', 'C[NH2+]NC'),
        ('CNN', 'C[NH2+]N'),
        ('N(C)(C)N', 'N(C)(C)[NH3+]'),
        ('N(C)N(C)C', '[NH2+](C)N(C)C'),
        ('c1cc2c(cc1C(=[NH2+])N)[nH]c([nH+]2)Cc3[nH]c4cc(ccc4n3)C(=[NH2+])N', "NC(=[NH2+])c1ccc2nc(Cc3nc4ccc(C(N)=[NH2+])cc4[nH]3)[nH]c2c1")
    ]
    for in_smi, out_smi in test_cases:
        test_protonation(in_smi, out_smi)