import os, sys
import argparse
import multiprocessing as mp
from datetime import datetime
import json
from tqdm import tqdm
import pandas as pd

from rdkit import Chem
import parmed
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors


def read_chain_ids_from_pdb(f_pdb):
    chain_ids = set()
    with open(f_pdb) as f:
        for line in f:
            if line.startswith('SEQRES'):
                chain_id = line[11]
                chain_ids.add(chain_id)
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                break
    if len(chain_ids) == 0:
        struct = parmed.load_file(f_pdb)
        chain_ids = set([r.chain for r in struct.residues])
    return chain_ids


def get_molecule_properties(mol):
    """
    Computes molecular properties and returns them in a dictionary.
    
    :param mol: RDKit molecule object
    :return: Dictionary with molecular properties
    """
    properties = {
        'Ligand SMILES': Chem.MolToSmiles(mol),
        'Ligand MW': Descriptors.MolWt(mol),
        'Ligand LogP': Crippen.MolLogP(mol),
        'Ligand TPSA': rdMolDescriptors.CalcTPSA(mol),
        'Ligand NumRotBond': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'Ligand NumHeavyAtoms': mol.GetNumHeavyAtoms(),
        'Ligand NumHDon': rdMolDescriptors.CalcNumHBD(mol),
        'Ligand NumHAcc': rdMolDescriptors.CalcNumHBA(mol),
        'Ligand QED': Descriptors.qed(mol)
    }
    return properties


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', help='Input CSV of the process.py')
    parser.add_argument('-d', '--dir', dest='dir', help='Directory of the process.py output')
    parser.add_argument('-m', '--num_workers', dest='num_workers', type=int, default=64, help="Number of workers to run in parallel")
    parser.add_argument('-o', dest='output', help='Output metadata (csv) format')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Print information as much as possible')

    args = parser.parse_args()

    def log(*msg):
        if args.verbose:
            print(*msg)

    metadata = pd.read_csv(args.input)

    def process(pdbid):
        directory = os.path.join(args.dir, pdbid)
        if not os.path.isfile(os.path.join(directory, 'done.tag')):
            return []
        
        jfile = os.path.join(directory, 'rcsb_data.json')
        with open(jfile) as f:
            jdata = json.load(f)

        data = jdata['data']['entry']

        release_time = data['rcsb_accession_info']['initial_release_date']
        year = datetime.fromisoformat(release_time.rstrip('Z')).year

        refine = data['refine']
        if refine is None:
            resolution = 'NMR' 
        else:
            resolution = refine[0]['ls_d_res_high']

        chain_infos = {}
        for entity in data['polymer_entities']:
            entity_info = entity.get('rcsb_polymer_entity_container_identifiers', {})
            uniprot_ids = entity_info['uniprot_ids'] if entity_info['uniprot_ids'] is not None else []
            chains = entity_info.get('auth_asym_ids', [])
            name = entity.get('rcsb_polymer_entity', {}).get('pdbx_description', '')

            if len(uniprot_ids) == 1:
                uniprot_id = uniprot_ids[0]
            elif len(uniprot_ids) == 0:
                # print("No uniprot ID found for chains:", chains)
                uniprot_id = ""
                name = ""
            else:
                log(f"{pdbid} - More than one uniprot ID found for chains:", chains)
                uniprot_id = ""
                name = ""
            
            for chain_id in chains:
                chain_infos[chain_id] = {
                    'uniprot_id': uniprot_id,
                    'name': name
                }

        identifiers = []
        for dirname in os.listdir(directory):
            if (not dirname.startswith('.')) and os.path.isdir(os.path.join(directory, dirname)):
                identifiers.append(dirname)
        
        ligand_infos_by_idts = {}
        protein_infos_by_idts = {}
        for idt in identifiers:
            # protein uniprot name and id
            f_pdb = os.path.join(directory, idt, f'{idt}_protein_refined.pdb')
            chain_ids = read_chain_ids_from_pdb(f_pdb)
            pinfo = {'Protein UniProtID': [], 'Protein UniProtName': []}
            for cid in chain_ids:
                if cid == ' ':
                    continue
                if chain_infos[cid]['uniprot_id']:
                    pinfo['Protein UniProtID'].append(chain_infos[cid]['uniprot_id'])
                    pinfo['Protein UniProtName'].append(chain_infos[cid]['name'])
            protein_infos_by_idts[idt] = {key: ','.join(value) for key, value in pinfo.items()}
            # ligand
            mol = Chem.SDMolSupplier(os.path.join(directory, idt, f'{idt}_ligand_refined.sdf'))[0]
            linfo = get_molecule_properties(mol)
            ligand_infos_by_idts[idt] = linfo

        subdf = metadata.query(f'PDBID == "{pdbid}"')
        recs = []
        if subdf['origin'].unique().shape[0] == 1:
            # There is only one unique Binding affinity Annotations 
            for idt in identifiers:
                _, name, chain, resnum = tuple(idt.split('_'))
                record = {
                    "PDBID": pdbid,
                    "Resolution": resolution,
                    "Year": year,
                    "Ligand Name": name,
                    "Ligand Chain": chain,
                    "Ligand Residue Number": resnum,
                    "Binding Affinity Measurement": subdf.iloc[0]['measurement'],
                    "Binding Affinity Sign": subdf.iloc[0]['sign'],
                    "Binding Affinity Value": subdf.iloc[0]['value'],
                    "Binding Affinity Unit": subdf.iloc[0]['unit'],
                    "Log Binding Affinity": subdf.iloc[0]['logvalue'],
                    "Binding Affinity Source": subdf.iloc[0]['source'],
                    "Binding Affinity Annotation": subdf.iloc[0]['origin'],
                }
                record.update(protein_infos_by_idts[idt])
                record.update(ligand_infos_by_idts[idt])
                recs.append(record)
        else:
            # More than one binding affinity annotations associated with one PDBID
            for _, row in subdf.iterrows():
                chain = str(row['Ligand chain'])
                resname = str(row['Ligand CCD'])
                resnum = str(row['Ligand residue sequence number'])
                idt = f'{pdbid}_{resname}_{chain}_{resnum}'
                if idt in identifiers:
                    record = {
                        "PDBID": pdbid,
                        "Resolution": resolution,
                        "Year": year,
                        "Ligand Name": resname,
                        "Ligand Chain": chain,
                        "Ligand Residue Number": resnum,
                        "Binding Affinity Measurement": row['measurement'],
                        "Binding Affinity Sign": row['sign'],
                        "Binding Affinity Value": row['value'],
                        "Binding Affinity Unit": row['unit'],
                        "Log Binding Affinity": row['logvalue'],
                        "Binding Affinity Source": row['source'],
                        "Binding Affinity Annotation": row['origin'],
                    }
                    record.update(protein_infos_by_idts[idt])
                    record.update(ligand_infos_by_idts[idt])
                    recs.append(record)
                    identifiers.remove(idt)
            if len(identifiers) > 0:
                log(f'{pdbid} - Some items does not match binding affinity: {identifiers}')

        return recs
    
    pdb_ids = list(metadata['PDBID'].unique())
    with mp.Pool(64) as p:
        results = list(tqdm(p.imap_unordered(process, pdb_ids, chunksize=1), total=len(pdb_ids)))
    
    records = []
    for recs in results:
        if len(recs) == 0:
            continue
        records += recs

    df = pd.DataFrame(records)
    df = df.sort_values(by=['PDBID', 'Ligand Chain'])
    df.to_csv(args.output, index=None)
    
    print(f"Number of PDB entries: {df['PDBID'].unique().shape[0]}")
    print(f"Number of protein-ligand structures: {df.shape[0]}")