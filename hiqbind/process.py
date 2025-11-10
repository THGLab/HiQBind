import os, glob
from collections import defaultdict, Counter
import multiprocessing as mp
import traceback
import networkx as nx
from typing import List, Optional, Union, Tuple, Dict

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import gemmi
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import openmm.app as app

from fix_protein import *
from fix_ligand import *
from fix_polymer import *
from rcsb import *
# from refine import *


standardResidues = [
    'ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR',
    'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL',
    # 'CYM', 'CYX', 'GLH', 'ASH', 'LYN', 'HID', 'HIE', 'HIP', 'HIN' 
    'A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI'
]

VERBOSE = False

# Ligands with elements other than this list will be discarded
COMMON_ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
# Ligands with less than 4 heavy atoms will be discarded
MAX_HEAVY_ATOMS = 4
# Ligands with distances < 2.0 Angstrom to the protein will be discarded
STERIC_CLASH_THRESH = 2.0
# Chains with no. of residues between 2-20 will be considered as a "ligand"
MIN_NUM_RES_POLY = 2
MAX_NUM_RES_POLY = 20
# Number of missing residues larger than this value will not be added
MAX_ADD_MISSING_RES = 10
# Chains closer than 10 Angstrom will be included
BINDING_CUTOFF: float = 10.0
# HETATMs closer than 4 Angstrom to a chain will also be included
HETATM_CUTOFF: float = 4.0


def mmcif_corrector(value, type = str):
    if value == '?': return None
    else:
        if type == str: return value.replace("'", "")
        elif type == int: return int(value)
        elif type == float: return float(value)
        elif type == bool: return True if value == 'yes' else False
        else: return value


def get_cif_header(cif_path, chains = None, lig_of_interest = None):
    """
    This function is for getting the cif header file for each individual pdb code for alignment purposes
    
    - the cif header contains rich information about the reference uniprot id, the sequence length, and the chain id
    """ 
    doc = gemmi.cif.read(cif_path)  # Using read_url to directly read from RCSB PDB
    block = doc.sole_block()
    if not block:
        raise ValueError("No Block Found")
    chain_meta_data = defaultdict(dict)
    uniprot_meta_data = {}
    entity_to_chain = defaultdict(list)

    # Prepare DataFrame for alignment data
    columns = ['pdb_id', 'chain_id', 'entity_id', 'pdbx_seq_one_letter_code', 'pdbx_seq_one_letter_code_can', 'uniprot_id', ' uniprot_name', 'ref_seq_one_letter',
               'pdb_seq_align_beg', 'pdb_seq_align_end', 'uniprot_seq_align_beg', 'uniprot_seq_align_end', 'nstd_linkage', 'nstd_monomer']
    alignment_data = []

    # Prepare DataFrame for sequence differences
    diff_columns = ['pdb_id', 'chain_id', 'pdb_res_num', 'pdb_res_name', 'uniprot_res_num', 'uniprot_res_name', 'details', 'in_align']
    sequence_differences = []
    pdb_id = block.find('_entry.', ['id'])[0][0]

    # Extracting _entity_poly information
    for item in block.find('_entity_poly.', ['entity_id', 'pdbx_strand_id', 'pdbx_seq_one_letter_code', 'pdbx_seq_one_letter_code_can', 'nstd_linkage', 'nstd_monomer']):
        chain_ids = item[1].split(",")
        for chain_id in chain_ids:
            entity_to_chain[item[0]].append(chain_id)
            if chains is None or chain_id in chains:
                entity_id = item[0]
                sequence = item[2].replace('\n', '').replace(';', '')  # Adjust sequence formatting
                canonical_sequence = item[3].replace('\n', '').replace(';', '')
                nstd_linkage = item[4]
                nstd_monomer = item[5]
                chain_meta_data[chain_id]['chain_id'] = mmcif_corrector(chain_id, str) 
                chain_meta_data[chain_id]['pdb_id'] = mmcif_corrector(pdb_id, str)
                chain_meta_data[chain_id]['entity_id'] = mmcif_corrector(entity_id, int)
                chain_meta_data[chain_id]['pdbx_seq_one_letter_code'] = mmcif_corrector(sequence, str)
                chain_meta_data[chain_id]['pdbx_seq_one_letter_code_can'] = mmcif_corrector(canonical_sequence, str)
                chain_meta_data[chain_id]['nstd_linkage'] = mmcif_corrector(nstd_linkage, bool)
                chain_meta_data[chain_id]['nstd_monomer'] = mmcif_corrector(nstd_monomer, bool)
        
    # Extracting reference sequence information
    struct_ref_table = block.find('_struct_ref.', ['db_name', 'db_code', 'entity_id', 'pdbx_db_accession', 'pdbx_seq_one_letter_code'])
    pdb_chains = struct_ref_table.find_column('entity_id')
    counter = Counter(pdb_chains)
    entity_to_skip = []
    for counter_key, counter_value in counter.items():
        if counter_value > 1:
            entity_to_skip.append(counter_key)
    entity_to_skip = set(entity_to_skip)
    chain_to_skip = []
    for key, value in entity_to_chain.items():
        if key in entity_to_skip:
            chain_to_skip += value
        
    for row in struct_ref_table:
        entity_id = row[2]
        if entity_id in entity_to_skip: continue
        if row[0] != 'UNP': continue
        db_code = row[1]
        uniprot_id = row[3]
        reference_sequence = row[4]
        uniprot_meta_data[uniprot_id] = {'db_code': db_code, 'reference_sequence': reference_sequence}

    # Extracting _struct_ref_seq category for sequence alignment
    for row in block.find('_struct_ref_seq.', ['pdbx_strand_id', 'pdbx_db_accession', 'db_align_beg', 'db_align_end', 'pdbx_auth_seq_align_beg', 'pdbx_auth_seq_align_end']):
        chain_ids = row[0].split(",")
        for chain_id in chain_ids:
            if chains is None or chain_id in chains:
                uniprot_id = row[1]
                if uniprot_id not in uniprot_meta_data:
                    reference_sequence  = None
                    uniprot_name = None
                    uniprot_id = None
                    uniprot_align_beg = None
                    uniprot_align_end = None
                    if chain_id in chain_to_skip:
                        pdb_align_beg = 1
                        pdb_align_end = len(chain_meta_data[chain_id]['pdbx_seq_one_letter_code'])
                    else:
                        pdb_align_beg = mmcif_corrector(row[4], int)
                        pdb_align_end = mmcif_corrector(row[5], int)
                else:
                    reference_sequence = mmcif_corrector(uniprot_meta_data.get(uniprot_id).get('reference_sequence').replace('\n', '').replace(';', ''), str)
                    uniprot_name = mmcif_corrector(uniprot_meta_data.get(uniprot_id).get('db_code'), str)
                    uniprot_id = mmcif_corrector(uniprot_id, str)
                    pdb_align_beg = mmcif_corrector(row[4], int)
                    pdb_align_end = mmcif_corrector(row[5], int)
                    uniprot_align_beg = mmcif_corrector(row[2], int)
                    uniprot_align_end = mmcif_corrector(row[3], int)
                chain_meta_data[chain_id]['uniprot_id'] = uniprot_id
                chain_meta_data[chain_id]['uniprot_name'] = uniprot_name
                chain_meta_data[chain_id]['ref_seq_one_letter'] = reference_sequence
                chain_meta_data[chain_id]['pdb_seq_align_beg'] = pdb_align_beg
                chain_meta_data[chain_id]['pdb_seq_align_end'] = pdb_align_end
                chain_meta_data[chain_id]['uniprot_seq_align_beg'] = uniprot_align_beg
                chain_meta_data[chain_id]['uniprot_seq_align_end'] = uniprot_align_end
    
    for key, value in chain_meta_data.items():
        alignment_data.append([value.get('pdb_id'), key, value.get('entity_id'), value.get('pdbx_seq_one_letter_code'), 
                               value.get('pdbx_seq_one_letter_code_can'), value.get('uniprot_id'), value.get('uniprot_name'), 
                               value.get('ref_seq_one_letter'), value.get('pdb_seq_align_beg'), value.get('pdb_seq_align_end'), 
                               value.get('uniprot_seq_align_beg'), value.get('uniprot_seq_align_end'), value.get('nstd_linkage'), value.get('nstd_monomer')])

    # Extracting _struct_ref_seq_dif category for sequence differences
    for row in block.find('_struct_ref_seq_dif.', ['pdbx_pdb_strand_id', 'mon_id', 'db_mon_id', 'pdbx_seq_db_seq_num', 'details', 'pdbx_auth_seq_num']):
        chain_id = row[0]
        if chains is None or chain_id in chains:
            pdb_res_name = mmcif_corrector(row[1], str)
            uniprot_res_name = mmcif_corrector(row[2], str)
            uniprot_res_num = mmcif_corrector(row[3], int)
            pdb_res_num = mmcif_corrector(row[5], int)
            details = mmcif_corrector(row[4], str)
            if pdb_res_num and chain_meta_data[chain_id]['pdb_seq_align_beg'] <= pdb_res_num and pdb_res_num <= chain_meta_data[chain_id]['pdb_seq_align_end']:
                sequence_differences.append([pdb_id, chain_id, pdb_res_num, pdb_res_name, uniprot_res_num, uniprot_res_name, details, True])
            else:
                sequence_differences.append([pdb_id, chain_id, pdb_res_num, pdb_res_name, uniprot_res_num, uniprot_res_name, details, False])
                
    # Extracting _pdbx_poly_seq_scheme category for sequence resid mapping
    residue_num_mapping = defaultdict(dict)
    poly_residues = []
    for row in block.find('_pdbx_poly_seq_scheme.', ['seq_id', 'mon_id', 'pdb_seq_num', 'pdb_ins_code', 'pdb_strand_id', 'auth_mon_id']):
        chain_id = row[4]
        if chains is None or chain_id in chains:
            res_name = mmcif_corrector(row[1], str)
            seq_id = mmcif_corrector(row[0], int)
            res_num = mmcif_corrector(row[2], str)
            ins_code = mmcif_corrector(row[3], str) if row[3]!='.' else ''
            resnum_total = f"{res_num}{ins_code}"
            residue_num_mapping[chain_id][seq_id] = resnum_total
            if mmcif_corrector(row[-1], str) is not None:
                poly_residues.append((chain_id, resnum_total, res_name))
    
    # Extracting _pdbx_nonpoly_scheme for asymid and strand id mapping
    nonpoly_maaping = {}
    for row in block.find('_pdbx_nonpoly_scheme.', ['asym_id', 'mon_id', 'pdb_strand_id', 'pdb_seq_num', 'auth_seq_num', 'pdb_ins_code']):
        asym_id = mmcif_corrector(row[0], str)
        pdb_strand_id = mmcif_corrector(row[2], str)
        pdb_seq_num = mmcif_corrector(row[3], int)
        auth_seq_num = mmcif_corrector(row[4], int)
        ins_code = mmcif_corrector(row[5], str) if row[5] != '.' else ''
        nonpoly_maaping[(pdb_strand_id, pdb_seq_num, ins_code)] = (asym_id, auth_seq_num)
    
    # find prd_id
    prd_data = {}
    for row in block.find('_pdbx_molecule_features.', ['prd_id', 'name', 'type', 'class', 'details']):
        prd_data['id'] = mmcif_corrector(row[0], str)
        prd_data['name'] = mmcif_corrector(row[1], str)
        prd_data['type'] = mmcif_corrector(row[2], str)
        prd_data['class'] = mmcif_corrector(row[3], str)
        prd_data['details'] = mmcif_corrector(row[4], str)
        
    # Create DataFrames
    differences_df = pd.DataFrame(sequence_differences, columns=diff_columns)
    alignment_df = pd.DataFrame(alignment_data, columns=columns)

    return alignment_df, differences_df, residue_num_mapping, nonpoly_maaping, prd_data, poly_residues


def extract_pdb_information(pdb_file):
    # Define the headers we are interested in
    headers_of_interest = {"REMARK 465", "REMARK 470", "SEQRES", "MODRES", "SSBOND"}
    lines = defaultdict(list)
    interchain_ss = defaultdict(list)
    modres_info = {}
    connect = {}
    with open(pdb_file, 'r') as file:
        for line in file:
            # Check if the line starts with one of the headers of interest
            if any(line.startswith(header) for header in headers_of_interest):
                if line.startswith("REMARK"):
                    lines[line[:10]].append(line.strip())
                else:
                    lines[line[:6]].append(line.strip())
            if line.startswith("SSBOND") and len(line) > 29:
                chain1 = line[15]
                chain2 = line[29]
                if chain1 != chain2:
                    interchain_ss[chain1].append(chain2)
                    interchain_ss[chain2].append(chain1)
            if line.startswith("MODRES") and len(line) > 11:
                chain = line[16]
                res_name = line[12:15].strip()
                res_num = int(line[18:22])
                std_name = line[24:27].strip()
                icode = line[22]
                if res_name in standardResidues:
                    continue
                if std_name not in standardResidues:
                    continue
                modres_info[(chain, res_num, icode, res_name)] = std_name
            if line.startswith("CONECT"):
                atom1 = int(line[6:11].strip())
                atoms = connect.get(atom1, [])
                atoms.append(int(line[11:16].strip()))
                if len(line) > 16 and line[16:21].strip():
                    atoms.append(int(line[16:21].strip()))
                if len(line) > 21 and line[21:26].strip():
                    atoms.append(int(line[21:26].strip()))
                if len(line) > 26 and line[26:31].strip():
                    atoms.append(int(line[26:31].strip()))
                connect[atom1] = atoms
    return lines, interchain_ss, modres_info, connect


def extract_chain_specific_information(lines, chain_list):
    chain_info = []
    ssbond_lines = []
    for field, line in lines.items():
        if field == 'REMARK 465':
            chain_info.extend(line[:7])
            for l in line[7:]:
                if l[19] in chain_list:
                    chain_info.append(l)
        elif field == 'REMARK 470':
            chain_info.extend(line[:6])
            for l in line[6:]:
                if l[19] in chain_list:
                    chain_info.append(l)
        elif field == 'SEQRES':
            for l in line:
                if l[11] in chain_list:
                    chain_info.append(l)
        elif field == 'MODRES':
            for l in line:
                if l[16] in chain_list:
                    chain_info.append(l)
        elif field == 'SSBOND':
            for l in line:
                if l[15] in chain_list or l[29] in chain_list:
                    chain_info.append(l)
                    ssbond_lines.append(l)
    return chain_info, ssbond_lines


def create_residue_connection_graph(topology: app.Topology):
    graph = nx.Graph()
    graph.add_nodes_from([res for res in topology.residues()])

    non_metals = {
        'H', 'He', 
        'B', 'C', 'N', 'O', 'F', 'Ne', 
        'Si', 'P', 'S', 'Cl', 'Ar',
        'As', 'Se', 'Br', 'Kr',
        'Te', 'I', 'Xe', 'Rn'
    }
    metal_residues = set()
    for residue in topology.residues():
        if len(residue) == 1 and list(residue.atoms())[0].element.symbol not in non_metals:
            metal_residues.add(residue)
    for bond in topology.bonds():
        res1, res2 = bond.atom1.residue, bond.atom2.residue
        # don't include metal bonds
        if (res1 in metal_residues) or (res2 in metal_residues):
            continue
        graph.add_edge(res1, res2)
    return graph


def find_connected_residues(graph, residue):
    find = None
    for subgraph in nx.connected_components(graph):
        if residue in subgraph:
            find = subgraph
            break
    return find


def find_ligand_residues(topology: app.Topology, ligand_chain: str, ligand_residue_numbers: List, max_num_residues: int = 20, find_connected: bool = True, enforce_connected: bool = False):
    """
    Identify and return a list of ligand residues from a specified chain in a molecular topology.

    This function searches for specified residues in the given `ligand_chain` of a molecular topology 
    (represented by an OpenMM `Topology` object). It can optionally find residues that are connected 
    to the specified ligand residues, and enforce that all provided residues are connected in the topology.

    
    Parameters
    ----------
    topology : app.Topology
        The OpenMM Topology object representing the molecular system.
    ligand_chain : str
        The identifier of the chain that contains the ligand residues.
    ligand_residue_numbers : List[int] or List[str]
        A list of residue numbers for the ligand that should be identified.
    max_num_residues : int, optional
        The maximum number of ligand residues that can be returned (default is 20).
    find_connected : bool, optional
        If True, attempt to find and include residues connected to the specified ligand residues (default is True).
    enforce_connected : bool, optional
        If True, ensure that all specified residues are connected in the topology; raises an assertion error otherwise (default is False).
        Only be used when `find_connected` is True.

    Returns
    -------
    residues: List[app.Residue]
        A list of residues that match the specified ligand residue numbers or are connected to them.

    Raises
    ------
    AssertionError:
        If the specified residues do not exist in the topology or if the number of identified residues exceeds `max_num_residues`.
        If `enforce_connected` is True and the residues are not all connected.

    """
    ligand_residues_number_str = [str(x) for x in ligand_residue_numbers]
    graph = create_residue_connection_graph(topology)
    
    residues = set()
    for chain in topology.chains():
        if chain.id != ligand_chain:
            continue
        for residue in chain.residues():
            resid = f'{residue.id}{residue.insertionCode}'.strip()
            if resid in ligand_residues_number_str:
                if find_connected:
                    candidate = find_connected_residues(graph, residue)
                    if len(residues) == 0:
                        residues = candidate
                    else:
                        if enforce_connected:
                            assert candidate == residues, "Provided residues are not connected"
                        else:
                            residues = residues.union(candidate)
                else:
                    residues.add(residue)
    assert len(residues) > 0, "Provided residues don't exist"
    if max_num_residues is not None:
        assert len(residues) <= max_num_residues, f"Number of ligand residues ({len(residues)}) exceed limit ({max_num_residues})"
    return list(residues)


def process_everything(
    pdb_id: str, 
    ligand_id: Union[None, str, List[str]] = None, 
    ligand_info: Optional[List[Tuple[str, List[int]]]] = None, 
    dataset_dir: Optional[os.PathLike] = '../raw_data',
    binding_cutoff: float = BINDING_CUTOFF, 
    hetatm_cutoff: float = HETATM_CUTOFF,
    find_connected_ligand_residues: bool = True,
):
    """
    Run process workflow
    """

    # Create folder
    if not os.path.isdir(dataset_dir):
        os.path.mkdir(dataset_dir)
    folder = os.path.join(dataset_dir, pdb_id)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    download_pdb_cif(pdb_id, folder)
    get_rcsb_data(pdb_id, os.path.join(folder, 'rcsb_data.json'))
    
    pdb_file = os.path.join(folder, f'{pdb_id}.pdb')
    cif_file = os.path.join(folder, f'{pdb_id}.cif')

    # read in the key properties from the original pdb and cif file
    key_properties, interchain_ss, modres_info, connect = extract_pdb_information(pdb_file)
    alignment_info, diff_info, res_num_mapping, nonpoly_mapping, prd_data, poly_residues_cif = get_cif_header(cif_file)
    sequences = {}
    for _, row in alignment_info.iterrows():
        sequences[row['chain_id']] = convert_to_three_letter_seq(row['pdbx_seq_one_letter_code'])
    with open(os.path.join(folder, 'res_num_mapping.json'), 'w') as f:
        json.dump(res_num_mapping, f, indent=4)
    
    # OpenMM will change residue namings
    app.PDBFile._loadNameReplacementTables()
    app.PDBFile._residueNameReplacements = {k:v for k, v in app.PDBFile._residueNameReplacements.items() if k == v}
    pdb_omm = app.PDBFile(pdb_file)
    struct = Structure(pdb_omm.topology, pdb_omm.positions)
    
    # Find ligand residues
    if ligand_info is None:
        ligand_info = []
        if ligand_id:
            ligand_id = ligand_id if isinstance(ligand_id, list) else [ligand_id]
            for residue in struct.topology.residues():
                if residue.name in ligand_id:
                    ligand_info.append([residue.chain.id, [f'{residue.id}{residue.insertionCode}'.strip()]])
        else:
            # Use for polymer ligand, such as peptides, polysaccharides
            # Try to use sequence information
            for chainId, seq in sequences.items():
                if len(seq) > MAX_NUM_RES_POLY or len(seq) < MIN_NUM_RES_POLY:
                    continue
                resnums = []
                for residue in struct.topology.residues():
                    if (residue.chain.id == chainId) and (residue.name in seq):
                        resnums.append(f'{residue.id}{residue.insertionCode}'.strip())
                ligand_info.append([chainId, resnums])
            
            if len(ligand_info) == 0:
                graph = create_residue_connection_graph(struct.topology)
                for subgraph in nx.connected_components(graph):
                    if len(subgraph) > MAX_NUM_RES_POLY or len(subgraph) < MIN_NUM_RES_POLY:
                        continue
                    residues = list(subgraph)
                    resnums = [f'{r.id}{r.insertionCode}'.strip() for r in residues]
                    ligand_info.append([residues[0].chain.id, resnums])
    assert len(ligand_info) > 0, "No ligands found"

    ligand_residues_list = []
    for chain, residue_numbers in ligand_info:
        ligand_residues = find_ligand_residues(struct.topology, chain, residue_numbers, max_num_residues=MAX_NUM_RES_POLY, find_connected=find_connected_ligand_residues)
        ligand_residues_list.append(ligand_residues)
    
    
    # Classifiy residues to polymer, ligand and hetatoms
    polymer_residues_info = []
    for chain in res_num_mapping:
        for value in res_num_mapping[chain].values():
            polymer_residues_info.append((chain, value))

    polymer_residues = defaultdict(list)
    hetero_residues = []
    for residue in struct.topology.residues():
        if any([residue in ligand_residues for ligand_residues in ligand_residues_list]):
            continue
        if (residue.chain.id, f'{residue.id}{residue.insertionCode}'.strip(), residue.name) in poly_residues_cif:
            polymer_residues[residue.chain.id].append(residue)
        else:
            hetero_residues.append(residue)

    all_chains_to_include = set()
    # Find residues to include and save to PDB files
    for ligand_residues in ligand_residues_list:
        # Rare elements - raise an error
        ligand_heavy_atoms = []
        for residue in ligand_residues:
            for atom in residue.atoms():
                if atom.element.symbol not in COMMON_ELEMENTS:
                    raise RuntimeError(f'Rare element in ligand: {atom.element.symbol}')
                if atom.element.symbol != 'H':
                    ligand_heavy_atoms.append(atom)
        if len(ligand_heavy_atoms) < MAX_HEAVY_ATOMS:
            raise RuntimeError(f"Too few #heavy atoms in ligand: {ligand_heavy_atoms}")
                
        # Get some info
        ligand_residues.sort(key=lambda res: res.index)
        ligand_chain = ligand_residues[0].chain.id
        if len(ligand_residues) == 1:
            ligand_name = ligand_residues[0].name
            ligand_number = f"{ligand_residues[0].id}{ligand_residues[0].insertionCode}".strip()
        else:
            # Handle polymers     
            ligand_name = f'{ligand_residues[0].name}-{ligand_residues[-1].name}'
            ligand_number =  f"{ligand_residues[0].id}{ligand_residues[0].insertionCode}".strip() + '-' + f"{ligand_residues[-1].id}{ligand_residues[-1].insertionCode}".strip()

        basename = f"{pdb_id}_{ligand_name}_{ligand_chain}_{ligand_number}"
        subfolder = os.path.join(folder, basename)
        if not os.path.isdir(subfolder): os.mkdir(subfolder)
        
        ligand_positions = struct.get_positions_by_atoms(ligand_heavy_atoms)
        include = defaultdict(list)

        # Proteins chains that are within `binding_cutoff`
        chains_include = set()
        for chain_id in polymer_residues.keys():
            if chain_id in chains_include:
                continue

            # Get positions of heavy atoms
            chain_heavy_atoms = []
            for residue in polymer_residues[chain_id]:
                for atom in residue.atoms():
                    if atom.element.symbol != 'H':
                        chain_heavy_atoms.append(atom)
            chain_positions = struct.get_positions_by_atoms(chain_heavy_atoms)

            dist_mat = cdist(ligand_positions, chain_positions)
            min_dist = np.min(dist_mat) * 10
            argmin = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            assert min_dist >= STERIC_CLASH_THRESH, f"Steric clash (dist. {min_dist:.2f}) observed between {ligand_heavy_atoms[argmin[0]]} and {chain_heavy_atoms[argmin[1]]}"

            if min_dist < binding_cutoff:
                chains_include.add(chain_id)
                # include chains that are ssbond connected
                chains_include = chains_include.union(interchain_ss.get(chain_id, set()))
        all_chains_to_include.update(chains_include)
        
        for chain_id in chains_include:
            include['polymer'] += polymer_residues[chain_id]
        
        # HETATM that are within `hetatm_cutoff`
        positions = struct.get_positions_by_residues(ligand_residues + include['polymer'])
        for residue in hetero_residues:
            het_positions = struct.get_positions_by_residues([residue])
            if np.min(cdist(positions, het_positions)) * 10 < hetatm_cutoff:
                include['hetatm'].append(residue)
        # Record ligand
        include['ligand'] = ligand_residues
        
        # ligand
        ligand_pdb = os.path.join(subfolder, f'{basename}_ligand.pdb')
        struct.select_residues(include['ligand']).save(ligand_pdb)

        if len(ligand_residues) == 1:
            query_id = ligand_name
            asym_id, auth_seq_id = nonpoly_mapping[(ligand_chain, int(ligand_residues[0].id), ligand_residues[0].insertionCode.strip())]
            sdf = download_ligand_sdf(
                pdb_id, ligand_name, asym_id, auth_seq_id, 
                subfolder, 
                basename=f"{basename}_ligand_rcsb.sdf", 
                raise_error=False,
                overwrite=False
            )
            # Bad sdf
            try:
                mol = Chem.SDMolSupplier(sdf, sanitize=False)[0]
                assert mol is not None
            except:
                if os.path.isfile(sdf): os.remove(sdf)
        elif prd_data:
            query_id = prd_data['id']
        else:
            query_id = None
        
        ref_smi, ref_name = get_reference_smi(pdb_id, query_id)
        if not ref_smi:
            # Generate reference smiles from squence
            try:
                seq = sequences[ligand_residues[0].chain.id]
                if MIN_NUM_RES_POLY < len(seq) <= MAX_NUM_RES_POLY:
                    ref_mol = mol_from_seq(seq)
                    ref_smi = Chem.MolToSmiles(ref_mol)
                    ref_name = 'seq:' + ','.join(seq)
            except:
                ref_smi = None

        if ref_smi:
            with open(os.path.join(subfolder, f'ref.smi'), 'w') as f:
                f.write(ref_smi + ' ' + ref_name)
        
        # convert PDB to SDF
        ligand_sdf = os.path.join(subfolder, f'{basename}_ligand.sdf')
        mol = read_by_obabel(ligand_pdb)
        if mol is not None:
            mol.SetProp("Origin", "OpenBabel")
        else:
            mol = Chem.MolFromPDBFile(ligand_pdb)
            mol.SetProp("Origin", 'RDKit')
        write_sdf(mol, ligand_sdf)
        
        # protein only
        seqres = []
        for chain in chains_include:
            if chain in sequences:
                seqres.append(convert_to_seqres(sequences[chain], chain))
        # modified residues
        modres_list = []
        for modres, stdname in modres_info.items():
            chain, resnum, icode, resname = modres
            if chain in chains_include:
                resnum = [k for k, v in res_num_mapping[chain].items() if v == f'{resnum}{icode}'.strip()][0]
                modres_list.append((chain, resnum, icode, resname, stdname))
        modres = StandardizedPDBFixer.getModresRecords(modres_list, pdb_id=pdb_id)

        protein_pdb = os.path.join(subfolder, f'{basename}_protein.pdb')
        struct.select_residues(include['polymer']).save(
            protein_pdb, 
            header='\n'.join(seqres + modres), 
            res_num_mapping=res_num_mapping
        )
        # hetetro atoms
        hetatm_pdb = os.path.join(subfolder, f'{basename}_hetatm.pdb')
        struct.select_residues(include['hetatm']).save(hetatm_pdb)
        
        # all 
        chain_properties, ssbond_lines = extract_chain_specific_information(key_properties, chains_include)
        all_pdb = os.path.join(subfolder, f'{basename}_protein_hetatm.pdb')
        struct.select_residues(include['polymer'] + include['hetatm']).save(all_pdb, header='\n'.join(chain_properties))
    
    # Record alignment info
    alignment_info = alignment_info[alignment_info['chain_id'].isin(all_chains_to_include)]
    diff_info = diff_info[diff_info['chain_id'].isin(all_chains_to_include)]
    alignment_info.to_csv(os.path.join(folder, 'alignment_info.csv'), index=None)
    diff_info.to_csv(os.path.join(folder, 'diff_info.csv'), index=None)

    fix_ligands_in_folder(folder)
    refine_structure_with_ligand(folder)

    fp = open(os.path.join(folder, 'done.tag'), 'w')
    fp.close()


def refine_structure_with_ligand(folder):
    pdb_id = os.path.basename(folder)
    res_num_mapping = {}
    with open(os.path.join(folder, 'res_num_mapping.json')) as f:
        for chain, mapping in json.load(f).items():
            res_num_mapping[chain] = {int(k): v for k, v in mapping.items()}
    
    for protein_pdb in glob.glob(os.path.join(folder, '*/*_protein.pdb')):
        ligand_sdf = protein_pdb.replace("_protein.pdb", "_ligand_fixed.sdf")
        if VERBOSE:
            print(f'Processing {protein_pdb}')
        fixer = StandardizedPDBFixer(protein_pdb=protein_pdb, ligand_sdf=ligand_sdf, pdb_id=pdb_id, verbose=False)
        fixer.runFixWorkflow(
            output_protein=protein_pdb.replace("_protein.pdb", "_protein_refined.pdb"),
            output_ligand=ligand_sdf.replace("_fixed.sdf", "_refined.sdf"),
            res_num_mapping=res_num_mapping,
            refine_positions=True,
            skip_long_missing_residues=MAX_ADD_MISSING_RES
        )


def fix_ligands_in_folder(folder):
    # Fix Ligands
    err_log = []
    for ligand_pdb in glob.glob(os.path.join(folder, '*/*_ligand.pdb')):
        smi_file = os.path.join(os.path.dirname(ligand_pdb), 'ref.smi')
        if not os.path.isfile(smi_file):
            ref_smi = None
        else:
            with open(smi_file) as f:
                ref_smi = f.read().split()[0]
        name = os.path.basename(ligand_pdb)[:-11]
        out_sdf = ligand_pdb.replace("_ligand.pdb", "_ligand_fixed.sdf")
        
        err_msg = ""
        try:
            sdf = ligand_pdb.replace("_ligand.pdb", "_ligand.sdf")
            fix_ligand(sdf, ref_smi, out_sdf, name)
        except Exception as err:
            sdf = ligand_pdb.replace("_ligand.pdb", "_ligand_rcsb.sdf")
            err_msg = traceback.format_exc()
            if os.path.isfile(sdf):
                try:
                    fix_ligand(sdf, ref_smi, out_sdf, name)
                    err_msg = ''
                except Exception as err:
                    err_msg = traceback.format_exc()
        if err_msg:
            err_log.append((name, err_msg))
    
    err_msg = '----\n'.join(f'\nError occurs when fixing {name}: \n{err_msg}' for name, err_msg in err_log)
    if err_msg:
        raise LigandFixException(err_msg)
        

if __name__ == "__main__":
    import warnings
    import shutil
    warnings.filterwarnings("ignore")
    import pandas as pd
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', help='input csv')
    parser.add_argument('-d', '--output', dest='output', help='output directory')
    parser.add_argument('--poly', dest='poly', action='store_true', help='if polymer csv')
    parser.add_argument('--serial', dest='serial', action='store_true', help='if run serially (not in parallel)')
    input_args = parser.parse_args()

    dataset_dir = input_args.output
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    def wrap_process_wf(args):
        pdbid, ligand_ccd, ligand_info = args
        try:
            process_everything(pdbid, ligand_ccd, ligand_info, dataset_dir)
        except Exception as e:
            # raise e
            errmsg = traceback.format_exc()
            with open(os.path.join(dataset_dir, f'{pdbid}/err'), 'w') as f:
                f.write(errmsg)

    # Read CSV
    df = pd.read_csv(input_args.input, dtype=str)
    
    if not input_args.poly:
        args = []
        for pdbid, subdf in df.groupby('PDBID'):
            ligand_info, ligand_ccd = [], None
            for _, row in subdf.iterrows():
                chain, resnum = row['Ligand chain'], row['Ligand residue sequence number']
                if str(chain) == 'nan' or str(resnum) == 'nan':
                    continue
                ligand_info.append((chain, [resnum]))
            
            if len(ligand_info) == 0:
                ligand_info = None
                ligand_ccd = subdf['Ligand CCD'].unique().tolist()
            arg = (pdbid, ligand_ccd, ligand_info)
            args.append(arg)
    else:
        args = [(pdbid, None, None) for pdbid in df['PDBID'].unique()]

    # done.tag / err exists means they have already been processed
    newargs = []
    for arg in args:
        wdir = os.path.join(dataset_dir, arg[0])
        if os.path.isfile(os.path.join(wdir, 'done.tag')) or os.path.isfile(os.path.join(wdir, 'err')):
            continue
        newargs.append(arg)
    args = newargs

    for arg in args:
        if os.path.isdir(os.path.join(dataset_dir, arg[0])):
            shutil.rmtree(os.path.join(dataset_dir, arg[0]))

    use_mpi = (not input_args.serial)
    if use_mpi:
        num_proc = 64
        chunksize = 1
        with mp.Pool(num_proc) as p:
            results = list(tqdm(p.imap_unordered(wrap_process_wf, args, chunksize=chunksize), total=len(args)))
    else:
        for arg in tqdm(args):
            wrap_process_wf(arg)
