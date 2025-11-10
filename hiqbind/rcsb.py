'''
Author: Oliver Sun, Eric Wang
Date: 09/02/2024

This file contains factory functions related to query RCSB database
'''
import os
import re
import requests
import json
from bs4 import BeautifulSoup
from rdkit import Chem


def download_file_request(url, local_filename):
    # Send a GET request to the URL
    with requests.get(url, stream=True) as r:
        # Raise an error if the request was unsuccessful
        r.raise_for_status()
        # Open a local file in binary write mode
        with open(local_filename, 'wb') as f:
            # Write the file in chunks to avoid using too much memory
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def download_file(url: str, fp: os.PathLike, overwrite: bool = False, raise_error: bool = True):
    """
    Download file from given URL

    Parameters
    ----------
    url: str
        URL of the file to be downloaded
    fp: os.PathLike
        Local path to the downloaded file
    overwrite: bool
        If True, will overwrite the exisiting file. 
        If False, will skip the download process if the file exists. Default False.
    raise_error: bool
        If True, will raise error if the download fails. Default True.
    """
    if overwrite or (not os.path.isfile(fp)):
        try:
            download_file_request(url, fp)
            # Eric 09/04 - on Nersc neither of these works fine with multiprocessing, sometimes it will just download an empty file
            # urllib.request.urlretrieve(url, fp)
            # wget.download(url, fp, bar=None)
            # subprocess.run(['wget', '-q', '-O', fp, url])
            return True
        except Exception as e:
            if raise_error:
                raise e
            else:
                msg = f"Fail to download {url}. Error: {e}"


def download_pdb_cif(pdb_id: str, folder: os.PathLike, overwrite: bool = False, raise_error: bool = True):
    """
    Download a PDB & CIF file from RCSB.

    Parameters
    ----------
    pdb_id: str
        The PDB ID of the file to download.
    folder: os.PathLike 
        The path to the folder where the file will be saved.
    overwrite: bool
        If True, will overwrite the exisiting file. 
        If False, will skip the download process if the file exists. Default False.
    raise_error: bool
        If True, will raise error if the download fails. Default True.
    """
    # URL for the PDB file (Replace with the base URL of your choice)
    url_cif = f"https://files.rcsb.org/download/{pdb_id}.cif"
    url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    download_file(url_cif, os.path.join(folder, f'{pdb_id}.cif'), overwrite, raise_error)
    download_file(url_pdb, os.path.join(folder, f'{pdb_id}.pdb'), overwrite, raise_error)


def get_smiles_from_rcsb(comp_id: str):
    """
    Query ligand SMILES from RCSB

    Parameters
    ----------
    comp_id: str
        The ligand ID, usually a three-letter code
    
    Returns
    -------
    smi: str
        The SMILES of the query ligand. If fail to get, will return a vacant string
    """
    query = '''{chem_comp(comp_id: "%s") {
        rcsb_chem_comp_descriptor {
        SMILES_stereo SMILES InChI
        }
    }
    }''' % comp_id
    query = re.sub(r'\s+', ' ', query)
    try:
        res = requests.get('https://data.rcsb.org/graphql?query=' + query)
        smi = res.json()['data']['chem_comp']['rcsb_chem_comp_descriptor']['SMILES_stereo']
        if smi is None:
            smi = res.json()['data']['chem_comp']['rcsb_chem_comp_descriptor']['SMILES']
        if smi is None:
            print("Inferred from InChI")
            m = Chem.MolFromInchi(res.json()['data']['chem_comp']['rcsb_chem_comp_descriptor']['InChI'])
            smi = Chem.MolToSmiles(m)
        assert smi is not None, "No reference smiles"
        return smi
    except:
        return ""


def download_ligand_sdf(
    pdb_id: str, 
    ligand_id: str, asym_id: str, auth_seq_num: int, 
    folder: os.PathLike,
    basename: str = "",
    overwrite: bool = False,
    raise_error: bool = True
):
    """
    Download a ligand SDF file using wget.

    Parameters
    ----------
    pdb_id: str
        The 4-letter PDB ID of the ligand.
    ligand_id: str
        The ligand ID, usually a 3-letter code. (_pdbx_nonpoly_scheme.mon_id in cif header)
    asym_id: str
        The `label_asym_id` (chain_id) of the ligand. (_pdbx_nonpoly_scheme.asym_id in cif header)
    auth_seq_num: int
        The `auth_seq_num` (residue number) of the ligand. (_pdbx_nonpoly_scheme.auth_seq_num in cif header)
    folder: os.PathLike
        Path to the folder where the file will be saved.
    basename: str
        Customized file name. If set to "", will use `{pdb_id}_{asym_id}_{ligand_id}.sdf`
    query_smiles: bool
        If query SMILES from RCSB
    overwrite: bool
        If True, will overwrite the exisiting file. 
        If False, will skip the download process if the file exists. Default False.
    raise_error: bool
        If True, will raise error if the download fails. Default True.
    """
    # URL for the SDF file
    url = f"https://models.rcsb.org/v1/{pdb_id}/ligand?auth_seq_id={auth_seq_num}&label_asym_id={asym_id}&encoding=sdf&filename={pdb_id}_{asym_id}_{ligand_id}.sdf"
    basename = f'{pdb_id}_{asym_id}_{ligand_id}.sdf' if not basename else basename
    fname = os.path.join(folder, basename)
    download_file(url, fname, overwrite, raise_error)
    return fname


with open(os.path.join(os.path.dirname(__file__), 'query.txt')) as f:
    QUERY = f.read()

def get_rcsb_data(pdb_id: str, output_path: str = "", raise_error: bool = True):
    """
    Query data  from RCSB

    Parameters
    ----------
    comp_id: str
        The ligand ID, usually a three-letter code
    
    Returns
    -------
    smi: str
        The SMILES of the query ligand. If fail to get, will return a vacant string
    """

    url = "https://data.rcsb.org/graphql"
    response = requests.post(url, json={'query': QUERY, 'variables': {"id": pdb_id}})
    
    if response.status_code == 200:
        jdata = response.json()
    else:
        if raise_error:
            raise Exception(f"Query failed with status code {response.status_code}: {response.text}")
        else:
            jdata = {}
    
    if jdata['data']['entry'] is None:
        print(f"{pdb_id} is not valid, check if this is superseded.")
        try:
            text = requests.get(f'https://www.rcsb.org/structure/removed/{pdb_id}').text
            soup = BeautifulSoup(text, 'html.parser')
            new_pdbid = soup.find(id='note_obsoletedBy').find('a').text.lower()
            print(f"{pdb_id} is superseded by {new_pdbid}.")
        except:
            new_pdbid = None
            raise Exception(f'The provided PDB ID is obsolete and no superseding PDB id is found.')
        
        if new_pdbid:
            jdata = get_rcsb_data(new_pdbid)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(jdata, f, indent=4)
    return jdata
    

if __name__ == '__main__':
    import os, glob, shutil
    import multiprocessing as mp
    from tqdm import tqdm
    import time
    from random import randint

    dirpath = '../raw_data_pdbbind_sm'

    def query_rcsb(pdbid):
        jfile = os.path.join(dirpath, pdbid, 'rcsb_data.json')
        # if os.path.isfile(jfile):
        #     os.remove(jfile)
        #     return (pdbid, True)

        if os.path.isfile(jfile):
            with open(jfile) as f:
                jdata = json.load(f)
            if jdata['data']['entry'] is not None:
                return (pdbid, True)
        
        try:
            get_rcsb_data(pdbid, jfile)
            time.sleep(randint(1, 5))
            return (pdbid, True)
        except:
            return (pdbid, False)

    pdbids = list(os.listdir(dirpath))
    with mp.Pool(16) as p:
        results = list(tqdm(p.imap_unordered(query_rcsb, pdbids), total=len(pdbids)))
    
    for r in results:
        if not r[1]:
            print(r[0])