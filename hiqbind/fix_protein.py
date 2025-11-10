import os
import warnings
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Union
import xml.etree.ElementTree as ET 
import xml.dom.minidom

from textwrap import wrap
from Bio.PDB import Select

import numpy as np
import openmm as mm 
import openmm.app as app
import openmm.unit as unit
from rdkit import Chem
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from pdbfixer import PDBFixer


aa_mapping = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR','V': 'VAL',
}


def convert_to_three_letter_seq(sequence: str):
    # Convert the one-letter code to three-letter code
    result = []
    i = 0
    while i < len(sequence):
        if sequence[i] == '(':
            # Find the closing parenthesis
            closing_index = sequence.find(')', i)
            if closing_index != -1:
                # Append the content inside the parentheses as is
                result.append(sequence[i+1:closing_index])
                i = closing_index + 1
            else:
                raise ValueError("Unmatched parenthesis in the sequence.")
        else:
            # Convert the single-letter code to three-letter code
            result.append(aa_mapping.get(sequence[i], 'XAA'))  # 'XAA' for unknown residues
            i += 1
    return result


def convert_to_seqres(sequence: List[str], chain_id: str):
    lines = wrap(' '.join([s.upper() for s in sequence]), 51)
    # Create the SEQRES lines
    seqres_lines = []
    for i, line in enumerate(lines):
        seqres_lines.append(f"SEQRES  {i+1: >2} {chain_id} {len(sequence): >4}  {line}")
    return '\n'.join(seqres_lines)


class Structure:
    def __init__(self, topology: app.Topology, positions: Union[unit.Quantity, np.ndarray], seqs: Dict[str, List[str]] = dict()):
        self.topology = topology
        self.positions = positions
        self.positions_numpy = np.array([[vec.x, vec.y, vec.z] for vec in self.positions], dtype=np.float32)
        self.seqs = seqs

    def get_positions_by_residues(self, residues):
        selected = []
        for residue in residues:
            for atom in residue.atoms():
                selected.append(atom.index)
        return self.positions_numpy[selected]
    
    def get_positions_by_atoms(self, atoms: List[app.Atom]):
        return self.positions_numpy[[atom.index for atom in atoms]]

    def select_residues(self, residues: List[app.Residue]):
        modeller = app.Modeller(self.topology, self.positions)
        to_delete = [res for res in self.topology.residues() if res not in residues]
        modeller.delete(to_delete)
        struct = Structure(modeller.topology, modeller.positions)
        return struct
    
    def select(self, select: Select):
        to_delete = []
        modeller = app.Modeller(self.topology, self.positions)
        for chain in modeller.topology.chains():
            if not select.accept_chain(chain):
                to_delete.append(chain)
                continue
            for residue in chain.residues():
                if not select.accept_residue(residue):
                    to_delete.append(residue)
                    continue
                for atom in residue.atoms():
                    if not select.accept_atom(atom):
                        to_delete.append(atom)
                        continue
        modeller.delete(to_delete)
        return Structure(modeller.topology, modeller.positions)

    def save(self, file, select: Optional[Select] = None, keepIds: bool = True, header: str = '', res_num_mapping: Optional[Dict[str, Dict[int, str]]] = None):
        if isinstance(file, str) or isinstance(file, os.PathLike):
            fp = open(file, 'w')
            self.save(fp, select, keepIds, header, res_num_mapping)
            fp.close()
        else:
            if select is not None:
                sub = self.select(select)
                top, pos = sub.topology, sub.positions
            else:
                top, pos = self.topology, self.positions

            if res_num_mapping:
                for residue in top.residues():
                    chain = residue.chain.id
                    new_res_num = [k for k, v in res_num_mapping[chain].items() if v == f"{residue.id}{residue.insertionCode}".strip()][0]
                    residue.id = str(new_res_num)
                    residue.insertionCode = " "
            
            app.PDBFile.writeHeader(top, file)
            if header:
                print(header[:-1] if header[-1] == '\n' else header, file=file)
            app.PDBFile.writeModel(top, pos, file, keepIds=keepIds)
            app.PDBFile.writeFooter(top, file)


def write_xml(element: ET.Element, fpath: os.PathLike):
    uglystr = ET.tostring(element, 'unicode', method='xml')
    pretxml = xml.dom.minidom.parseString(uglystr)
    pretstr = pretxml.toprettyxml(indent='  ').lstrip('<?xml version="1.0" ?>')
    with open(fpath, 'w') as f:
        f.write(pretstr)


def createPatch(residue, folder, amino_acid=True):
    atom_types = []
    atoms = []
    bonds = []
    nbforces = []

    for bond in residue.bonds():
        if (bond.atom1.residue is residue) and (bond.atom2.residue is residue):
            bonds.append({'atomName1': bond.atom1.name, 'atomName2': bond.atom2.name})
        elif amino_acid:
            tup = (bond.atom1.name, bond.atom2.name)
            assert tup == ('N', 'C') or tup == ('C', 'N'), f'Not a modified amino acid: {residue} with {bond}'
    
    for atom in residue.atoms():
        atype = f'{residue.name}-{atom.name}'
        ele = atom.element if atom.element is not None else app.Element.getBySymbol(atom.name[0])
        atom_types.append({'element': ele.symbol, 'name': atype, 'class': atype, 'mass': str(ele.mass._value)})
        atoms.append({'name': atom.name, 'type': atype, 'charge': '0.0'})
        nbforces.append({'type': atype, 'sigma': '0.2', 'epsilon': '0.2'})
    
    top_xml = os.path.join(folder, residue.name+'_top.xml')
    ff_xml = os.path.join(folder, residue.name+'.xml')
    
    ffElement = ET.Element('ForceField')
    atypesElement = ET.SubElement(ffElement, 'AtomTypes')
    for atype in atom_types:
        ET.SubElement(atypesElement, 'Type', attrib=atype)
    residuesElement = ET.SubElement(ffElement, 'Residues')

    for prefix in ['N', 'C', '']:
        residueElement = ET.SubElement(residuesElement, 'Residue', attrib={'name': prefix + residue.name})
        for atom in atoms:
            ET.SubElement(residueElement, 'Atom', attrib=atom)
        for bond in bonds:
            ET.SubElement(residueElement, 'Bond', attrib=bond)
        if prefix == 'N':
            ext = ['C']
        elif prefix == 'C':
            ext = ['N']
        else:
            ext = ['C', 'N']
        for name in ext:
            ET.SubElement(residueElement, 'ExternalBond', attrib={'atomName': name})

    nbElement = ET.SubElement(ffElement, 'NonbondedForce', attrib={'coulomb14scale': "0.8333333333333334", 'lj14scale': "0.5"})
    ET.SubElement(nbElement, 'UseAttributeFromResidue', attrib={'name': 'charge'})
    for nbf in nbforces:
        ET.SubElement(nbElement, 'Atom', attrib=nbf)
    write_xml(ffElement, ff_xml)

    topElement = ET.Element('Residues')
    resElement = ET.SubElement(topElement, 'Residue', attrib={'name': residue.name})
    if amino_acid:
        ET.SubElement(resElement, 'Bond', attrib={'from': '-C', 'to': 'N'})
    for bond in bonds:
        ET.SubElement(resElement, 'Bond', attrib={'from': bond['atomName1'], 'to': bond['atomName2']})
    write_xml(topElement, top_xml)
    return top_xml, ff_xml


def to_quantity(ndarray):
    value = [mm.Vec3(float(arr[0]), float(arr[1]), float(arr[2])) for arr in ndarray]
    quantity = unit.Quantity(value, unit=unit.nanometers)
    return quantity


class StandardizedPDBFixer(PDBFixer):
    """
    A class to fix standarized PDB file. Here the standardized means that:
    1. SEQRES record well documented in the PDB header
    2. PDB residues are numbered in `_pdbx_poly_seq_scheme.seq_id` with no insertion code. 
        This will make PDBFixer able to find all missing residues.
    """
    def __init__(self, protein_pdb: os.PathLike, ligand_sdf: Optional[os.PathLike] = None, verbose: bool = False, pdb_id: str = ""):
        super().__init__(filename=str(protein_pdb))
        
        if ligand_sdf:
            self.addLigand(ligand_sdf)
            self._has_ligand = True
        else:
            self._has_ligand = False
        
        self.mod_res_info = []
        self.missing_residues_added = []
        self.missing_residues_skipped = []
        self.missing_atoms_added = []
        self.verbose = verbose
        self.pdb_id = pdb_id
    
    def addLigand(self, ligand_sdf: os.PathLike):
        # Build ligand force field
        mol = Chem.SDMolSupplier(ligand_sdf, removeHs=False)[0]
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        self.ligand_missing_atoms = [int(x) for x in mol.GetPropsAsDict().get('Missing Atoms', '')]
        self.rdmol = mol
        
        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
        off_mol.assign_partial_charges('gasteiger')
        self.off_mol = off_mol
        # self.ligand_ff_generator = GAFFTemplateGenerator(molecules=[off_mol], forcefield='gaff-2.11')
        # self.ligand_ff_generator = SMIRNOFFTemplateGenerator(molecule=[off_mol])

        self.ligand_pos = to_quantity(mol.GetConformer().GetPositions() / 10)
        self.ligand_top = Topology.from_molecules(off_mol).to_openmm()
        modeller = app.Modeller(self.topology, self.positions)
        modeller.add(self.ligand_top, self.ligand_pos)
        self.topology = modeller.getTopology()
        self.positions = modeller.getPositions()

    def log(self, *args):
        if self.verbose:
            print(*args)
    
    def findNonstandardResidues(self):
        # Find non-std residues
        super().findNonstandardResidues()
        for residue, std_res_name in self.nonstandardResidues:
            self.mod_res_info.append(
                (residue.chain.id, int(residue.id), residue.insertionCode, residue.name, std_res_name)
            )

    def findMissingResidues(self, skip_terminals: bool = True, skip_long_residues: Optional[int] = None):
        sequences = {seq.chainId: seq.residues for seq in self.sequences}
        chains = list(self.topology.chains())
        
        # Find missing residues
        super().findMissingResidues()
        num_missing_residues = {chain.id: 0 for chain in chains}
        item_to_pop = []
        for info, residues in self.missingResidues.items():
            chain_index, res_id_start = info
            chain_id = chains[chain_index].id
            is_ter = (res_id_start == 0) or (num_missing_residues[chain_id] + len(residues) + res_id_start == len(sequences[chain_id]))
            offset = res_id_start + num_missing_residues[chain_id] + 1
            if (is_ter and skip_terminals) or (isinstance(skip_long_residues, int) and len(residues) > skip_long_residues):
                item_to_pop.append(info)
                for i, residue in enumerate(residues):
                    self.missing_residues_skipped.append((chain_id, offset + i, residue))
            else:
                for i, residue in enumerate(residues):
                    self.missing_residues_added.append((chain_id, offset + i, residue))
            num_missing_residues[chain_id] += len(residues)

        for info in item_to_pop:
            self.missingResidues.pop(info)
        self.log('Add  missing residues:', self.missing_residues_added)
        self.log('Skip missing residues:', self.missing_residues_skipped)
        
    def findMissingAtoms(self, skip_terminals: bool = False):
        # Find missing atoms
        super().findMissingAtoms()
        if skip_terminals:
            self.missingTerminals = {}
        
        missingAtomNames = {residue: [at.name for at in atoms] for residue, atoms in self.missingAtoms.items()}
        for residue in self.missingTerminals:
            if residue not in missingAtomNames:
                missingAtomNames[residue] = self.missingTerminals[residue]
            else:
                missingAtomNames[residue] += self.missingTerminals[residue]

        for residue, atom_names in missingAtomNames.items():            
            self.missing_atoms_added.append(
                (residue.chain.id, int(residue.id), residue.name, atom_names)
            )

        self.log('Missing atoms:', self.missing_atoms_added)

    def addMissingAtoms(self, seed=None):
        return super().addMissingAtoms(seed)
    
    @classmethod
    def _getMappedResnumWithIcode(cls, res_id: int, chain: str, res_num_mapping: Dict[str, Dict[int, str]]) -> Tuple[str, str]:
        res_id_mapped = res_num_mapping[chain][res_id]
        insert_code = ' '
        if res_id_mapped[-1].isalpha():
            insert_code = res_id_mapped[-1]
            res_id_mapped = res_id_mapped[:-1]
        return res_id_mapped, insert_code


    @classmethod
    def getFixedResidueRemarks(cls, missing_residues_info: List[Tuple[str, int, str]], res_num_mapping=None, use_fixed_remark: bool = True):
        if len(missing_residues_info) == 0:
            return []
        
        remark_465_lines = [
            "REMARK 465",
            "REMARK 465 FIXED MISSING RESIDUES" if use_fixed_remark else  "REMARK 465 MISSING RESIDUES",
            "REMARK 465 (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN",
            "REMARK 465 IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)",
            "REMARK 465",
            "REMARK 465   M RES C SSSEQI"
        ]
        for chain, res_id, res_name in missing_residues_info:
            if not res_num_mapping:
                insert_code = ' '
            else:
                res_id, insert_code = cls._getMappedResnumWithIcode(res_id, chain, res_num_mapping)
            remark_465_lines.append(f"REMARK 465     {res_name:3} {chain:1}  {res_id:>4}{insert_code:>1}")
        return remark_465_lines
    
    @classmethod
    def getFixedAtomRemarks(cls, missing_atoms_info: List[Tuple[str, str, int, List[str]]], res_num_mapping=None):
        if len(missing_atoms_info) == 0:
            return []
        
        remark_470_lines = [
            "REMARK 470",
            "REMARK 470 FIXED MISSING ATOM",
            "REMARK 470 (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN",
            "REMARK 470 IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)",
            "REMARK 470",
            "REMARK 470   M RES CSSEQI  ATOMS"
        ]
        for chain, res_id, res_name, atoms in missing_atoms_info:
            if not res_num_mapping:
                insert_code = ' '
            else:
                res_id, insert_code = cls._getMappedResnumWithIcode(res_id, chain, res_num_mapping)
            atom_list = " ".join([f"{atom:<4}" for atom in atoms])
            remark_470_lines.append(f"REMARK 470     {res_name:3} {chain:1} {res_id:>4}{insert_code:>1}  {atom_list}")
        return remark_470_lines
    
    @classmethod
    def getModresRecords(cls, mod_res_info: List[Tuple[str, int, str, str, str]], res_num_mapping=None, pdb_id=""):
        pdb_id = 'XXXX' if not pdb_id else pdb_id.upper()
        modres_lines = []
        for chain, res_id, icode, res_name, std_res_name in mod_res_info:
            if res_num_mapping:
                res_id = res_num_mapping[chain][res_id]
                if res_id[-1].isalpha():
                    icode = res_id[-1]
                    res_id = res_id[:-1]
            # TODO: need to figure out how to deal with modfied residues with insertion code (Eric)
            modres_lines.append(f"MODRES {pdb_id:>4} {res_name:3} {chain:1} {res_id:>4}{icode:1} {std_res_name:3}   MODIFIED RESIDUE")
        return modres_lines
    
    def refineAddedAtomPositions(self, forcefield=None):
        if forcefield is None:
            forcefield = app.ForceField('amber14-all.xml', 'tip3p.xml')
        # Conver List missing atoms information to dictionary, for better indexing
        missing_atoms_info_as_dict = defaultdict(list)
        for chain, res_id, res_name, atoms in self.missing_atoms_added:
            missing_atoms_info_as_dict[(chain, res_id, res_name)] += atoms
        system = forcefield.createSystem(self.topology, nonbondedMethod=app.CutoffNonPeriodic, constraints=None, rigidWater=False)
        nonstd_names = [res.name for res, stdname in self.nonstandardResidues]
        for residue in self.topology.residues():
            resdata = (residue.chain.id, int(residue.id), residue.name)
            if resdata in self.missing_residues_added:
                self.log(f'Found fixed residue: {residue}')
                continue
            
            for i, atom in enumerate(residue.atoms()):
                # Always constrained all atoms in modified residue, including hydrogens (because we don't have good force field)
                if residue.name in nonstd_names:
                    system.setParticleMass(atom.index, 0.0)
                    continue
                if (resdata in missing_atoms_info_as_dict) and (atom.name in missing_atoms_info_as_dict[resdata]):
                    self.log(f'Found fixed atom: {atom}')
                    continue
                if atom.element is app.element.hydrogen:
                    continue
                if (self._has_ligand) and (residue.index == self.topology.getNumResidues() - 1) and (i in self.ligand_missing_atoms):
                    continue
                system.setParticleMass(atom.index, 0.0)

        integrator = mm.LangevinIntegrator(300*unit.kelvin, 10/unit.picosecond, 5*unit.femtosecond)
        context = mm.Context(system, integrator)
        context.setPositions(self.positions)
        mm.LocalEnergyMinimizer.minimize(context, tolerance=10)
        self.positions = context.getState(getPositions=True).getPositions()
        return self.positions
    
    def runFixWorkflow(
        self, 
        output_protein: os.PathLike,
        output_ligand: Optional[os.PathLike] = None,
        skip_add_terminal_residues: bool = True, 
        skip_add_terminal_oxygens: bool = False, 
        skip_long_missing_residues: Optional[int] = 10,
        add_hydrogens: bool = True, 
        refine_positions: bool = True, 
        res_num_mapping: Optional[Dict] = None
    ):
        """
        Parameters
        """
        self.findNonstandardResidues()
        self.findMissingResidues(skip_terminals=skip_add_terminal_residues, skip_long_residues=skip_long_missing_residues)
        self.findMissingAtoms(skip_terminals=skip_add_terminal_oxygens)
        self.addMissingAtoms()
        if add_hydrogens or refine_positions:
            self.addMissingHydrogens(7.4)

        top_xmls, ff_xmls = [], []
        dirname = os.path.dirname(output_protein)

        # Don't use self.nonstandardResidues directly, because PDBFixer will add atoms to it
        nonstd_names = [res.name for res, stdname in self.nonstandardResidues]
        for residue in self.topology.residues():
            if residue.name in nonstd_names:
                self.log('Modified residue:', residue)
                self.log('Bonds:', list(residue.bonds()))
                self.log('Atoms:', list(residue.atoms()))
                top_xml, ff_xml = createPatch(residue, dirname)
                top_xmls.append(top_xml)
                ff_xmls.append(ff_xml)
        
        # Re-add ligand topology because PDBFixer will remove all the ligand bonds...
        if self._has_ligand:
            protein = Structure(self.topology, self.positions).select_residues([residue for residue in self.topology.residues()][:-1])
            protein_top = protein.topology
            protein_pos = protein.positions
            modeller = app.Modeller(protein_top, protein_pos)
            modeller.add(self.ligand_top, self.ligand_pos)
            self.topology = modeller.getTopology()
            self.positions = modeller.getPositions()

        if refine_positions:
            for top_xml in top_xmls:
                app.Topology.loadBondDefinitions(top_xml)
            self.topology.createStandardBonds()
            try:
                ff = app.ForceField('amber14-all.xml', 'tip3p.xml', *list(set(ff_xmls)))
                if self._has_ligand:
                    generator = SMIRNOFFTemplateGenerator(molecules=[self.off_mol]).generator
                    ff.registerTemplateGenerator(generator)
                self.refineAddedAtomPositions(ff)
            except ValueError:
                for residue in self.topology.residues():
                    if residue.name == 'PCA':
                        print(list(residue.atoms()), list(residue.bonds()))
                # Some cases OpenFF will failed, use GAFF
                ff = app.ForceField('amber14-all.xml', 'tip3p.xml', *list(set(ff_xmls)))
                if self._has_ligand:
                    generator = GAFFTemplateGenerator(molecules=[self.off_mol], forcefield='gaff-2.11').generator
                    ff.registerTemplateGenerator(generator)
                self.refineAddedAtomPositions(ff)

        
        if self._has_ligand:
            protein = Structure(self.topology, self.positions).select_residues([residue for residue in self.topology.residues()][:-1])
            protein_top = protein.topology
            protein_pos = protein.positions
        else:
            protein_top = self.topology
            protein_pos = self.positions
        
        # Save protein
        seqres = [(seq.chainId, convert_to_seqres(seq.residues, seq.chainId)) for seq in self.sequences]
        seqres.sort(key=lambda x: x[0])
        headers = [x[1] for x in seqres]
        headers += StandardizedPDBFixer.getFixedResidueRemarks(self.missing_residues_skipped, res_num_mapping, use_fixed_remark=False)
        headers += StandardizedPDBFixer.getFixedResidueRemarks(self.missing_residues_added, res_num_mapping)
        headers += StandardizedPDBFixer.getFixedAtomRemarks(self.missing_atoms_added, res_num_mapping)
        headers += StandardizedPDBFixer.getModresRecords(self.mod_res_info, res_num_mapping, self.pdb_id)

        fp = open(output_protein, 'w')
        app.PDBFile.writeHeader(protein_top, fp)
        for line in headers:
            print(line, file=fp)
        # map back residue id and icode
        if res_num_mapping:
            for residue in protein_top.residues():
                res_id = res_num_mapping[residue.chain.id][int(residue.id)]
                if res_id[-1].isalpha():
                    insert_code = res_id[-1]
                    res_id = res_id[:-1]
                else:
                    insert_code = " "
                residue.id = res_id
                residue.insertionCode = insert_code

        app.PDBFile.writeModel(protein_top, protein_pos, keepIds=True, file=fp)
        app.PDBFile.writeFooter(protein_top, file=fp)

        # Save ligand
        if self._has_ligand:
            for i in range(self.rdmol.GetNumAtoms()):
                vec = self.positions[i + protein_top.getNumAtoms()]
                self.rdmol.GetConformer().SetAtomPosition(i, [vec.x * 10, vec.y * 10, vec.z * 10])
            with Chem.SDWriter(output_ligand) as w:
                w.write(self.rdmol)
