


def duplicate_missing_info_by_chain_map(missing_residues_added, missing_atoms_added, chain_map):
    from collections import defaultdict
    dup_missing_residues = []
    dup_missing_atoms = []

    chain_targets = defaultdict(list)
    for src_chain, out_chain in chain_map:
        chain_targets[src_chain].append(out_chain)

    for chain, res_id, res_name in missing_residues_added:
        for out_chain in chain_targets.get(chain, []):
            dup_missing_residues.append((out_chain, res_id, res_name))

    for chain, res_id, res_name, atoms in missing_atoms_added:
        for out_chain in chain_targets.get(chain, []):
            dup_missing_atoms.append((out_chain, res_id, res_name, list(atoms)))

    return dup_missing_residues, dup_missing_atoms

def _find_assembly_in_structure(st, assembly_name):
    for assembly in st.assemblies:
        if assembly.name == str(assembly_name):
            return assembly
    return None


def _build_subchain_maps_from_model(model):
    exact_map = {}
    seq_map = {}
    subchain_to_chain = {}

    for chain in model:
        for residue in chain:
            subchain = residue.subchain
            if not subchain:
                subchain = chain.name

            key_exact = (chain.name, residue.seqid.num, residue.seqid.icode, residue.name)
            key_seq = (chain.name, residue.seqid.num, residue.seqid.icode)

            exact_map[key_exact] = subchain
            if key_seq not in seq_map:
                seq_map[key_seq] = subchain
            if subchain not in subchain_to_chain:
                subchain_to_chain[subchain] = chain.name

    return exact_map, seq_map, subchain_to_chain


def extract_assembly_instruction_gemmi(cif_file, assembly_name='1'):
    import gemmi
    import numpy as np

    st = gemmi.read_structure(cif_file)
    assembly = _find_assembly_in_structure(st, assembly_name)
    if assembly is None:
        print(f'Assembly {assembly_name} not found in {cif_file}')
        return None

    _, _, subchain_to_chain = _build_subchain_maps_from_model(st[0])

    instructions = []
    for i_gen, gen in enumerate(assembly.generators):
        source_chains = list(gen.chains)
        source_subchains = list(gen.subchains)

        if len(source_chains) == 0 and len(source_subchains) > 0:
            for subchain in source_subchains:
                chain_name = subchain_to_chain.get(subchain, subchain)
                if chain_name not in source_chains:
                    source_chains.append(chain_name)

        for i_op, oper in enumerate(gen.operators):
            tr = oper.transform
            instructions.append({
                'assembly_name': assembly.name,
                'generator_index': i_gen,
                'operator_index': i_op,
                'operator_name': oper.name,
                'operator_type': oper.type,
                'chains': source_chains,
                'subchains': source_subchains,
                'matrix': np.asarray(tr.mat.tolist(), dtype=float),
                'vector': np.asarray(tr.vec.tolist(), dtype=float),
            })

    return instructions


def _structure_to_numpy_arrays(st):
    import numpy as np

    model = st[0]
    atom_rows = []
    coords = []

    for chain in model:
        for residue in chain:
            subchain = residue.subchain
            if not subchain:
                subchain = chain.name

            for atom in residue:
                atom_rows.append({
                    'chain_id': chain.name,
                    'subchain_id': subchain,
                    'resname': residue.name,
                    'resseq': residue.seqid.num,
                    'icode': residue.seqid.icode,
                    'atom_name': atom.name,
                })
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])

    coords = np.asarray(coords, dtype=float)
    return coords, atom_rows


def read_pdb_coords_gemmi(pdb_file):
    import gemmi

    st = gemmi.read_structure(pdb_file)
    coords, atom_rows = _structure_to_numpy_arrays(st)
    return coords, atom_rows, st


def _copy_subchain_ids_from_original_cif_to_refined_pdb(original_cif_file, refined_pdb_file):
    import gemmi

    original_st = gemmi.read_structure(original_cif_file)
    refined_st = gemmi.read_structure(refined_pdb_file)

    exact_map, seq_map, _ = _build_subchain_maps_from_model(original_st[0])

    for chain in refined_st[0]:
        for residue in chain:
            key_exact = (chain.name, residue.seqid.num, residue.seqid.icode, residue.name)
            key_seq = (chain.name, residue.seqid.num, residue.seqid.icode)

            if key_exact in exact_map:
                residue.subchain = exact_map[key_exact]
            elif key_seq in seq_map:
                residue.subchain = seq_map[key_seq]
            else:
                residue.subchain = chain.name

    return original_st, refined_st


def _prune_assembly_to_present_content(assembly, model):
    import gemmi

    present_subchains = set()
    present_chains = set()

    for chain in model:
        present_chains.add(chain.name)
        for residue in chain:
            subchain = residue.subchain
            if not subchain:
                subchain = chain.name
            present_subchains.add(subchain)

    pruned = gemmi.Assembly(assembly.name)
    pruned.author_determined = assembly.author_determined
    pruned.software_determined = assembly.software_determined
    pruned.oligomeric_details = assembly.oligomeric_details
    pruned.special_kind = assembly.special_kind

    for gen in assembly.generators:
        new_gen = gemmi.Assembly.Gen()

        if len(gen.subchains) > 0:
            kept_subchains = [s for s in gen.subchains if s in present_subchains]
            if len(kept_subchains) == 0:
                continue
            new_gen.subchains = kept_subchains
        else:
            kept_chains = [c for c in gen.chains if c in present_chains]
            if len(kept_chains) == 0:
                continue
            new_gen.chains = kept_chains

        for op in gen.operators:
            new_gen.operators.append(op)

        pruned.generators.append(new_gen)

    return pruned


def apply_assembly_instructions_gemmi(original_cif_file, refined_pdb_file, assembly_name='1'):
    import numpy as np

    instructions = extract_assembly_instruction_gemmi(original_cif_file, assembly_name=assembly_name)
    if instructions is None:
        return None

    _, refined_st = _copy_subchain_ids_from_original_cif_to_refined_pdb(original_cif_file, refined_pdb_file)
    coords, atom_rows = _structure_to_numpy_arrays(refined_st)

    present_subchains = set(row['subchain_id'] for row in atom_rows)
    present_chains = set(row['chain_id'] for row in atom_rows)

    copies = []
    for inst in instructions:
        atom_indices = []

        if len(inst['subchains']) > 0:
            wanted_subchains = [s for s in inst['subchains'] if s in present_subchains]
            for i_atom, row in enumerate(atom_rows):
                if row['subchain_id'] in wanted_subchains:
                    atom_indices.append(i_atom)
        else:
            wanted_chains = [c for c in inst['chains'] if c in present_chains]
            for i_atom, row in enumerate(atom_rows):
                if row['chain_id'] in wanted_chains:
                    atom_indices.append(i_atom)

        atom_indices = np.asarray(atom_indices, dtype=int)
        coords_copy = coords[atom_indices] @ inst['matrix'].T + inst['vector']

        copies.append({
            'assembly_name': inst['assembly_name'],
            'generator_index': inst['generator_index'],
            'operator_index': inst['operator_index'],
            'operator_name': inst['operator_name'],
            'operator_type': inst['operator_type'],
            'chains': list(inst['chains']),
            'subchains': list(inst['subchains']),
            'subchains_present': wanted_subchains if len(inst['subchains']) > 0 else [],
            'atom_indices': atom_indices,
            'coords': coords_copy,
            'matrix': inst['matrix'],
            'vector': inst['vector'],
        })

    return copies


def write_biological_assembly_pdb_gemmi(original_cif_file,
                                        refined_pdb_file,
                                        pdb_out,
                                        assembly_name='1',
                                        chain_naming='Short',
                                        merge_dist=0.0):
    import gemmi

    original_st, refined_st = _copy_subchain_ids_from_original_cif_to_refined_pdb(original_cif_file, refined_pdb_file)

    assembly = _find_assembly_in_structure(original_st, assembly_name)
    if assembly is None:
        print(f'Assembly {assembly_name} not found in {original_cif_file}')
        return None

    pruned_assembly = _prune_assembly_to_present_content(assembly, refined_st[0])
    input_model = refined_st[0]
    input_chain_count = len(input_model)
    how = getattr(gemmi.HowToNameCopiedChain, chain_naming)
    assembled_model = gemmi.make_assembly(pruned_assembly, refined_st[0], how)
    has_new_assembled_chain = len(assembled_model) > input_chain_count
    if merge_dist > 0:
        gemmi.merge_atoms_in_expanded_model(assembled_model, gemmi.UnitCell(), max_dist=merge_dist)

    out_st = refined_st.clone()
    while len(out_st) > 1:
        del out_st[1]
    out_st[0] = assembled_model
    out_st.assign_serial_numbers()
    out_st.write_pdb(pdb_out)

    return out_st, has_new_assembled_chain


def _get_chain_order_from_model(model):
    return [chain.name for chain in model]


def _get_generator_source_chain_order(pruned_assembly, source_model):
    out = []

    for gen in pruned_assembly.generators:
        if len(gen.subchains) > 0:
            source_chains = []
            for chain in source_model:
                chain_subchains = set()
                for residue in chain:
                    subchain = residue.subchain
                    if not subchain:
                        subchain = chain.name
                    chain_subchains.add(subchain)

                if len(chain_subchains.intersection(set(gen.subchains))) > 0:
                    source_chains.append(chain.name)
        else:
            source_chains = list(gen.chains)

        for op in gen.operators:
            for chain_name in source_chains:
                out.append(chain_name)

    return out


def get_assembly_chain_map_gemmi(original_cif_file, source_pdb_file, assembled_pdb_file, assembly_name='1'):
    import gemmi

    original_st, source_st = _copy_subchain_ids_from_original_cif_to_refined_pdb(original_cif_file, source_pdb_file)

    assembly = _find_assembly_in_structure(original_st, assembly_name)
    pruned_assembly = _prune_assembly_to_present_content(assembly, source_st[0])

    source_chain_order = _get_generator_source_chain_order(pruned_assembly, source_st[0])

    assembled_st = gemmi.read_structure(assembled_pdb_file)
    output_chain_order = _get_chain_order_from_model(assembled_st[0])

    return list(zip(source_chain_order, output_chain_order))



def _replace_char_in_line(line, idx, new_char):
    return line[:idx] + new_char + line[idx + 1:]
def ensure_ter_between_chains_in_pdb(pdb_file):
    out_lines = []
    last_chain = None
    last_was_ter = False

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                chain = line[21]
                if (last_chain is not None) and (chain != last_chain) and (not last_was_ter):
                    out_lines.append("TER\n")
                    last_was_ter = True

                out_lines.append(line)
                last_chain = chain
                last_was_ter = False
                continue

            if line.startswith("TER"):
                out_lines.append(line if line.endswith("\n") else line + "\n")
                last_was_ter = True
                continue

            out_lines.append(line)

        if (last_chain is not None) and (not last_was_ter):
            out_lines.append("TER\n")

    with open(pdb_file, "w") as f:
        f.writelines(out_lines)

def rewrite_assembled_pdb_header_with_chain_annotations(source_pdb_file, assembled_pdb_file, chain_map):
    from collections import defaultdict
    from hiqbind.fix_protein import convert_to_seqres
    def _replace_char(line, idx, new_char):
        return line[:idx] + new_char + line[idx + 1:]

    def _is_remark465_data_line(line):
        return (
            line.startswith('REMARK 465')
            and len(line) >= 27
            and line[19].strip() != ''
            and line[21:26].strip().isdigit()
        )

    def _is_remark470_data_line(line):
        return (
            line.startswith('REMARK 470')
            and len(line) >= 27
            and line[19].strip() != ''
            and line[20:26].strip() != ''
        )

    def _make_ter_line(serial, last_coord_line):
        resname = last_coord_line[17:20]
        chain = last_coord_line[21]
        resseq = last_coord_line[22:26]
        icode = last_coord_line[26]
        return f"TER   {serial:>5d}      {resname} {chain}{resseq}{icode}\n"

    with open(source_pdb_file, 'r') as f:
        source_lines = f.readlines()

    source_header = []
    for line in source_lines:
        if line.startswith(('ATOM  ', 'HETATM', 'MODEL ')):
            break
        source_header.append(line.rstrip('\n'))

    with open(assembled_pdb_file, 'r') as f:
        assembled_lines = f.readlines()

    body_start = 0
    for i, line in enumerate(assembled_lines):
        if line.startswith(('ATOM  ', 'HETATM', 'MODEL ')):
            body_start = i
            break

    body_lines = assembled_lines[body_start:]

    other_header = []
    source_seq_by_chain = defaultdict(list)
    modres_by_src = defaultdict(list)
    remark465_prefix = []
    remark465_by_src = defaultdict(list)
    remark470_prefix = []
    remark470_by_src = defaultdict(list)

    for line in source_header:
        if line.startswith('SEQRES'):
            chain = line[11]
            source_seq_by_chain[chain].extend(line[19:].split())

        elif line.startswith('MODRES'):
            modres_by_src[line[16]].append(line)

        elif line.startswith('REMARK 465'):
            if _is_remark465_data_line(line):
                remark465_by_src[line[19]].append(line)
            else:
                remark465_prefix.append(line)

        elif line.startswith('REMARK 470'):
            if _is_remark470_data_line(line):
                remark470_by_src[line[19]].append(line)
            else:
                remark470_prefix.append(line)

        else:
            other_header.append(line)

    grouped_targets = defaultdict(list)
    for src_chain, out_chain in chain_map:
        grouped_targets[src_chain].append(out_chain)

    new_header = []
    new_header.extend(other_header)

    # regenerate SEQRES from full per-chain sequence
    for src_chain, out_chains in grouped_targets.items():
        seq = source_seq_by_chain.get(src_chain, [])
        for out_chain in out_chains:
            if len(seq) > 0:
                new_header.extend(convert_to_seqres(seq, out_chain).split('\n'))

    # duplicate MODRES
    for src_chain, out_chains in grouped_targets.items():
        for out_chain in out_chains:
            for line in modres_by_src.get(src_chain, []):
                new_header.append(_replace_char(line, 16, out_chain))

    # preserve REMARK 465 prefix once, duplicate chain-specific rows
    new_header.extend(remark465_prefix)
    for src_chain, out_chains in grouped_targets.items():
        for out_chain in out_chains:
            for line in remark465_by_src.get(src_chain, []):
                new_header.append(_replace_char(line, 19, out_chain))

    # preserve REMARK 470 prefix once, duplicate chain-specific rows
    new_header.extend(remark470_prefix)
    for src_chain, out_chains in grouped_targets.items():
        for out_chain in out_chains:
            for line in remark470_by_src.get(src_chain, []):
                new_header.append(_replace_char(line, 19, out_chain))

    # rewrite body with fresh serials and proper TER records
    new_body = []
    trailing = []

    serial = 1
    last_coord_line = None
    last_segment_key = None

    for line in body_lines:
        rec = line[:6]

        if rec in ('ATOM  ', 'HETATM'):
            chain = line[21]
            record_type = rec.strip()
            segment_key = (record_type, chain)

            if (last_segment_key is not None) and (segment_key != last_segment_key):
                new_body.append(_make_ter_line(serial, last_coord_line))
                serial += 1

            new_line = line[:6] + f"{serial:>5d}" + line[11:]
            if not new_line.endswith('\n'):
                new_line += '\n'

            new_body.append(new_line)
            last_coord_line = new_line
            last_segment_key = segment_key
            serial += 1
            continue

        if rec == 'TER   ':
            continue

        if line.startswith('END') or line.startswith('CONECT'):
            trailing.append(line if line.endswith('\n') else line + '\n')

    if last_coord_line is not None:
        new_body.append(_make_ter_line(serial, last_coord_line))
        serial += 1

    if len(trailing) == 0:
        trailing = ['END\n']

    with open(assembled_pdb_file, 'w') as f:
        for line in new_header:
            f.write(line + '\n')
        for line in new_body:
            f.write(line)
        for line in trailing:
            f.write(line)

