"""Build molecular-graph inputs for refrigerant-viscosity prediction.

The data-building notebook imports this module to retrieve molecular metadata
from CoolProp, construct PyTorch Geometric graphs from SMILES strings, and
evaluate reference viscosities. Graph construction follows the paper workflow:
explicit hydrogens and a three-dimensional conformer are generated with RDKit,
the geometry is relaxed, and atom-, bond-, and graph-level descriptors are
stored in a :class:`torch_geometric.data.Data` object.

GFN2-xTB is the default energy backend used by the paper. Set ``XTB_COMMAND``
to the xTB executable before rebuilding the dataset. ``GRAPH_ENERGY_BACKEND``
may be set to ``mmff94`` for a portable development fallback, but graphs built
with that fallback do not exactly reproduce the paper's energy descriptor.
"""

from CoolProp.CoolProp import PropsSI, get_fluid_param_string
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from pathlib import Path
try:
    from ase import Atoms
    from ase.optimize import BFGS
    from xtb_ase import XTB, XTBProfile
except ImportError:  # Allow MMFF94-only use when optional xTB packages are absent.
    Atoms = BFGS = XTB = XTBProfile = None




def get_smiles(fluid_enum, skip_fluids):
    """Return a CoolProp SMILES string for an eligible pure fluid.

    Fluids are rejected when their identifier is not a string, they are listed
    in ``skip_fluids``, they appear to be mixtures, CoolProp does not identify
    them as pure, or viscosity/SMILES lookup fails.

    Parameters
    ----------
    fluid_enum : str
        CoolProp-compatible fluid identifier.
    skip_fluids : Collection[str]
        Fluid identifiers that must be excluded explicitly.

    Returns
    -------
    tuple[str, str] or None
        ``(fluid_name, smiles)`` for a usable fluid; otherwise ``None``.
    """
    fluid_name = fluid_enum

    # Some enumeration entries represent mixtures as non-string objects.
    if not isinstance(fluid_name, str):
        return None

    # CoolProp mixture identifiers conventionally include a .mix suffix.
    if '.mix' in fluid_name.lower():
        return None

    if fluid_name in skip_fluids:
        return None

    # Confirm purity through CoolProp rather than relying only on the name.
    if not get_fluid_param_string(fluid_name, "pure"):
        return None
    
    # Probe transport-property support at a benign state point.
    try:
        PropsSI('V', 'T', 298.15, 'P', 100e3, fluid_name)
    except Exception:
        return None

    try:
        smiles = get_fluid_param_string(fluid_name, "SMILES")
        return (fluid_name, smiles)
    except Exception:
        return None

def get_mmff94_energy(mol, verbose=False):
    """Evaluate the RDKit MMFF94 force-field energy of a conformer.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    verbose : bool, optional
        If True, print energy (default: False)

    Returns
    -------
    tuple[float, rdkit.Chem.Mol]
        Energy in kcal/mol and the unchanged input molecule.
    """
    props = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    energy = ff.CalcEnergy()
    if verbose:
        print(f"MMFF94 energy: {energy:.6f} kcal/mol")
    return energy, mol


def get_xtb_ase_energy(mol, method='GFN2-xTB', verbose=False, xtb_command=None):
    """Relax an RDKit conformer with xTB through ASE and return its energy.

    Atomic symbols and coordinates are copied into an ASE ``Atoms`` object.
    BFGS optimization is run until the maximum force is below 0.01 eV/Å, and
    the optimized coordinates are copied back to the RDKit conformer.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object with 3D coordinates
    method : str, optional
        One of ``GFN2-xTB``, ``GFN1-xTB``, ``GFN0-xTB``, or ``GFN-FF``.
    verbose : bool, optional
        If True, print energy and details (default: False)
    xtb_command : str, optional
        Path to the xTB executable. When omitted, ``XTB_COMMAND`` is used.

    Returns
    -------
    tuple[float, rdkit.Chem.Mol]
        xTB energy in eV and the molecule with its optimized conformer.

    Notes
    -----
    The input molecule must already contain a three-dimensional conformer.
    ASE reports energy in eV; multiplying by 23.0609 converts it to kcal/mol.
    """
    if Atoms is None or XTB is None:
        raise ImportError(
            "The optional ASE/xTB integration is not installed. Install 'ase' and "
            "the project-compatible 'xtb_ase' package, or use "
            "GRAPH_ENERGY_BACKEND=mmff94."
        )

    # Resolve the xTB executable from the argument or environment.
    if xtb_command is None:
        xtb_command = os.environ.get("XTB_COMMAND")
    if not xtb_command:
        raise RuntimeError(
            "Set XTB_COMMAND to the xTB executable when using "
            "GRAPH_ENERGY_BACKEND=xtb."
        )
    
    # Transfer the embedded RDKit conformer into ASE.
    conformer = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = conformer.GetPositions()
    
    atoms = Atoms(symbols=symbols, positions=positions)
    
    # XTBProfile receives the executable as the first command-line argument.
    profile = XTBProfile(argv=[xtb_command])

    # Translate public method names to the command-line values used by xtb_ase.
    method_map = {
        'GFN2-xTB': 'gfn2-xtb',
        'GFN1-xTB': 'gfn1-xtb',
        'GFN0-xTB': 'gfn0-xtb',
        'GFN-FF': 'gfn-ff'
    }
    method_arg = method_map.get(method, 'gfn2-xtb')

    atoms.calc = XTB(method=method_arg, profile=profile)

    dyn = BFGS(atoms)
    dyn.run(fmax=0.01)

    energy = atoms.get_potential_energy()  # ASE energy unit: eV.

    # Write optimized positions back to the RDKit conformer
    optimized_positions = atoms.get_positions()
    conformer = mol.GetConformer()
    for i, pos in enumerate(optimized_positions):
        conformer.SetAtomPosition(i, pos.tolist())
    
    if verbose:
        print(f"xTB energy ({method}): {energy:.6f} eV ({energy * 23.0609:.6f} kcal/mol)")
    
    return energy, mol




def smiles_to_graph(smiles, verbose=False):
    """Convert a SMILES representation into the graph used by the GNN.
    
    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule
    verbose : bool, optional
        If True, print detailed information during processing (default: False)
        
    Returns
    -------
    torch_geometric.data.Data
        Graph with atom features in ``x``, bidirectional connectivity in
        ``edge_index``, bond features in ``edge_attr``, molecular descriptors
        in ``u``, and matching labels in ``global_feature_names``.
    """
    # Parse, sanitize, and make hydrogens explicit so atom counts and geometry
    # match the representation used to build the paper dataset.
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    mol = Chem.AddHs(mol, addCoords=True)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)

    # Gasteiger charge is one of the atom-level model inputs.
    AllChem.ComputeGasteigerCharges(mol)

    # ETKDG provides the initial geometry; MMFF removes severe local strain
    # before the selected energy backend performs its final evaluation.
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    
    energy_backend = os.environ.get("GRAPH_ENERGY_BACKEND", "xtb").lower()
    if energy_backend == "xtb":
        energy, mol = get_xtb_ase_energy(
            mol,
            method="GFN2-xTB",
            verbose=verbose,
            xtb_command=os.environ.get("XTB_COMMAND"),
        )
    elif energy_backend == "mmff94":
        energy, mol = get_mmff94_energy(mol, verbose=verbose)
    else:
        raise ValueError(
            "GRAPH_ENERGY_BACKEND must be either 'mmff94' or 'xtb'."
        )

    # Use a fixed 118-element vocabulary so every graph has the same width.
    atomic_numbers = np.array([[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
    one_hot_encoder = OneHotEncoder(categories=[list(range(1, 119))], sparse_output=False)
    one_hot_atomic_numbers = one_hot_encoder.fit_transform(atomic_numbers)
    
    # Atom features combine elemental identity and local physicochemical data.
    node_features = []
    for idx, atom in enumerate(mol.GetAtoms()):
        if verbose:
            print(idx, atom.GetSymbol())
        features = [
            *one_hot_atomic_numbers[idx],  # one-hot encoded atomic number
            atom.GetDegree(),             # degree
            atom.GetFormalCharge(),       # formal charge
            atom.GetIsAromatic(),         # aromaticity
            float(atom.GetProp("_GasteigerCharge")),  # Gasteiger charge
            atom.GetMass(),               # atomic mass
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Encode each covalent bond in both directions for message passing.
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        bond_features = [
            bond.GetBondType(),   # bond type
            bond.GetIsConjugated(),  # conjugation
            bond.IsInRing(),         # in ring
            float(mol.GetConformer().GetAtomPosition(i).Distance(mol.GetConformer().GetAtomPosition(j)))  # bond length (Å)
        ]
        edge_features.append(bond_features)
        edge_features.append(bond_features)
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Prepend three custom descriptors to RDKit's complete descriptor vector.
    base_global = [
        mol.GetNumAtoms(),    # number of atoms
        mol.GetNumBonds(),    # number of bonds
        energy,               # GFN2-xTB energy by default; MMFF94 in fallback mode.
    ]
    
    # Replace undefined descriptor values with zero to keep tensors finite.
    desc_dict = Descriptors.CalcMolDescriptors(mol)
    desc_names = list(desc_dict.keys())
    desc_values = [
        float(v) if v is not None and not (isinstance(v, float) and (v != v)) else 0.0
        for v in desc_dict.values()
    ]
    
    global_features = torch.tensor([base_global + desc_values], dtype=torch.float)
    
    # Keep descriptor names alongside the tensor for interpretation and
    # permutation-importance reporting.
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=global_features)
    energy_feature_name = "xtb_energy" if energy_backend == "xtb" else "mmff94_energy"
    data.global_feature_names = ["num_atoms", "num_bonds", energy_feature_name] + desc_names
    
    if verbose:
        print(data)
        print(f"Global feature length: {data.u.shape[1]}")
    
    return data


def calculate_viscosities(fluid_name, temperatures=None, pressures=None, verbose=True):
    """Evaluate CoolProp viscosity over a temperature-pressure grid.
    
    Parameters
    ----------
    fluid_name : str
        Name of the fluid
    temperatures : list, optional
        List of temperatures in Kelvin (default: [298.15, 313.15, 343.15, 373.15])
    pressures : list, optional
        List of pressures in Pascals (default: [10e6, 15e6,  20e6, 25e6,  30e6, 35e6, 40e6, 45e6, 50e6])
    verbose : bool, optional
        If True, print detailed information (default: True)
        
    Returns
    -------
    tuple[numpy.ndarray, int] or None
        Flattened viscosities in µPa·s and the number of failed state points.
        Returns ``None`` when every CoolProp call fails.
    """
    if temperatures is None:
        temperatures = [25 + 273.15, 40 + 273.15, 70 + 273.15, 100 + 273.15]
    
    if pressures is None:
        # Default diagnostic grid: 10–50 MPa in 5 MPa increments.
        pressures = [10e6, 15e6, 20e6, 25e6, 30e6, 35e6, 40e6, 45e6, 50e6]
    
    viscosities = []
    failed_count = 0
    
    for T in temperatures:
        for P in pressures:
            try:
                visc = PropsSI('V', 'T', T, 'P', P, fluid_name)
                viscosities.append(visc)
            except Exception as e:
                if verbose:
                    print(f"❌ Failed at T={T:.2f}K, P={P/1e6:.1f}MPa: {str(e)[:80]}")
                viscosities.append(np.nan)
                failed_count += 1
    
    # CoolProp returns Pa·s; the notebooks and paper report µPa·s.
    viscosities = np.array(viscosities) * 1e6
    
    # A partially populated grid remains useful to callers; a fully failed grid
    # is represented by None so the fluid can be skipped explicitly.
    total_calculations = len(temperatures) * len(pressures)
    if failed_count == total_calculations:
        if verbose:
            print(f"⚠️  WARNING: All viscosity calculations failed for {fluid_name}!")
            print(f"   Skipping this fluid...")
        return None
    elif failed_count > 0:
        if verbose:
            print(f"⚠️  {failed_count} out of {total_calculations} calculations failed")
    else:
        if verbose:
            print(f"✓ All {total_calculations} viscosity calculations successful")
    
    if verbose:
        print(f"Viscosities: {viscosities}")
    
    return viscosities, failed_count


def count_non_h_atoms(smiles):
    """Return a lightweight count of uppercase element tokens in a SMILES.

    This helper is retained for compatibility with earlier exploratory code.
    RDKit atom counts should be preferred for rigorous molecular processing.
    """
    count = 0
    for char in smiles:
        # Unbracketed organic elements begin with an uppercase character.
        if char.isupper():
            count += 1
    return count

