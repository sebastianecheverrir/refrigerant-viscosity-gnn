# Data sources and provenance

## Thermophysical properties

- **NIST Chemistry WebBook:** <https://webbook.nist.gov/chemistry/fluid/>
- **CoolProp:** <https://coolprop.org/>
- **pyfluids:** <https://github.com/portyanikhin/pyfluids>

The dataset-building notebook identifies pure refrigerants, samples dynamic viscosity on a regular temperature–pressure grid, and selects a complete source per refrigerant. CoolProp is preferred when it covers the complete grid; NIST is used as a whole-fluid fallback.

## Molecular representations and descriptors

- SMILES and molecular formula strings are retrieved through CoolProp.
- Molecular graphs and descriptors are generated with RDKit.
- Graph tensors are represented with PyTorch Geometric.
- The paper workflow uses GFN2-xTB energy from an xTB-relaxed conformer. The exact graph tensors used in the paper are preserved in `data/fluids_data.pkl`.

## Archived dataset

`data/fluids_data.pkl` is the authoritative snapshot used for the paper and trained checkpoints. It contains Python/PyTorch Geometric objects and must only be loaded from a trusted copy.

Expected schema:

| Column | Meaning |
|---|---|
| `Name` | Refrigerant identifier |
| `smiles` | Molecular SMILES representation |
| `pressure` | Pressure in Pa |
| `temperature` | Temperature in K |
| `viscosity` | Dynamic viscosity in μPa·s |
| `class` | Refrigerant chemical class |
| `graph` | PyTorch Geometric molecular graph |

Upstream databases and property libraries can change. Rebuilding the dataset at a later date may therefore produce different coverage or values; use the archived snapshot for exact paper reproduction.

