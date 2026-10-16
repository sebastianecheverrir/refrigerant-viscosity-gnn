# Viscosity Prediction for Unseen Refrigerants from Molecular Structure Using a Graph Neural Network

This repository is the reproducibility package for **“Viscosity prediction for unseen refrigerants from molecular structure using a graph neural network”** by Sebastián Echeverri Restrepo and Guillermo E. Morales-Espejel, prepared for *Fluid Phase Equilibria*.

It contains the complete notebook workflow, molecular-graph utilities, the processed dataset used in the paper, training histories, and the four trained checkpoints used for the loss-function comparison. A reader can either reproduce the published tables and figures immediately from the archived artifacts or rebuild the dataset and retrain the models.

## Repository layout

```text
.
├── notebooks/
│   ├── 00_build_dataset.ipynb
│   ├── 01_train_model.ipynb
│   ├── 02_evaluate_model.ipynb
│   └── 03_reproduce_paper_results.ipynb
├── src/fluid_processing.py
├── data/
│   ├── fluids_data.pkl
│   └── rdkit_feature_ranking.csv
├── training_runs/
│   └── <criterion_run>/{mape_history.txt,best_checkpoint.pth}
├── results/
├── environment.yml
├── requirements.txt
├── CITATION.cff
└── MODEL_CARD.md
```

## Two reproducibility routes

### Route A — reproduce paper results from archived artifacts

This is the recommended route for exact paper reproduction. The committed pickle and checkpoints preserve the graph features, reference viscosities, model weights, and training histories used in the manuscript.

1. Create the environment.
2. Start Jupyter from the **repository root**.
3. Run `notebooks/02_evaluate_model.ipynb` to verify the primary checkpoint.
4. Run `notebooks/03_reproduce_paper_results.ipynb` to regenerate the paper tables and figures.
5. The final notebook also generates the 10–50 MPa per-refrigerant curves cited in the Supplementary Information.

### Route B — rebuild and retrain from source

1. Run `notebooks/00_build_dataset.ipynb`.
2. Run `notebooks/01_train_model.ipynb` once for each desired loss criterion by changing `criterion_name`.
3. Update the run-folder/checkpoint constants in `notebooks/03_reproduce_paper_results.ipynb` if new run names are used.
4. Run notebooks 02 and 03.

Dataset reconstruction queries NIST WebBook and CoolProp, so exact future coverage can be affected by upstream database or software changes. For this reason, `data/fluids_data.pkl` is the authoritative paper snapshot.

## Installation

Python 3.12 was used for the archived results.

### Conda

```bash
conda env create -f environment.yml
conda activate refrigerant-gnn
python -m ipykernel install --user --name refrigerant-gnn --display-name "Python (refrigerant-gnn)"
jupyter lab
```

### Virtual environment

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Launch Jupyter while the current directory is the repository root; notebook paths are intentionally root-relative.

## Molecular-geometry energy backend

The paper workflow uses GFN2-xTB energy from an xTB-relaxed conformer. Set the xTB executable before rebuilding the molecular graphs:

```text
GRAPH_ENERGY_BACKEND=xtb
XTB_COMMAND=/absolute/path/to/xtb
```

This requires ASE, the `xtb_ase` integration, and a separately installed xTB executable. Because xTB installation is platform-specific, it is not required for Route A. An MMFF94 fallback remains available for development, but it does **not** reproduce the molecular-energy descriptor used in the paper.

## Trained artifacts

The package includes one paper-selected checkpoint and the complete MAPE history for each objective:

| Objective | Run folder | Selected epoch |
|---|---|---:|
| MSE | `mse_20260622_162350` | 1133 |
| MSRE | `msre_20260622_190440` | 134 |
| MAPE | `mape_20260622_200909` | 2543 |
| Max-MSRE | `max_msre_20260623_001251` | 2102 |

The primary paper model is Max-MSRE. See `MODEL_CARD.md` for scope and limitations.

## Expected paper snapshot

- 29 refrigerants and 725 state points
- 24 training refrigerants (600 points)
- 5 held-out validation refrigerants (125 points)
- Temperature range: 283.15–363.15 K
- Pressure range: 10–50 MPa
- Primary Max-MSRE validation MAPE: approximately 5.09%
- Primary Max-MSRE validation MAE: approximately 12.84 μPa·s

Small numerical differences can arise from hardware, dependency, and upstream-data changes. Route A should be used when exact artifact-level reproduction is required.

## Data provenance and network access

Viscosity data are queried from CoolProp and the NIST Chemistry WebBook. Molecular identifiers and formulae are obtained through CoolProp/pyfluids, and molecular descriptors are calculated with RDKit. See `DATA_SOURCES.md` for details and attribution links.

The data-building notebook currently disables TLS certificate verification for NIST requests to support a corporate proxy environment. On networks with a normal certificate chain, change `verify=False` to `verify=True` before running it.

## Reproducibility caveats

- The split is by refrigerant identity, not by individual state point.
- Validation tests molecular transfer inside the sampled thermodynamic domain; it is not unrestricted pressure or temperature extrapolation.
- Pickle files should only be loaded from trusted sources because Python pickle can execute arbitrary code.
- Publication-quality PGF/PDF output in notebook 03 requires a working LaTeX installation with `pdflatex`.

## Citation

Use the metadata in `CITATION.cff`. Add the journal DOI, year, volume, and pages when the final bibliographic record is available.

## License before public release

No software or data license has been selected in this draft package. The copyright owner should choose and approve licenses before publishing the GitHub repository; see `LICENSE.md`.

