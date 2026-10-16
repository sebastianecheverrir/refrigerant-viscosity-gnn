# Model card: refrigerant-viscosity GAT

## Intended use

The model predicts dynamic viscosity for pure refrigerants from a molecular graph, molecular descriptors, pressure, and temperature. It is intended for research reproduction and method comparison within the domain represented by the archived dataset.

## Architecture

- Three graph-attention (`GATConv`) layers
- Four attention heads per layer
- Hidden dimension: 128
- Global mean pooling for the molecular embedding
- Separate fully connected embedding for graph-level descriptors and state variables
- Softplus scalar output to enforce non-negative viscosity

## Training and evaluation

- 29 refrigerants, 725 total state points
- 24 refrigerants used for training
- 5 refrigerants held out entirely for validation: R12, R124, R125, R1234zeE, and R218
- Pressure range: 10–50 MPa
- Temperature range: 283.15–363.15 K
- Primary objective: maximum mean squared relative error (Max-MSRE)

The split prevents the same refrigerant graph from appearing in training and validation. Normalization parameters are fitted on training data only.

## Primary archived model

`training_runs/max_msre_20260623_001251/best_avgmape_iter_01_epoch_2102_avgmape_6p0950_valmape_5p0922_trainmape_7p0977.pth`

Approximate validation metrics reported by the paper workflow:

- MAPE: 5.09%
- MAE: 12.84 μPa·s
- RMSE: 18.83 μPa·s
- Maximum pointwise APE: 22.11%

## Limitations

- The model is not validated for mixtures.
- Validation covers unseen molecules but remains inside the sampled pressure–temperature domain.
- Predictions outside 10–50 MPa or 283.15–363.15 K are extrapolations.
- Accuracy is not established for refrigerant classes or molecular structures absent from the dataset.
- The model should not replace reference equations of state or experimental validation in safety-critical engineering.
- PyTorch state dictionaries and pickle files should only be loaded from trusted sources.

