# GraphVAE for EGFR Molecule Generation

This project implements a Graph Variational Autoencoder (GraphVAE) for
molecule generation trained on the ChEMBL EGFR IC50 dataset.

## Dataset
The EGFR IC50 target dataset was obtained from ChEMBL.
The epidermal growth factor receptor is a transmembrane protein that is a receptor for members of the epidermal growth factor family (EGF family) of extracellular protein ligands.
- For more info:
https://en.wikipedia.org/wiki/Epidermal_growth_factor_receptor
- Dataset can be downloaded from:
https://www.ebi.ac.uk/chembl/explore/activities/STATE_ID:RpXoAD36T_GefL3XVyrKqA%3D%3D
### Representation:
- SMILES strings converted to molecular graphs using RDKit
- Node features: One-hot encoding of atom types
- Edge features: Bond types (single, double, triple, aromatic)

## Model
The model follows the Variational Autoencoder (VAE) framework adapted to graphs.
### Encoder:
- Graph Convolutional Networks (GCN)
- Global mean pooling
- Outputs latent mean (μ) and log-variance (logσ²)
### Latent Space:
- Gaussian prior
- Reparameterization used for backpropagation
### Decoder:
- Node decoder: predicts atom types
- Edge decoder: predicts bond types between node pairs
- Fixed maximum number of nodes per molecule

## Training
Training is performed in two stages - node decoder training, edge decoder training.  
Loss consists of:
- Node Reconstruction Loss
- Edge Reconstruction Loss
- KL Divergence to regularize the latent space  



## Molecule Generation
- Samples latent vectors.
- Decodes node and edge probabilities.
- Applies thresholding and symmetry constraints
- Converts decoded graphs to RDKit molecules
- Validates and computes molecular properties (MW, logP, QED)

## Future Scope
- Conditional generation based on IC50 or bioactivity.
- Extend to multi-target drug discovery task using other datasets like QM9, ZINC250K etc.
- Use autoregressive or flow based decoders.
- Fine tune generation using reinforcement learning.

## Requirements
- PyTorch
- PyTorch Geometric
- RDKit
- Pandas
- NumPy

## Files
- `graphvae_egfr.ipynb` – Full Colab notebook 
- `graph_vae_egfr.pt` – Trained model weights
- `config.pkl` – Model configuration
- `vocab.pkl` – Atom and Bond mappings

## Notes
This notebook was developed and trained using Google Colab.
