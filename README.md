# Installation and relevant scripts

Installation of python packages: 
`conda env create -f install.yml`

Other applications:
HADDOCK
GROMACS

## Identification of distinct antibodies
run_cluster_identify.sh
cdr_analysis.ipynb

## MD of apo
md_production.sh: High temperature MD of apo antibodies starting from igfold.
md.mdp: MD parameters
correction_md.sh: Correct the generated trajectories.
cluster_md_apo.sh: Clustering apo ensemble from MD.

## Docking
find_cdrs.py: Identify cdrs.
find_cdrs.sh: Run to identify cdrs.
1execute_haddock.sh: Docking run and parameters.
relax.sh: Relax docked pose.

## RAbD design
identify_lightchain.sh: To identify the light chain of the antibody
anarci_aho.py: Use ANARCI to renumber as per AHO (modified from https://github.com/Graylab/IgFold/tree/main/igfold).
renumber_chothia-aho.sh: Convert numbering scheme from Chothia to AHO.
rabd_design.sh: Rosetta antibody design from docked pose

## Binding energy
run_ddg.sh: Compute binding energies.
ddg.xml: param file for computing binding energies.

## ESM embedding generation and training NN.
run_esm.sh: Generate embeddings from ESM2 model.
esm2.py: Learning ΔG from esm2 embedding.
esm2_transformer_lightning.py: lightning version of transformer code.

## Others
diffab_run.sh: Design using diffab.
