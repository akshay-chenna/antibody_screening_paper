#Identification of distinct antibodies
run_cluster_identify.sh
cdr_analysis.ipynb

# MD of apo
md_production.sh: High temperature MD of apo antibodies starting from igfold.
md.mdp: MD parameters
correction_md.sh: Correct the generated trajectories.
cluster_md_apo.sh: Clustering apo ensemble from MD.

# Docking
find_cdrs.py: Identify cdrs.
find_cdrs.sh: Run to identify cdrs.
1execute_haddock.sh: Docking run and parameters.
relax.sh: Relax docked pose.

# RAbD design
rabd_design.sh: Rosetta antibody design from docked pose
# DiffAb design
diffab_run.sh: Design using diffab.

# Binding energy
run_ddg.sh: Compute binding energies.
ddg.xml: param file for computing binding energies.

# ESM embedding generation and training NN.
run_esm.sh: Generate embeddings from ESM2 model.
esm2.py: Learning ΔG from esm2 embedding.
esm2_transformer_lightning.py: lightning version of transformer code.
