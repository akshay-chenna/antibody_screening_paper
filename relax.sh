cd $PBS_O_WORKDIR
module load apps/Rosetta/2021.38/gnu

find renumbered_chothia-aho -name "*.pdb" > relax.list

mkdir relax_renumbered_chothia-aho
$ROSETTA_BIN/relax.mpi.linuxgccrelease -l relax.list -nstruct 1 \
-relax:dualspace true \
-default_repeats 1
