cd $PBS_O_WORKDIR
source ~/apps/scripts/source_conda.sh
conda activate py310

mkdir outputs_token
#esm-extract esm2_t48_15B_UR50D esm1.fasta outputs_mean --include mean 
esm-extract esm2_t30_150M_UR50D sequences.fasta outputs_token --include per_tok
