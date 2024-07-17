cd $PBS_O_WORKDIR
module load apps/Rosetta/2021.38/gnu

folder=designs
nstruct=3

rm tracer*
rm -rf $folder
mkdir $folder

$ROSETTA_BIN/antibody_designer.mpi.linuxgccrelease \
        -l design.list \
        -nstruct $nstruct \
        -out:file:scorefile design.sc \
        -input_ab_scheme "AHO" \
        -disallow_aa CYS \
        -seq_design_cdrs H1 H2 H3 L1 L2 L3 \
        -mc_optimize_dG \
        -mc_total_weight 0 -mc_interface_weight 1 \
        -mintype relax \
        -inner_kt 7.5 -outer_kt 7.5 \
        -top_designs 10 \
        -outer_cycle_rounds 25 \
        -inner_cycle_rounds 1
