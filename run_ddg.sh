cd $PBS_O_WORKDIR
module load apps/Rosetta/2021.38/gnu

folder=ddg
nstruct=1

rm tracer*
rm -rf $folder
mkdir $folder

$PBS_NTASKS $ROSETTA_BIN/rosetta_scripts.mpi.linuxgccrelease -parser:protocol ddg.xml \
-l ddg1.list \
-nstruct $nstruct \
-out:file:scorefile ddg.sc
