export OMP_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=$PWD
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
nvidia-cuda-mps-control -d

gmx_mpi grompp -f em1.mdp -c conf_i.gro -r conf_i.gro -p topol.top -o em1.tpr -po em1out.mdp -maxwarn 2
mpirun -np 1 gmx_mpi mdrun -v -s em1.tpr -o em1.trr -g em1.log -c em1.gro -ntomp 1

gmx_mpi grompp -f em2.mdp -c em1.gro -r em1.gro -p topol.top -o em2.tpr -po em2out.mdp -maxwarn 2
mpirun -np 1 gmx_mpi mdrun -v -s em2.tpr -o em2.trr -g em2.log -c em2.gro -ntomp 1

gmx_mpi grompp -f sa.mdp -c em2.gro -r em2.gro -p topol.top -o sa.tpr -po saout.mdp -maxwarn 2
mpirun -np 1 gmx_mpi mdrun -v -s sa.tpr -o sa.trr -x sa.xtc -cpo sa.cpt -e sa.edr -g sa.log -c sa.gro -ntomp 1 -nstlist 150 -nb gpu -bonded gpu -pme gpu -update gpu

gmx_mpi grompp -f nvt.mdp -c sa.gro -r sa.gro -p topol.top -o nvt.tpr -po nvtout.mdp -maxwarn 2
mpirun -np 1 gmx_mpi mdrun -v -s nvt.tpr -o nvt.trr -x nvt.xtc -cpo nvt.cpt -e nvt.edr -g nvt.log -c nvt.gro -ntomp 1 -nstlist 150 -nb gpu -bonded gpu -pme gpu -update gpu

gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr -po nptout.mdp -maxwarn 2
mpirun -np 1 gmx_mpi mdrun -v -s npt.tpr -o npt.trr -x npt.xtc -cpo npt.cpt -e npt.edr -g npt.log -c npt.gro -ntomp 1 -nstlist 150 -nb gpu -bonded gpu -pme gpu -update gpu

gmx_mpi grompp -f md.mdp -c npt.gro -r npt.gro -p topol.top -o md.tpr -po mdout.mdp -maxwarn 2
mpirun -np 1 gmx_mpi mdrun -v -s md.tpr -o md.trr -x md.xtc -cpo md.cpt -e md.edr -g md.log -c md.gro -ntomp 1 -nstlist 150 -nb gpu -bonded gpu -pme gpu -update gpu
