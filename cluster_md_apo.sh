cd $PBS_O_WORKDIR
module load apps/gromacs/2022.1/gnu
#for i in {8001..8946}
while read -r i
do
cp inputs/correction.sh trajectories/$i/.
cd trajectories/$i
bash correction.sh
mkdir cluster
cd cluster
echo 18 1 | gmx_mpi cluster -f ../md_fitted.xtc -s ../md.tpr -n ../bb.ndx -method gromos -cutoff 0.25 -dt 100 -clndx -clid -cl -tr -sz -g
csplit -s -z -f cluster_${i}_ -n 1 clusters.pdb /ENDMDL/+1 {*}
for j in cluster_${i}_* ; do mv ${j} ${j}.pdb ; done
cd ../../../
done < cluster_empty.txt
