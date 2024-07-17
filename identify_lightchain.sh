cd $PBS_O_WORKDIR

source ~/apps/scripts/source_conda.sh
conda activate py310

a=1
while read -r l
do
pdb_selchain -L uniq_fv_9k_pdbs/${l} | pdb_tofasta > out.txt
b=`ANARCI -i out.txt | sed -n "6p" | cut -d '|' -f 3`
echo -e "${a} \t ${b}" >> light_chain_type.txt
a=$((a+1))
done < list.txt
