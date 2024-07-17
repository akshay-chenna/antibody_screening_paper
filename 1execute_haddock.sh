
cd /home/chemical/phd/chz198152/scratch/antibody/dock
mkdir $i

cp active-passive_ab/active-passive_${i}.txt ${i}/.
cp synu_pdbs/synu_active.txt ${i}/.

echo "AMBIG_TBL=./ambig.tbl" >> ${i}/run.param 
echo "HADDOCK_DIR=/home/chemical/phd/chz198152/apps/haddock2.4-2021-05" >> ${i}/run.param
echo "N_COMP=2" >> ${i}/run.param
echo "PDB_FILE1=./cluster_${i}_0.pdb" >> ${i}/run.param
echo "PDB_FILE2=./C1.pdb" >> ${i}/run.param
echo "PDB_LIST1=./ab_list.list" >> ${i}/run.param
echo "PROJECT_DIR=./" >> ${i}/run.param
echo "RUN_NUMBER=1" >> ${i}/run.param

cd ${i}

/home/apps/anaconda3/5.2.0/gnu/bin/python ~/apps/haddock-tools/active-passive-to-ambig.py active-passive_${i}.txt synu_active.txt | sed s/segid/name\ CA\ and\ segid/g | sed s/2.0/3.0/g > ambig.tbl

cp ../inputs/generate_run.sh .
cp ../cluster_centers/cluster_${i}_*.pdb .
cp ../synu_pdbs/C1.pdb .
sed -i 's/ENDMDL/END/g' *.pdb
ls cluster_*.pdb >> ab_list.list
sed -i 's/^\(.\)\(.*\)\(.\)$/"\1\2\3"/' ab_list.list
./generate_run.sh	

n=`ls cluster_${i}_*.pdb | wc -l`

cd run1
sed -i 's/noecv=true/noecv=false/g' run.cns
sed -i "s/structures_0=1000/structures_0=$((n*100))/g" run.cns
sed -i 's/structures_1=200/structures_1=0/g' run.cns
sed -i 's/anastruc_1=200/anastruc_1=0/g' run.cns
sed -i 's/firstwater="yes"/firstwater="no"/g' run.cns
sed -i 's/fcc_ignc=false/fcc_ignc=true/g' run.cns
sed -i 's/cpunumber_1=1/cpunumber_1=2/g' run.cns
random=`shuf -i 100-999 -n1`
seed=`grep iniseed run.cns | cut -d = -f 5 | cut -d ';' -f 1 | head -1`
sed -i "s/iniseed=${seed}/iniseed=${random}/g" run.cns

cp ../generate_run.sh .
./generate_run.sh >> run.out	

cp ../../inputs/run_analysis.sh ./structures/it0/.
cd structures/it0/
./run_analysis.sh

awk '{ print $1 "\t" $2 "\t" 1.9*$8 -0.8*$NF -0.02*$((NF-2))}' structures_ene-sorted.stat | column -t | tail -n +2 | sort -nk 3 > ene-reweighted.txt
cd ../../../
awk '{print $1}' run1/structures/it0/ene-reweighted.txt | head -200 | while read -r l ; do echo "run1/structures/it0/$l" >> run1_it0.list ; done
ln -s ../tools . 
python2.7 tools/make_contacts.py -f run1_it0.list
sed -e 's/pdb/contacts/' run1_it0.list | sed -e '/^$/d' > run1_it0.contacts
python2.7 tools/calc_fcc_matrix.py -f run1_it0.contacts -o run1_it0_fcc_matrix.out
python2.7 tools/cluster_fcc.py run1_it0_fcc_matrix.out 0.6 -c 4 >> run1_cluster_06fcc.txt

cp ../inputs/cluster_analysis.csh run1/structures/it0/
cp ../inputs/cluster_analysis.sh run1/structures/it0/
cp run1_cluster_06fcc.txt run1/structures/it0/
cd run1/structures/it0/
./cluster_analysis.csh
bash cluster_analysis.sh
cd -
cp -r cluster_minimas/*.pdb ../docked_cluster_ensemble/. 
cd ..

tar cvf ${i}.tar.xz --use-compress-program='xz' ${i}
mv ${i}.tar.xz docked_ensemble/.
rm -rf ${i}
rm master/execute_haddock-1_${i}.sh
