a=$1
pdb_splitchain docked_cluster_ensemble/${a}.pdb
val=`echo ${a} | cut -d _ -f 1`
pdb=`sed -n "${val}p" list.txt`
x=`pdb_selchain -H uniq_fv_9k_pdbs/${pdb} | pdb_wc -r | head -1 | awk '{print $3}'`
pdb_selres -1:${x} ${a}_A.pdb | pdb_rplchain -A:H > ${a}_H.pdb
pdb_selres -$((x+1)): ${a}_A.pdb | pdb_rplchain -A:L > ${a}_L.pdb
pdb_tidy ${a}_B.pdb | pdb_rplchain -B:A > ${a}_antigen.pdb
pdb_merge ${a}_H.pdb ${a}_L.pdb | pdb_reres -1 | pdb_tidy > ${a}_antibody.pdb
python anarci_.py ${a}_antibody.pdb ${a}_renum.pdb
pdb_merge ${a}_renum.pdb ${a}_antigen.pdb | grep -v END | pdb_reatom -1 > ${a}_chothia.pdb
$ROSETTA_BIN/antibody_numbering_converter.mpi.linuxgccrelease -s ${a}_chothia.pdb -input_ab_scheme AHO -output_ab_scheme AHO
mv ${a}_chothia_0001.pdb ${a}_aho.pdb
mv ${a}_aho.pdb renumbered_chothia-aho/
rm ${a}_*.pdb
