#PBS -lwalltime=7:59:00
#PBS -lselect=1:ncpus=32:mem=100000mb

module load anaconda3/personal
source activate py39

cd $HOME/Causal-DAGs/

python3 HPC_test.py

echo "Job done"

