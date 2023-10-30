
#PBS -N GPUDeepAnt
#PBS -l select=1:ngpus=1

#PBS -l walltime=1:00:00 
#PBS -oe
#PBS -m abe
#PBS -M emerson.okano@unifesp.br

  
#PBS -V
python /lustre/eyokano/DeepAntTF/run_DA.py --path /lustre/eyokano/datasets/UCR_Anomaly_FullData/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt --WL 100 --n 2 --i 1 --seed 1 
