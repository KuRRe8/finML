#!/bin/sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
# Number of cores
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:59:00
#PBS -M yang0886@e.ntu.edu.sg
#PBS -m abe
#PBS -N finML_gpu
# Start of commands
cd $PBS_O_WORKDIR
echo thisss is the start of PBS script

module load miniforge3
conda activate xgb
cd ~/proj/finML

python XGBoost.py
python DeepLearning/tabresnet.py
