#!/bin/sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
# Number of cores
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:59:00
#PBS -M yang0886@e.ntu.edu.sg
#PBS -m abe
#PBS -N finML_job_2hrs
# Start of commands
cd $PBS_O_WORKDIR
echo thisss is the start of PBS script

module load miniforge3
conda activate deep
cd ~/proj/finML

python DeepLearning/tabresnet.py
