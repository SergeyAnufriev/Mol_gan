#!/bin/bash -l
. $MODULESHOME/init/bash

#$ -S /bin/bash
#$ -l gpu=1
#$ -l h_rt=14:00:0
#$ -l mem=1G
#$ -l tmpfs=15G
#$ -N GPUJob
#$ -wd /home/zcemg08/Scratch/Mol_gan2

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

module unload compilers mpi
module load compilers/gnu/4.9.2
module load python/miniconda3/4.5.11
module load cuda/10.1.243/gnu-4.9.2

PYTHONPATH=~/Mol_gan:${PYTHONPATH}
export PYTHONPATH

source activate env5
wandb login 'type api key'

python /home/zcemg08/Scratch/Mol_gan2/name.py
line=$(head -n 1 /home/zcemg08/Scratch/Mol_gan2/id.txt)
wandb agent zcemg08/gan_molecular2/$line

