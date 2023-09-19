#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J super
#SBATCH -n 2
#SBATCH -p htcgpu8
#SBATCH -q pturagagpu1

#SBATCH -t 00-05:00

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rgoel15@asu.edu

conda env create -n medical_mae -f medical_mae.yml 