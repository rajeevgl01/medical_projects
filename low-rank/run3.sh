#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J super
#SBATCH -c 48
#SBATCH --mem=500G
#SBATCH -p general
#SBATCH -q public
#SBATCH --gres=gpu:1g.10gb:1

#SBATCH -t 02-00:00

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL         # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=rgoel15@asu.edu   # send-to address

module purge
module load mamba/latest
mamba info --envs
source activate /home/unath/anaconda3/envs/swin
module load cuda-11.7.0-gcc-11.2.0
module load cudnn-8.0.5.39-10.1-gcc-12.1.0

python low_rank.py --rank_factor 0.1 \
--mapping_path /data/yyang409/rgoel15/medical_mae_features/linearMapping_vit_small_mae_imagenet.pt \
--out_dir /data/yyang409/rgoel15/medical_mae_linear_mappings/ \
--model vit_small_mae_imagenet