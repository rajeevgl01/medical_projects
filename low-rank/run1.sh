#!/bin/bash

#SBATCH --get-user-env
#SBATCH -J super
#SBATCH -c 32
#SBATCH --mem=500G
#SBATCH -p general
#SBATCH -q public

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
--mapping_path /data/yyang409/rgoel15/medical_mae_features/linearMapping_vit_small_mae.pt \
--out_dir /data/yyang409/rgoel15/medical_mae_linear_mappings/ \
--model vit_small_mae


python low_rank_fast.py --rank_factor 0.01 \
--mapping_path /data/yyang409/rgoel15/medical_mae_features/linearMapping_vit_base.pt \
--out_dir /data/yyang409/rgoel15/medical_mae_linear_mappings/ \
--model vit_base


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--master_port 12345 --nproc_per_node=1 \
--use_env main_finetune_chestxray_soft_lowrank.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--get_features --soft_low_rank --softRank 76 --train_all \
--tn_loss_factor 0.00125

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--master_port 12346 --nproc_per_node=1 \
--use_env main_finetune_chestxray_soft_lowrank.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--get_features --soft_low_rank --softRank 76 --train_all \
--tn_loss_factor 0.0025

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
--master_port 12347 --nproc_per_node=1 \
--use_env main_finetune_chestxray_soft_lowrank.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--get_features --soft_low_rank --softRank 76 --train_all \
--tn_loss_factor 0.005

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
--master_port 12348 --nproc_per_node=1 \
--use_env main_finetune_chestxray_soft_lowrank.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--get_features --soft_low_rank --softRank 76 --train_all \
--tn_loss_factor 0.01

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--master_port 12345 --nproc_per_node=1 \
--use_env main_finetune_chestxray_soft_lowrank.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--get_features --soft_low_rank --softRank 76 --train_all \
--tn_loss_factor 0.02

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--master_port 12346 --nproc_per_node=1 \
--use_env main_finetune_chestxray_soft_lowrank.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--get_features --soft_low_rank --softRank 76 --train_all \
--tn_loss_factor 0.04

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--master_port 12345 --nproc_per_node=1 \
--use_env main_finetune_chestxray_soft_lowrank.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--get_features --soft_low_rank --softRank 76 --train_all \
--tn_loss_factor 0.08

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--master_port 12346 --nproc_per_node=1 \
--use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 75 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 76 --train_all --tn_loss_factor 0.00125

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--master_port 12345 --nproc_per_node=1 \
--use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 100 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 76 --train_all --tn_loss_factor 0.00125

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--master_port 12346 --nproc_per_node=1 \
--use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_cross \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 100 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 76 --train_all --tn_loss_factor 0.01

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
--master_port 12348 --nproc_per_node=1 \
--use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/unath/medical_mae_original_models/ablation_models \
--log_dir /data/yyang409/unath/medical_mae_original_models/ablation_models \
--batch_size 512 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-b_CXR_0.5M_mae.pth \
--epochs 75 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_base_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official_50.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1'

python -m torch.distributed.launch \
--master_port 12346 --nproc_per_node=4 \
--use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/rgoel15/medical_mae_temp \
--log_dir /data/yyang409/rgoel15/medical_mae_temp \
--batch_size 256 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-s_CXR_0.3M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_small_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--eval --eval_path /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-s_CXR_0.3M_mae.pth


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --master_port 12347 \
--use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/rgoel15/medical_mae_low_rank/resnet_imagenet \
--log_dir /data/yyang409/rgoel15/medical_mae_soft_low_rank/resnet_imagenet \
--batch_size 512 \
--retrain /data/yyang409/rgoel15/medical_mae_original_models/finetuned_models/resnet50_imagenet_swav.pth \
--checkpoint_type "smp_encoder" \
--epochs 40 \
--blr 2.5e-5 --weight_decay 0 \
--model 'resnet50' \
--warmup_epochs 5 \
--drop_path 0 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt \
--nb_classes 14 \
--eval_interval 1 \
--min_lr 1e-6 \
--build_timm_transform \
--aa 'rand-m6-mstd0.5-inc1' --softRank 204 --train_all --tn_loss_factor 0.00125

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--master_port 12345 --nproc_per_node=1 --use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b --log_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b \
--batch_size 512 --retrain /data/yyang409/unath/medical_mae_original_models/ablation_models/checkpoint-best_auc_vit_base_patch16_1_%.pth  --epochs 40 \
--blr 2.5e-5 --layer_decay 0.55 --weight_decay 0 --model vit_base_patch16 --warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 \
--reprob 0 --vit_dropout_rate 0 --data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 --train_list ./data_splits/chestxray/train_official_1.txt \
--val_list ./data_splits/chestxray/val_official.txt --test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-6 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 76 --train_all --tn_loss_factor 0.00125

CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
--master_port 12349 --nproc_per_node=1 --use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b --log_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b \
--batch_size 512 --retrain /data/yyang409/unath/medical_mae_original_models/ablation_models/checkpoint-best_auc_vit_base_patch16_5_%.pth  --epochs 40 \
--blr 2.5e-5 --layer_decay 0.55 --weight_decay 0 --model vit_base_patch16 --warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 \
--reprob 0 --vit_dropout_rate 0 --data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 --train_list ./data_splits/chestxray/train_official_5.txt \
--val_list ./data_splits/chestxray/val_official.txt --test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-6 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 38 --train_all --tn_loss_factor 0.00125

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--master_port 12347 --nproc_per_node=1 --use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b --log_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b \
--batch_size 512 --retrain /data/yyang409/unath/medical_mae_original_models/ablation_models/checkpoint-best_auc_vit_base_patch16_25_%.pth  --epochs 40 \
--blr 2.5e-5 --layer_decay 0.55 --weight_decay 0 --model vit_base_patch16 --warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 \
--reprob 0 --vit_dropout_rate 0 --data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 --train_list ./data_splits/chestxray/train_official_25.txt \
--val_list ./data_splits/chestxray/val_official.txt --test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-6 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 38 --train_all --tn_loss_factor 0.00125

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch \
--master_port 12348 --nproc_per_node=1 --use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b --log_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_b \
--batch_size 512 --retrain /data/yyang409/unath/medical_mae_original_models/ablation_models/checkpoint-best_auc_vit_base_patch16_50_%.pth  --epochs 40 \
--blr 2.5e-5 --layer_decay 0.55 --weight_decay 0 --model vit_base_patch16 --warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 \
--reprob 0 --vit_dropout_rate 0 --data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 --train_list ./data_splits/chestxray/train_official_50.txt \
--val_list ./data_splits/chestxray/val_official.txt --test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-6 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--softRank 76 --train_all --tn_loss_factor 0.00125

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 12349 \
--nproc_per_node=1 --use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_s \
--log_dir /data/yyang409/unath/medical_mae_soft_low_rank/ablation/vit_s \
--batch_size 512 --retrain /data/yyang409/unath/medical_mae_original_models/ablation_models/checkpoint-best_auc_vit_small_patch16_50_%.pth \
--epochs 40 --blr 2.5e-5 --layer_decay 0.55 --weight_decay 0 --model vit_small_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/ChestX-ray14/images --num_workers 8 \
--train_list ./data_splits/chestxray/train_official_50.txt --val_list ./data_splits/chestxray/val_official.txt \
--test_list ./data_splits/chestxray/test_official.txt --nb_classes 14 --eval_interval 1 \
--min_lr 1e-6 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' --softRank 38 --train_all --tn_loss_factor 0.00125

python main_classification.py --data_set ChestXray14  \
--proxy_dir Models/Classification/ChestXray14/Resnet50_Imagenet/Resnet50_Imagenet_run_0.pth.tar --mode test \
--data_dir /data/yyang409/rgoel15/ChestX-ray14/images \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt 

python -m torch.distributed.launch \
--master_port 12346 --nproc_per_node=4 \
--use_env main_finetune_chestxray.py \
--output_dir /data/yyang409/rgoel15/medical_mae_temp \
--log_dir /data/yyang409/rgoel15/medical_mae_temp \
--dataset chexpert \
--batch_size 256 --finetune /data/yyang409/rgoel15/medical_mae_original_models/pretrained/vit-s_CXR_0.3M_mae.pth \
--epochs 40 --blr 2.5e-4 --layer_decay 0.55 --weight_decay 0.05 --model vit_small_patch16 \
--warmup_epochs 5 --drop_path 0.2 --mixup 0 --cutmix 0 --reprob 0 --vit_dropout_rate 0 \
--data_path /data/yyang409/rgoel15/CheXpert-v1.0 --num_workers 8 \
--train_list /data/yyang409/rgoel15/CheXpert-v1.0/train.csv --val_list /data/yyang409/rgoel15/CheXpert-v1.0/valid.csv \
--test_list /data/yyang409/rgoel15/CheXpert-v1.0/test_labels.csv --nb_classes 14 --eval_interval 1 \
--min_lr 1e-5 --build_timm_transform --aa 'rand-m6-mstd0.5-inc1' \
--eval --eval_path /data/yyang409/rgoel15/medical_mae_original_models/finetuned/