#!/bin/bash

python train.py --device_id 0 --method fedavg --init_with_fedavg false --dataset tinyimagenet --mode personal --rounds 40 --data_root ./data --output_dir ./results/fedavg_init_tinyimagenet_c20_f0p8_resnet18 --save_checkpoint ./checkpoints/fedavg40_tinyimagenet_c20_f0p8_resnet18.pt

python train.py --device_id 0 --method ours --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18

python train.py --device_id 0 --method only_importance --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18

python train.py --device_id 0 --method only_consistency --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18

python train.py --device_id 0 --method topk_params --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18

python train.py --device_id 0 --method server_only --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_importance --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_consistency --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_staleness --dataset tinyimagenet --mode personal --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18
