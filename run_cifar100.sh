#!/bin/bash

python train.py --device_id 0 --method fedavg --init_with_fedavg false --dataset cifar100 --mode personal --rounds 40 --data_root ./data --output_dir ./results/fedavg_init_cifar100_c20_f0p8_resnet18 --save_checkpoint ./checkpoints/fedavg40_cifar100_c20_f0p8_resnet18.pt

python train.py --device_id 0 --method ours --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --device_id 0 --method only_importance --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --device_id 0 --method only_consistency --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --device_id 0 --method topk_params --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --device_id 0 --method server_only --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_importance --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_consistency --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_staleness --dataset cifar100 --mode personal --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

