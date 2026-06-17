#!/bin/bash

python train.py --device_id 0 --method fedavg --init_with_fedavg false --dataset cifar10 --mode personal --rounds 40 --data_root ./data --output_dir ./results/fedavg_init_cifar10_c20_f0p8_resnet18 --save_checkpoint ./checkpoints/fedavg40_cifar10_c20_f0p8_resnet18.pt

python train.py --device_id 0 --method ours --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

python train.py --device_id 0 --method only_importance --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

python train.py --device_id 0 --method only_consistency --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

python train.py --device_id 0 --method topk_params --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

python train.py --device_id 0 --method server_only --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_importance --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_consistency --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_staleness --dataset cifar10 --mode personal --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18

