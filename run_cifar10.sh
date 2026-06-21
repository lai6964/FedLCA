#!/bin/bash

python train.py --device_id 0 --method sequential_conv --dataset cifar10 --alpha 0.1 --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18_a0p1/sequential_conv

python train.py --device_id 0 --method server_top_importance_conv --dataset cifar10 --alpha 0.1 --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18_a0p1/server_top_importance_conv

python train.py --device_id 0 --method sequential_conv --dataset cifar10 --alpha 0.5 --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18_a0p5/sequential_conv

python train.py --device_id 0 --method server_top_importance_conv --dataset cifar10 --alpha 0.5 --data_root ./data --output_dir ./results/cifar10_c20_f0p8_resnet18_a0p5/server_top_importance_conv
