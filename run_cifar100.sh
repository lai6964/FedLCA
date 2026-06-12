#!/bin/bash

python train.py --method fedavg --init_with_fedavg false --dataset cifar100 --personalized_eval True --train_fc_first True --rounds 40 --data_root ./data --output_dir ./results/fedavg_init_cifar100_c20_f0p8_resnet18 --save_checkpoint ./checkpoints/fedavg40_cifar100_c20_f0p8_resnet18.pt

python train.py --method ours --dataset cifar100 --personalized_eval True --train_fc_first True --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --method topk_params --dataset cifar100 --personalized_eval True --train_fc_first True --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --method server_only --dataset cifar100 --personalized_eval True --train_fc_first True --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --method wo_importance --dataset cifar100 --personalized_eval True --train_fc_first True --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --method wo_consistency --dataset cifar100 --personalized_eval True --train_fc_first True --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

python train.py --method wo_staleness --dataset cifar100 --personalized_eval True --train_fc_first True --data_root ./data --output_dir ./results/cifar100_c20_f0p8_resnet18

