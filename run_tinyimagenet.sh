#!/bin/bash

python train.py --device_id 0 --method fedavg --dataset tinyimagenet --alpha 0.1 --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18_a0p1/fedavg

python train.py --device_id 0 --method topk_params --dataset tinyimagenet --alpha 0.1 --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18_a0p1/topk_params

python train.py --device_id 0 --method ours --dataset tinyimagenet --alpha 0.1 --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18_a0p1/ours

python train.py --device_id 0 --method fedavg --dataset tinyimagenet --alpha 0.5 --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18_a0p5/fedavg

python train.py --device_id 0 --method topk_params --dataset tinyimagenet --alpha 0.5 --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18_a0p5/topk_params

python train.py --device_id 0 --method ours --dataset tinyimagenet --alpha 0.5 --data_root ./data --output_dir ./results/tinyimagenet_c20_f0p8_resnet18_a0p5/ours
