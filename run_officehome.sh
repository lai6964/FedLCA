#!/bin/bash

python train.py --device_id 0 --method fedavg --init_with_fedavg false --dataset officehome --mode personal --rounds 40 --data_root ./data --output_dir ./results/fedavg_init_officehome_c20_f0p8_resnet18 --save_checkpoint ./checkpoints/fedavg40_officehome_c20_f0p8_resnet18.pt

python train.py --device_id 0 --method ours --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18

python train.py --device_id 0 --method only_importance --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18

python train.py --device_id 0 --method only_consistency --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18

python train.py --device_id 0 --method topk_params --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18

python train.py --device_id 0 --method server_only --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_importance --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_consistency --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18

python train.py --device_id 0 --method wo_staleness --dataset officehome --mode personal --data_root ./data --output_dir ./results/officehome_c20_f0p8_resnet18
