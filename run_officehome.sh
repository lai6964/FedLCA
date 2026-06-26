#!/bin/bash
cd "$(dirname "$0")"

#python train.py --device_id 0 --model resnet18 --method fedavg --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 40 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/fedavg_init --save_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt --init_with_fedavg false
#sleep 5
#
#python train.py --device_id 0 --model resnet18 --method fedavg --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/main/fedavg --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
#sleep 5
#
#python train.py --device_id 0 --model resnet18 --method topk_params --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/main/topk_params --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
#sleep 5
#
#python train.py --device_id 0 --model resnet18 --method sequential_conv --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/main/sequential_conv --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
#sleep 5
#
#python train.py --device_id 0 --model resnet18 --method server_top_importance_conv --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/main/server_top_importance_conv --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
#sleep 5

python train.py --device_id 0 --model resnet18 --method ours --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/main/ours --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method wo_importance --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/core/wo_importance --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method wo_consistency --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/core/wo_consistency --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method wo_staleness --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/core/wo_staleness --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method wo_client_adaptive --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/core/wo_client_adaptive --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method server_only --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/core/server_only --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method only_importance --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/core/only_importance --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method only_consistency --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/core/only_consistency --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method ours --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 1 --data_root ./data --output_dir ./results/ablations/officehome/sensitivity/ours_local_epochs1 --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method ours --dataset officehome --mode traditional --alpha 0.5 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --data_root ./data --output_dir ./results/ablations/officehome/sensitivity/ours_alpha0p5 --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method ours --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --server_layer_budget 1 --data_root ./data --output_dir ./results/ablations/officehome/sensitivity/ours_server_budget1 --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt
sleep 5

python train.py --device_id 0 --model resnet18 --method ours --dataset officehome --mode traditional --alpha 0.1 --num_clients 20 --client_frac 0.8 --rounds 200 --local_epochs 5 --client_layer_budget 1 --data_root ./data --output_dir ./results/ablations/officehome/sensitivity/ours_client_budget1 --init_with_fedavg true --init_checkpoint ./checkpoints/fedavg40_officehome_a0p1_c20_f0p8_resnet18.pt

echo Done.
