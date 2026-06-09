#!/bin/bash

DATASET=${1:-cifar100}
DEVICE_ID=${2:-0}
OUTPUT_ROOT=${3:-./results}

python train.py --model resnet18 --method fedavg --dataset "${DATASET}" --device_id "${DEVICE_ID}" --num_clients 20 --client_frac 0.8 --data_root ./data --output_dir "${OUTPUT_ROOT}/${DATASET}_c20_f0p8_resnet18/fedavg"
sleep 5
python train.py --model resnet18 --method server_only --dataset "${DATASET}" --device_id "${DEVICE_ID}" --num_clients 20 --client_frac 0.8 --data_root ./data --output_dir "${OUTPUT_ROOT}/${DATASET}_c20_f0p8_resnet18/server_only"
sleep 5
python train.py --model resnet18 --method ours --dataset "${DATASET}" --device_id "${DEVICE_ID}" --num_clients 20 --client_frac 0.8 --data_root ./data --output_dir "${OUTPUT_ROOT}/${DATASET}_c20_f0p8_resnet18/ours"
sleep 5
python train.py --model resnet18 --method wo_importance --dataset "${DATASET}" --device_id "${DEVICE_ID}" --num_clients 20 --client_frac 0.8 --data_root ./data --output_dir "${OUTPUT_ROOT}/${DATASET}_c20_f0p8_resnet18/wo_importance"
sleep 5
python train.py --model resnet18 --method wo_consistency --dataset "${DATASET}" --device_id "${DEVICE_ID}" --num_clients 20 --client_frac 0.8 --data_root ./data --output_dir "${OUTPUT_ROOT}/${DATASET}_c20_f0p8_resnet18/wo_consistency"
sleep 5
python train.py --model resnet18 --method wo_staleness --dataset "${DATASET}" --device_id "${DEVICE_ID}" --num_clients 20 --client_frac 0.8 --data_root ./data --output_dir "${OUTPUT_ROOT}/${DATASET}_c20_f0p8_resnet18/wo_staleness"
sleep 5
python train.py --model resnet18 --method wo_client_adaptive --dataset "${DATASET}" --device_id "${DEVICE_ID}" --num_clients 20 --client_frac 0.8 --data_root ./data --output_dir "${OUTPUT_ROOT}/${DATASET}_c20_f0p8_resnet18/wo_client_adaptive"
