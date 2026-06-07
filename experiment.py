import copy
import csv
import os

import numpy as np
import torch

from client import (
    client_select_layers,
    compute_layer_importance,
    estimate_local_consistency,
    train_local,
    train_local_head,
)
from data import build_client_loaders, load_dataset, pfl_partition
from evaluation import evaluate, evaluate_personalized
from models import (
    build_model,
    count_group_params,
    get_layer_groups,
    get_param_dict,
    load_param_dict,
    load_param_dict_excluding,
    set_trainable_layers,
)
from resources import calibrate_resource_budget, generate_client_resources
from server import aggregate_layer_updates, server_select_candidate_layers, update_server_statistics
from utils import get_device, set_seed

def run_experiment(args):
    set_seed(args.seed)
    device = get_device()
    print(device)

    train_set, test_set, num_classes = load_dataset(
        args.dataset,
        args.data_root,
        download=args.download_data,
    )

    client_indices, client_test_indices = pfl_partition(
        dataset=train_set,
        num_clients=args.num_clients,
        alpha=args.alpha,
        num_classes=num_classes,
        train_ratio=args.train_ratio,
    )

    client_loaders = build_client_loaders(
        train_set,
        client_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    client_test_loaders = build_client_loaders(
        test_set,
        client_test_indices,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model_fn = lambda: build_model(
        num_classes=num_classes,
        model_name=args.model,
    )

    global_model = model_fn()
    layer_groups = get_layer_groups(global_model)
    group_param_counts = count_group_params(global_model, layer_groups)

    if args.init_with_fedavg and args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        checkpoint_params = checkpoint.get("model_state", checkpoint)
        load_param_dict(global_model, checkpoint_params)
        print(f"Loaded initial model from: {args.init_checkpoint}")
    elif args.init_with_fedavg and args.method != "fedavg":
        raise ValueError(
            "init_with_fedavg=True requires --init_checkpoint for non-FedAvg methods"
        )
    else:
        print("Using randomly initialized model")

    global_params = get_param_dict(global_model)
    client_models = [model_fn().to(device) for _ in range(args.num_clients)]
    for client_model in client_models:
        load_param_dict(client_model, global_params)

    resources = generate_client_resources(
        args.num_clients,
        mode=args.resource_mode,
    )
    resources = calibrate_resource_budget(
        resources,
        group_param_counts,
        layer_budget=args.client_layer_budget,
    )

    groups = list(layer_groups.keys())
    trainable_groups = [
        group for group in groups
        if not (args.train_fc_first and group == "head")
    ]
    if args.train_fc_first:
        print(
            "train_fc_first=True: head stays local and global-model "
            "evaluation does not measure personalized head accuracy"
        )

    importance_stats = {g: 1.0 for g in groups}
    consistency_stats = {g: 1.0 for g in groups}
    last_selected_round = {g: -1 for g in groups}

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.method}_{args.dataset}_a{args.alpha}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "accuracy",
            "test_loss",
            "avg_train_loss",
            "upload_params",
            "download_params",
            "compute_params",
            "candidate_layers",
            "avg_selected_layers",
        ])

    total_params = sum(group_param_counts[group] for group in trainable_groups)

    for rnd in range(1, args.rounds + 1):
        selected_clients = np.random.choice(
            args.num_clients,
            size=max(1, int(args.num_clients * args.client_frac)),
            replace=False,
        ).tolist()

        old_global_params = None
        if args.method != "fedavg":
            old_global_params = copy.deepcopy(global_params)

        # ====================================================
        # 鏈嶅姟鍣ㄥ€欓€夊眰閫夋嫨
        # ====================================================
        if args.method == "fedavg":
            candidate_layers = trainable_groups
            global_scores = {g: 1.0 for g in trainable_groups}

        elif args.method in [
            "ours",
            "server_only",
            "wo_importance",
            "wo_consistency",
            "wo_staleness",
            "wo_client_adaptive",
        ]:
            imp_for_select = copy.deepcopy(importance_stats)
            cons_for_select = copy.deepcopy(consistency_stats)
            stale_threshold = args.staleness_threshold

            if args.method == "wo_importance":
                imp_for_select = {g: 1.0 for g in groups}

            if args.method == "wo_consistency":
                cons_for_select = {g: 1.0 for g in groups}

            if args.method == "wo_staleness":
                stale_threshold = 10 ** 9

            candidate_layers, global_scores = server_select_candidate_layers(
                layer_groups={
                    group: layer_groups[group] for group in trainable_groups
                },
                importance_stats=imp_for_select,
                consistency_stats=cons_for_select,
                last_selected_round=last_selected_round,
                current_round=rnd,
                layer_budget=args.server_layer_budget,
                staleness_threshold=stale_threshold,
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")

        client_updates = []
        client_weights = []
        importance_uploads = []
        train_losses = []

        round_upload_params = 0
        round_download_params = 0
        round_compute_params = 0
        selected_layer_nums = []

        # ====================================================
        # 瀹㈡埛绔湰鍦拌缁?
        # ====================================================
        for cid in selected_clients:
            # print("begin clint {}".format(cid))
            loader = client_loaders[cid]
            local_model = client_models[cid]
            num_samples = len(client_indices[cid])
            client_weights.append(num_samples)

            if args.train_fc_first:
                train_local_head(
                    global_params=global_params,
                    model=local_model,
                    loader=loader,
                    layer_groups=layer_groups,
                    device=device,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
                set_trainable_layers(local_model, trainable_groups, layer_groups)

            # 瀹㈡埛绔帴鏀舵ā鍨?
            if args.method == "fedavg":
                downloaded = total_params
            else:
                downloaded = sum(group_param_counts[g] for g in candidate_layers)
            round_download_params += downloaded

            if args.method == "fedavg":
                local_importance = {}
                local_consistency = {}
                selected_layers = trainable_groups

            elif args.method in ["server_only", "wo_client_adaptive"]:
                if not args.train_fc_first:
                    load_param_dict(local_model, global_params)

                # print("computing importance")
                local_importance = compute_layer_importance(
                    local_model,
                    loader,
                    device,
                    layer_groups,
                )

                # print("computing consistency")
                local_consistency = estimate_local_consistency(
                    local_importance=local_importance,
                    server_consistency=consistency_stats,
                    candidate_layers=candidate_layers,
                )

                selected_layers = candidate_layers

            else:
                if not args.train_fc_first:
                    load_param_dict(local_model, global_params)

                # print("computing importance")
                local_importance = compute_layer_importance(
                    local_model,
                    loader,
                    device,
                    layer_groups,
                )

                # print("computing consistency")
                local_consistency = estimate_local_consistency(
                    local_importance=local_importance,
                    server_consistency=consistency_stats,
                    candidate_layers=candidate_layers,
                )

                res = resources[cid]

                selected_layers, used_budget = client_select_layers(
                    candidate_layers=candidate_layers,
                    global_scores=global_scores,
                    local_importance=local_importance,
                    local_consistency=local_consistency,
                    group_param_counts=group_param_counts,
                    resource_budget=res["budget"],
                    compute_power=res["compute_power"],
                    bandwidth=res["bandwidth"],
                    min_layers=1,
                )

            # print("training local model")
            updates, train_loss = train_local(
                global_params=global_params,
                model=local_model,
                loader=loader,
                selected_layers=selected_layers,
                layer_groups=layer_groups,
                device=device,
                local_epochs=args.local_epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                preserve_local_groups=["head"] if args.train_fc_first else [],
            )

            upload_params = sum(
                group_param_counts[g] for g in selected_layers
            )
            compute_params = upload_params

            round_upload_params += upload_params
            round_compute_params += compute_params
            selected_layer_nums.append(len(selected_layers))

            client_updates.append(updates)
            importance_uploads.append(local_importance)
            train_losses.append(train_loss)

        # ====================================================
        # 鏈嶅姟鍣ㄥ垎灞傝仛鍚?
        # ====================================================
        if args.method == "fedavg":
            min_clients_per_layer = 1
            small_layer_lr = 1.0
        else:
            min_clients_per_layer = args.min_clients_per_layer
            small_layer_lr = args.small_layer_lr

        global_params, layer_client_count = aggregate_layer_updates(
            global_params=global_params,
            client_updates=client_updates,
            client_weights=client_weights,
            layer_groups=layer_groups,
            min_clients_per_layer=min_clients_per_layer,
            small_layer_lr=small_layer_lr,
        )

        if args.train_fc_first:
            head_names = set(layer_groups.get("head", []))
            for client_model in client_models:
                load_param_dict_excluding(client_model, global_params, head_names)
        else:
            for client_model in client_models:
                load_param_dict(client_model, global_params)

        # ====================================================
        # 鏇存柊鏈嶅姟鍣ㄥ眰绾х粺璁￠噺
        # ====================================================
        if args.method != "fedavg":
            importance_stats, consistency_stats = update_server_statistics(
                old_global=old_global_params,
                new_global=global_params,
                client_updates=client_updates,
                layer_groups=layer_groups,
                importance_uploads=importance_uploads,
                importance_stats=importance_stats,
                consistency_stats=consistency_stats,
                beta=args.stat_beta,
            )

        # ====================================================
        # Evaluation
        # ====================================================
        if rnd % args.eval_interval == 0 or rnd == args.rounds:
            # print("begin evaluation at {} round".format(rnd))
            if args.personalized_eval:
                test_sample_counts = [len(indices) for indices in client_test_indices]
                acc, test_loss = evaluate_personalized(
                    client_models,
                    global_params,
                    client_test_loaders,
                    device,
                    test_sample_counts,
                )
            else:
                eval_model = model_fn()
                all_test_indices = [
                    idx
                    for indices in client_test_indices
                    for idx in indices
                ]
                global_test_loader = build_client_loaders(
                    test_set,
                    [all_test_indices],
                    batch_size=args.test_batch_size,
                    num_workers=args.num_workers,
                    shuffle=False,
                )[0]
                acc, test_loss = evaluate(
                    eval_model,
                    global_params,
                    global_test_loader,
                    device,
                )

            avg_train_loss = float(np.mean(train_losses))
            avg_selected_layers = float(np.mean(selected_layer_nums))

            print(
                f"[Round {rnd:03d}] "
                f"Acc={acc * 100:.2f}% | "
                f"Loss={test_loss:.4f} | "
                f"TrainLoss={avg_train_loss:.4f} | "
                f"Upload={round_upload_params / 1e6:.2f}M | "
                f"Download={round_download_params / 1e6:.2f}M | "
                f"Compute={round_compute_params / 1e6:.2f}M | "
                f"Candidate={candidate_layers} | "
                f"AvgLayers={avg_selected_layers:.2f}"
            )

            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    rnd,
                    acc,
                    test_loss,
                    avg_train_loss,
                    round_upload_params,
                    round_download_params,
                    round_compute_params,
                    "|".join(candidate_layers),
                    avg_selected_layers,
                ])

    print(f"Results saved to: {log_path}")

    if args.save_checkpoint:
        checkpoint_dir = os.path.dirname(args.save_checkpoint)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(
            {
                "model_state": global_params,
                "dataset": args.dataset,
                "model": args.model,
                "method": args.method,
                "rounds": args.rounds,
                "num_clients": args.num_clients,
                "client_frac": args.client_frac,
                "seed": args.seed,
            },
            args.save_checkpoint,
        )
        print(f"Checkpoint saved to: {args.save_checkpoint}")


