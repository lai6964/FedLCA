import copy
import csv
import os

import numpy as np
import torch

from client import (
    build_topk_param_upload,
    client_select_layers,
    compute_layer_importance,
    estimate_local_consistency,
    train_local,
    train_local_head,
)
from data import build_client_loaders, load_dataset, load_or_create_pfl_partition
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
from server import (
    aggregate_layer_updates,
    aggregate_topk_param_updates,
    server_select_candidate_layers,
    server_select_top_importance_conv_layer,
    update_server_statistics,
)
from utils import get_device, set_seed


def count_layer_update_clients(client_updates, layer_groups):
    layer_counts = {group: 0 for group in layer_groups}
    for updates in client_updates:
        updated_names = set(updates.keys())
        for group, names in layer_groups.items():
            if any(name in updated_names for name in names):
                layer_counts[group] += 1
    return layer_counts


def run_experiment(args):
    set_seed(args.seed)
    device = get_device(args.device_id)
    print(device)

    train_set, test_set, num_classes = load_dataset(
        args.dataset,
        args.data_root,
        download=args.download_data,
    )

    client_indices, client_test_indices = load_or_create_pfl_partition(
        dataset=train_set,
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        num_classes=num_classes,
        train_ratio=args.train_ratio,
        seed=args.seed,
        partition_dir=args.partition_dir,
        regenerate=args.regenerate_partition,
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
    all_train_indices = [idx for indices in client_indices for idx in indices]
    server_loader = build_client_loaders(
        train_set,
        [all_train_indices],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )[0]

    model_fn = lambda: build_model(
        num_classes=num_classes,
        model_name=args.model,
        dataset_name=args.dataset,
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

    groups = list(layer_groups.keys())
    trainable_groups = [
        group for group in groups
        if not (args.mode == "personal" and group == "head")
    ]
    always_sync_groups = []
    if args.mode == "traditional" and "head" in trainable_groups:
        always_sync_groups = ["head"]
    selectable_groups = [
        group for group in trainable_groups
        if group not in always_sync_groups
    ]
    selectable_group_param_counts = {
        group: group_param_counts[group] for group in selectable_groups
    }
    if len(selectable_group_param_counts) == 0:
        selectable_group_param_counts = {
            group: group_param_counts[group] for group in trainable_groups
        }
    resources = calibrate_resource_budget(
        resources,
        selectable_group_param_counts,
        layer_budget=args.client_layer_budget,
    )

    def add_always_sync_groups(selected_groups):
        merged = list(selected_groups)
        for group in always_sync_groups:
            if group not in merged:
                merged.append(group)
        return merged

    def count_selected_layers(selected_groups):
        return len([
            group for group in selected_groups
            if group not in always_sync_groups
        ])

    sequential_conv_groups = [
        group for group in ["stem", "layer1", "layer2", "layer3", "layer4"]
        if group in selectable_groups
    ]
    if len(sequential_conv_groups) == 0:
        sequential_conv_groups = [
            group for group in selectable_groups if group != "head"
        ]
    if args.mode == "personal":
        print(
            "mode=personal: head stays local, FC is trained before local "
            "updates, and personalized evaluation is used"
        )
    elif always_sync_groups:
        print(
            "mode=traditional: head is synchronized every round for sparse "
            "layer-selection methods"
        )

    importance_stats = {g: 1.0 for g in groups}
    consistency_stats = {g: 1.0 for g in groups}
    last_selected_round = {g: -1 for g in groups}

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.method}_{args.dataset}_a{args.alpha}.csv")
    layer_count_log_path = os.path.join(
        args.output_dir,
        f"{args.method}_{args.dataset}_a{args.alpha}_layer_counts.csv",
    )
    layer_count_columns = [f"{group}_update_clients" for group in groups]

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
            *layer_count_columns,
        ])

    with open(layer_count_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "candidate_layers",
            "selected_clients",
            *layer_count_columns,
        ])

    total_params = sum(group_param_counts[group] for group in trainable_groups)

    for rnd in range(1, args.rounds + 1):
        selected_clients = np.random.choice(
            args.num_clients,
            size=max(1, int(args.num_clients * args.client_frac)),
            replace=False,
        ).tolist()

        old_global_params = None
        if args.method not in [
            "fedavg",
            "topk_params",
            "sequential_conv",
            "server_top_importance_conv",
        ]:
            old_global_params = copy.deepcopy(global_params)

        # ====================================================
        # 鏈嶅姟鍣ㄥ€欓€夊眰閫夋嫨
        # ====================================================
        if args.method == "fedavg":
            candidate_layers = trainable_groups
            global_scores = {g: 1.0 for g in trainable_groups}

        elif args.method == "topk_params":
            candidate_layers = trainable_groups
            global_scores = {g: 1.0 for g in trainable_groups}

        elif args.method == "sequential_conv":
            selected_group = sequential_conv_groups[
                (rnd - 1) % len(sequential_conv_groups)
            ]
            candidate_layers = [selected_group]
            global_scores = {selected_group: 1.0}

        elif args.method == "server_top_importance_conv":
            candidate_layers, global_scores = server_select_top_importance_conv_layer(
                model=model_fn(),
                params=global_params,
                loader=server_loader,
                device=device,
                layer_groups=layer_groups,
                candidate_groups=sequential_conv_groups,
            )

        elif args.method in [
            "ours",
            "server_only",
            "only_importance",
            "only_consistency",
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

            if args.method == "only_importance":
                cons_for_select = {g: 1.0 for g in groups}

            if args.method == "only_consistency":
                imp_for_select = {g: 1.0 for g in groups}

            if args.method == "wo_staleness":
                stale_threshold = 10 ** 9

            candidate_layers, global_scores = server_select_candidate_layers(
                layer_groups={
                    group: layer_groups[group] for group in selectable_groups
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

            if args.mode == "personal":
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
            if args.method in ["fedavg", "topk_params"]:
                downloaded = total_params
            elif args.method in ["sequential_conv", "server_top_importance_conv"]:
                download_layers = add_always_sync_groups(candidate_layers)
                downloaded = sum(group_param_counts[g] for g in download_layers)
            else:
                download_layers = add_always_sync_groups(candidate_layers)
                downloaded = sum(group_param_counts[g] for g in download_layers)
            round_download_params += downloaded

            if args.method == "fedavg":
                local_importance = {}
                local_consistency = {}
                selected_layers = trainable_groups

            elif args.method == "topk_params":
                local_importance = {}
                local_consistency = {}
                selected_layers = trainable_groups

            elif args.method in ["sequential_conv", "server_top_importance_conv"]:
                local_importance = {}
                local_consistency = {}
                selected_layers = add_always_sync_groups(candidate_layers)

            elif args.method in ["server_only", "wo_client_adaptive"]:
                if args.mode != "personal":
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
                selected_layers = add_always_sync_groups(selected_layers)

            else:
                if args.mode != "personal":
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
                selected_layers = add_always_sync_groups(selected_layers)

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
                preserve_local_groups=["head"] if args.mode == "personal" else [],
            )

            if args.method == "topk_params":
                updates, upload_params = build_topk_param_upload(
                    global_params=global_params,
                    model=local_model,
                    loader=loader,
                    device=device,
                    layer_groups=layer_groups,
                    candidate_layers=trainable_groups,
                    topk_ratio=args.topk_param_ratio,
                    dense_layers=["head"] if "head" in trainable_groups else [],
                )
            else:
                upload_params = sum(
                    group_param_counts[g] for g in selected_layers
                )

            compute_params = sum(group_param_counts[g] for g in selected_layers)

            round_upload_params += upload_params
            round_compute_params += compute_params
            selected_layer_nums.append(count_selected_layers(selected_layers))

            client_updates.append(updates)
            importance_uploads.append(local_importance)
            train_losses.append(train_loss)

        round_layer_counts = count_layer_update_clients(
            client_updates,
            layer_groups,
        )

        # ====================================================
        # 鏈嶅姟鍣ㄥ垎灞傝仛鍚?
        # ====================================================
        if args.method == "topk_params":
            global_params, layer_client_count = aggregate_topk_param_updates(
                global_params=global_params,
                client_uploads=client_updates,
                client_weights=client_weights,
            )
        else:
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

        if args.method in [
            "ours",
            "server_only",
            "only_importance",
            "only_consistency",
            "wo_importance",
            "wo_consistency",
            "wo_staleness",
            "wo_client_adaptive",
        ]:
            for group in selectable_groups:
                if round_layer_counts.get(group, 0) > 0:
                    last_selected_round[group] = rnd

        with open(layer_count_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                rnd,
                "|".join(candidate_layers),
                "|".join(str(cid) for cid in selected_clients),
                *[round_layer_counts[group] for group in groups],
            ])

        if args.mode == "personal":
            head_names = set(layer_groups.get("head", []))
            for client_model in client_models:
                load_param_dict_excluding(client_model, global_params, head_names)
        else:
            for client_model in client_models:
                load_param_dict(client_model, global_params)

        # ====================================================
        # 鏇存柊鏈嶅姟鍣ㄥ眰绾х粺璁￠噺
        # ====================================================
        if args.method not in [
            "fedavg",
            "topk_params",
            "sequential_conv",
            "server_top_importance_conv",
        ]:
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
            if args.mode == "personal":
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
            layer_count_text = ",".join(
                f"{group}:{round_layer_counts[group]}" for group in groups
            )

            print(
                f"[Round {rnd:03d}] "
                f"Acc={acc * 100:.2f}% | "
                f"Loss={test_loss:.4f} | "
                f"TrainLoss={avg_train_loss:.4f} | "
                f"Upload={round_upload_params / 1e6:.2f}M | "
                f"Download={round_download_params / 1e6:.2f}M | "
                f"Compute={round_compute_params / 1e6:.2f}M | "
                f"Candidate={candidate_layers} | "
                f"AvgLayers={avg_selected_layers:.2f} | "
                f"LayerCounts={{{layer_count_text}}}"
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
                    *[round_layer_counts[group] for group in groups],
                ])

    print(f"Results saved to: {log_path}")
    print(f"Layer counts saved to: {layer_count_log_path}")

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


