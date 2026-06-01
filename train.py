import os
import copy
import math
import csv
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# 1. Basic Utils
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_tensors(tensors):
    if len(tensors) == 0:
        return torch.tensor([])
    return torch.cat([t.detach().reshape(-1).cpu() for t in tensors])


def cosine_similarity(x, y, eps=1e-12):
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    return float(torch.dot(x, y) / (torch.norm(x) * torch.norm(y) + eps))


def normalize_dict(d):
    values = np.array(list(d.values()), dtype=np.float64)
    if len(values) == 0:
        return d
    min_v, max_v = values.min(), values.max()
    if abs(max_v - min_v) < 1e-12:
        return {k: 1.0 for k in d}
    return {k: float((v - min_v) / (max_v - min_v)) for k, v in d.items()}


def ema_update(old, new, beta=0.9):
    return beta * old + (1.0 - beta) * new


# ============================================================
# 2. Model
# ============================================================

def build_model(num_classes=10):
    model = torchvision.models.resnet18(num_classes=num_classes)

    # 修改 ResNet-18 以适应 CIFAR 输入
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def get_layer_groups(model):
    """
    将模型划分为层级模块。
    对 ResNet-18:
    conv1/bn1, layer1, layer2, layer3, layer4, fc
    """
    groups = defaultdict(list)

    for name, param in model.named_parameters():
        if name.startswith("conv1") or name.startswith("bn1"):
            groups["stem"].append(name)
        elif name.startswith("layer1"):
            groups["layer1"].append(name)
        elif name.startswith("layer2"):
            groups["layer2"].append(name)
        elif name.startswith("layer3"):
            groups["layer3"].append(name)
        elif name.startswith("layer4"):
            groups["layer4"].append(name)
        elif name.startswith("fc"):
            groups["head"].append(name)
        else:
            groups["others"].append(name)

    return dict(groups)


def get_param_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_param_dict(model, param_dict):
    model.load_state_dict(param_dict, strict=True)


def get_group_tensors(param_dict, group_names, layer_groups):
    tensors = []
    for g in group_names:
        for name in layer_groups[g]:
            if name in param_dict:
                tensors.append(param_dict[name])
    return tensors


def get_single_group_tensor(param_dict, group, layer_groups):
    tensors = []
    for name in layer_groups[group]:
        if name in param_dict:
            tensors.append(param_dict[name])
    return flatten_tensors(tensors)


def set_trainable_layers(model, selected_groups, layer_groups):
    selected_param_names = set()
    for g in selected_groups:
        selected_param_names.update(layer_groups[g])

    for name, param in model.named_parameters():
        param.requires_grad = name in selected_param_names


def count_group_params(model, layer_groups):
    param_counts = {}
    named_params = dict(model.named_parameters())

    for g, names in layer_groups.items():
        total = 0
        for name in names:
            if name in named_params:
                total += named_params[name].numel()
        param_counts[g] = total

    return param_counts


# ============================================================
# 3. Dataset and Partition
# ============================================================

def load_dataset(dataset_name, data_root="../Dataset"):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )

    elif dataset_name == "cifar100":
        num_classes = 100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ])

        train_set = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_set, test_set, num_classes


def dirichlet_partition(dataset, num_clients, alpha, num_classes):
    """
    按标签使用 Dirichlet 分布构造 Non-IID 客户端数据。
    """
    targets = np.array(dataset.targets)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(
            alpha=np.repeat(alpha, num_clients)
        )
        proportions = proportions / proportions.sum()

        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, splits)

        for cid, idx in enumerate(class_splits):
            client_indices[cid].extend(idx.tolist())

    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

    return client_indices


def build_client_loaders(train_set, client_indices, batch_size, num_workers=2):
    loaders = []
    for indices in client_indices:
        subset = Subset(train_set, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders


def build_test_loader(test_set, batch_size=256, num_workers=2):
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# 4. Local Client Operations
# ============================================================

def compute_layer_importance(model, loader, device, layer_groups):
    """
    计算每层参数重要性：
    I_l = mean(|theta_l * grad_l|)
    这里使用一个 batch 近似。
    """
    model.train()
    model.zero_grad()

    try:
        x, y = next(iter(loader))
    except StopIteration:
        return {g: 0.0 for g in layer_groups}

    x, y = x.to(device), y.to(device)

    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    importance = {}
    named_params = dict(model.named_parameters())

    for g, names in layer_groups.items():
        values = []
        for name in names:
            if name in named_params:
                p = named_params[name]
                if p.grad is not None:
                    values.append(torch.abs(p.detach() * p.grad.detach()).mean())
        if len(values) == 0:
            importance[g] = 0.0
        else:
            importance[g] = float(torch.stack(values).mean().detach().cpu())

    model.zero_grad()
    return importance


def train_local(
    global_params,
    model_fn,
    loader,
    selected_layers,
    layer_groups,
    device,
    local_epochs=1,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
):
    """
    客户端本地训练。
    只训练 selected_layers，其余层冻结。
    """
    model = model_fn().to(device)
    load_param_dict(model, global_params)

    set_trainable_layers(model, selected_layers, layer_groups)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model.train()
    total_loss = 0.0
    total_num = 0

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * y.size(0)
            total_num += y.size(0)

    new_params = get_param_dict(model)

    # 仅上传被选中层的更新
    updates = {}
    for g in selected_layers:
        for name in layer_groups[g]:
            if name in new_params:
                updates[name] = new_params[name] - global_params[name]

    avg_loss = total_loss / max(total_num, 1)
    return updates, avg_loss


def estimate_local_consistency(
    local_importance,
    server_consistency,
    candidate_layers,
):
    """
    简化实现：
    客户端使用服务器维护的层级一致性统计作为局部一致性先验。
    如果后续想更严格，可以让客户端用本地梯度方向与最近全局层更新方向计算余弦相似度。
    """
    local_cons = {}
    for g in candidate_layers:
        local_cons[g] = server_consistency.get(g, 1.0)
    return local_cons


def client_select_layers(
    candidate_layers,
    global_scores,
    local_importance,
    local_consistency,
    group_param_counts,
    resource_budget,
    compute_power,
    bandwidth,
    min_layers=1,
):
    """
    客户端自适应选择层。
    选择分数：
    score_i,l = global_score_l * local_importance_i,l * local_consistency_i,l / cost_i,l

    cost_i,l = params_l / bandwidth_i + params_l / compute_power_i
    """
    norm_local_imp = normalize_dict({
        g: local_importance.get(g, 0.0) for g in candidate_layers
    })

    items = []

    for g in candidate_layers:
        params = group_param_counts[g]

        comm_cost = params / max(bandwidth, 1e-12)
        comp_cost = params / max(compute_power, 1e-12)
        cost = comm_cost + comp_cost

        value = (
            global_scores.get(g, 1.0)
            * norm_local_imp.get(g, 1.0)
            * local_consistency.get(g, 1.0)
        )

        ratio = value / (cost + 1e-12)
        items.append((g, ratio, cost))

    items = sorted(items, key=lambda x: x[1], reverse=True)

    selected = []
    used_budget = 0.0

    for g, ratio, cost in items:
        if used_budget + cost <= resource_budget:
            selected.append(g)
            used_budget += cost

    # 防止客户端一层都不训练
    if len(selected) < min_layers and len(items) > 0:
        selected = [items[0][0]]

    return selected, used_budget


# ============================================================
# 5. Server Selection and Aggregation
# ============================================================

def server_select_candidate_layers(
    layer_groups,
    importance_stats,
    consistency_stats,
    last_selected_round,
    current_round,
    layer_budget,
    staleness_threshold,
):
    """
    服务器候选层选择：
    1. 重要性和一致性先归一化；
    2. score_l = importance_l * consistency_l；
    3. 陈旧度超过阈值的层优先进入候选集合；
    4. 剩余位置由 score 最高的层填充。
    """
    groups = list(layer_groups.keys())

    norm_imp = normalize_dict({g: importance_stats.get(g, 0.0) for g in groups})
    norm_cons = normalize_dict({g: consistency_stats.get(g, 1.0) for g in groups})

    scores = {}
    for g in groups:
        scores[g] = norm_imp[g] * norm_cons[g]

    stale_layers = []
    for g in groups:
        stale = current_round - last_selected_round.get(g, -1)
        if stale >= staleness_threshold:
            stale_layers.append(g)

    stale_layers = sorted(stale_layers, key=lambda x: scores[x], reverse=True)
    selected = stale_layers[:layer_budget]

    if len(selected) < layer_budget:
        remaining = [g for g in groups if g not in selected]
        remaining = sorted(remaining, key=lambda x: scores[x], reverse=True)
        selected += remaining[: layer_budget - len(selected)]

    for g in selected:
        last_selected_round[g] = current_round

    return selected, scores


def aggregate_layer_updates(
    global_params,
    client_updates,
    client_weights,
    layer_groups,
    min_clients_per_layer=1,
    small_layer_lr=0.5,
):
    """
    分层聚合。
    每层只聚合实际上传该层更新的客户端。
    如果某层参与客户端过少，可以降低聚合步长。
    """
    new_global = copy.deepcopy(global_params)
    layer_client_count = {}

    for g, names in layer_groups.items():
        available_clients = []
        for idx, updates in enumerate(client_updates):
            has_group = any(name in updates for name in names)
            if has_group:
                available_clients.append(idx)

        layer_client_count[g] = len(available_clients)

        if len(available_clients) == 0:
            continue

        total_weight = sum(client_weights[i] for i in available_clients)
        if total_weight <= 0:
            continue

        layer_lr = 1.0
        if len(available_clients) < min_clients_per_layer:
            layer_lr = small_layer_lr

        for name in names:
            if name not in new_global:
                continue

            agg_update = torch.zeros_like(new_global[name])

            for i in available_clients:
                if name in client_updates[i]:
                    w = client_weights[i] / total_weight
                    agg_update += w * client_updates[i][name]

            new_global[name] = new_global[name] + layer_lr * agg_update

    return new_global, layer_client_count


def update_server_statistics(
    old_global,
    new_global,
    client_updates,
    layer_groups,
    importance_uploads,
    importance_stats,
    consistency_stats,
    beta=0.9,
):
    """
    更新服务器维护的层级统计量：
    - 参数重要性：客户端上传标量后 EMA；
    - 更新一致性：客户端层更新与本轮全局层更新方向的余弦相似度；
    """
    groups = list(layer_groups.keys())

    # 更新重要性
    for g in groups:
        vals = []
        for imp in importance_uploads:
            if g in imp:
                vals.append(imp[g])
        if len(vals) > 0:
            avg_imp = float(np.mean(vals))
            importance_stats[g] = ema_update(
                importance_stats.get(g, 0.0), avg_imp, beta=beta
            )

    # 更新一致性
    for g in groups:
        global_delta = get_single_group_tensor(new_global, g, layer_groups) - \
                       get_single_group_tensor(old_global, g, layer_groups)

        vals = []
        for updates in client_updates:
            local_parts = []
            for name in layer_groups[g]:
                if name in updates:
                    local_parts.append(updates[name])
            if len(local_parts) == 0:
                continue

            local_delta = flatten_tensors(local_parts)
            cos = cosine_similarity(local_delta, global_delta)

            # 归一化到 [0, 1]
            cos01 = (cos + 1.0) / 2.0

            # 幅度因子：避免极小更新获得过高一致性
            local_norm = float(torch.norm(local_delta) + 1e-12)
            global_norm = float(torch.norm(global_delta) + 1e-12)
            mag = min(local_norm / global_norm, global_norm / local_norm)

            vals.append(cos01 * mag)

        if len(vals) > 0:
            avg_cons = float(np.mean(vals))
            consistency_stats[g] = ema_update(
                consistency_stats.get(g, 1.0), avg_cons, beta=beta
            )

    return importance_stats, consistency_stats


# ============================================================
# 6. Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, params, test_loader, device):
    model.to(device)
    load_param_dict(model, params)
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)

        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.size(0)
        loss_sum += float(loss.detach().cpu()) * y.size(0)

    acc = correct / max(total, 1)
    loss = loss_sum / max(total, 1)

    return acc, loss


# ============================================================
# 7. Resource Simulation
# ============================================================

def generate_client_resources(num_clients, mode="moderate"):
    """
    模拟客户端资源异构。
    compute_power 和 bandwidth 越大，训练大层代价越低。
    budget 越大，客户端可选择更多层。
    """
    resources = {}

    for cid in range(num_clients):
        r = random.random()

        if mode == "mild":
            compute_power = random.uniform(0.8, 1.2)
            bandwidth = random.uniform(0.8, 1.2)
            budget = random.uniform(0.8, 1.2)

        elif mode == "moderate":
            if r < 0.3:
                compute_power = random.uniform(0.3, 0.6)
                bandwidth = random.uniform(0.3, 0.6)
                budget = random.uniform(0.3, 0.6)
            elif r < 0.7:
                compute_power = random.uniform(0.7, 1.0)
                bandwidth = random.uniform(0.7, 1.0)
                budget = random.uniform(0.7, 1.0)
            else:
                compute_power = random.uniform(1.2, 2.0)
                bandwidth = random.uniform(1.2, 2.0)
                budget = random.uniform(1.2, 2.0)

        elif mode == "severe":
            if r < 0.5:
                compute_power = random.uniform(0.15, 0.4)
                bandwidth = random.uniform(0.15, 0.4)
                budget = random.uniform(0.15, 0.4)
            elif r < 0.8:
                compute_power = random.uniform(0.6, 1.0)
                bandwidth = random.uniform(0.6, 1.0)
                budget = random.uniform(0.6, 1.0)
            else:
                compute_power = random.uniform(1.5, 3.0)
                bandwidth = random.uniform(1.5, 3.0)
                budget = random.uniform(1.5, 3.0)
        else:
            raise ValueError(f"Unknown resource mode: {mode}")

        resources[cid] = {
            "compute_power": compute_power,
            "bandwidth": bandwidth,
            "budget": budget,
        }

    return resources


def calibrate_resource_budget(resources, group_param_counts, layer_budget):
    """
    将抽象 budget 映射到真实参数规模。
    这里用平均层代价作为尺度。
    """
    avg_params = np.mean(list(group_param_counts.values()))
    calibrated = {}

    for cid, res in resources.items():
        compute_power = res["compute_power"]
        bandwidth = res["bandwidth"]

        avg_cost = avg_params / bandwidth + avg_params / compute_power
        real_budget = res["budget"] * avg_cost * layer_budget

        calibrated[cid] = {
            "compute_power": compute_power,
            "bandwidth": bandwidth,
            "budget": real_budget,
        }

    return calibrated


# ============================================================
# 8. Main Training
# ============================================================

def run_experiment(args):
    set_seed(args.seed)
    device = get_device()
    print(device)

    train_set, test_set, num_classes = load_dataset(args.dataset, args.data_root)

    client_indices = dirichlet_partition(
        dataset=train_set,
        num_clients=args.num_clients,
        alpha=args.alpha,
        num_classes=num_classes,
    )

    client_loaders = build_client_loaders(
        train_set,
        client_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = build_test_loader(
        test_set,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    model_fn = lambda: build_model(num_classes=num_classes)

    global_model = model_fn()
    layer_groups = get_layer_groups(global_model)
    group_param_counts = count_group_params(global_model, layer_groups)

    global_params = get_param_dict(global_model)

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

    total_params = sum(group_param_counts.values())

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
        # 服务器候选层选择
        # ====================================================
        if args.method == "fedavg":
            candidate_layers = groups
            global_scores = {g: 1.0 for g in groups}

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
                layer_groups=layer_groups,
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
        # 客户端本地训练
        # ====================================================
        for cid in selected_clients:
            print("begin clint {}".format(cid))
            loader = client_loaders[cid]
            num_samples = len(client_indices[cid])
            client_weights.append(num_samples)

            # 客户端接收模型
            if args.method == "fedavg":
                downloaded = total_params
            else:
                downloaded = sum(group_param_counts[g] for g in candidate_layers)
            round_download_params += downloaded

            if args.method == "fedavg":
                local_importance = {}
                local_consistency = {}
                selected_layers = groups

            elif args.method in ["server_only", "wo_client_adaptive"]:
                temp_model = model_fn().to(device)
                load_param_dict(temp_model, global_params)

                print("computing importance")
                local_importance = compute_layer_importance(
                    temp_model,
                    loader,
                    device,
                    layer_groups,
                )

                print("computing consistency")
                local_consistency = estimate_local_consistency(
                    local_importance=local_importance,
                    server_consistency=consistency_stats,
                    candidate_layers=candidate_layers,
                )

                selected_layers = candidate_layers

            else:
                temp_model = model_fn().to(device)
                load_param_dict(temp_model, global_params)

                print("computing importance")
                local_importance = compute_layer_importance(
                    temp_model,
                    loader,
                    device,
                    layer_groups,
                )

                print("computing consistency")
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

            print("training local model")
            updates, train_loss = train_local(
                global_params=global_params,
                model_fn=model_fn,
                loader=loader,
                selected_layers=selected_layers,
                layer_groups=layer_groups,
                device=device,
                local_epochs=args.local_epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
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
        # 服务器分层聚合
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

        # ====================================================
        # 更新服务器层级统计量
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
            print("begin evaluation at {} round".format(rnd))
            eval_model = model_fn()
            acc, test_loss = evaluate(
                eval_model,
                global_params,
                test_loader,
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


# ============================================================
# 9. Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../Dataset")
    parser.add_argument("--output_dir", type=str, default="./results")

    parser.add_argument("--method", type=str, default="ours",
                        choices=[
                            "fedavg",
                            "server_only",
                            "ours",
                            "wo_importance",
                            "wo_consistency",
                            "wo_staleness",
                            "wo_client_adaptive",
                        ])

    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--client_frac", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--server_layer_budget", type=int, default=4)
    parser.add_argument("--client_layer_budget", type=int, default=2)
    parser.add_argument("--staleness_threshold", type=int, default=5)

    parser.add_argument("--min_clients_per_layer", type=int, default=2)
    parser.add_argument("--small_layer_lr", type=float, default=0.5)

    parser.add_argument("--stat_beta", type=float, default=0.9)
    parser.add_argument("--resource_mode", type=str, default="moderate",
                        choices=["mild", "moderate", "severe"])

    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.method = "fedavg"
    run_experiment(args)
