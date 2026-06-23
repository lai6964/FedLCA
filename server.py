import copy

import numpy as np
import torch
import torch.nn.functional as F

from models import get_single_group_tensor, load_param_dict
from utils import cosine_similarity, ema_update, flatten_tensors, normalize_dict


def server_select_top_importance_conv_layer(
    model,
    params,
    loader,
    device,
    layer_groups,
    candidate_groups,
):
    """
    Select exactly one convolutional group on the server.

    Parameter importance is approximated with one server-side batch:
    importance(theta_j) = abs(theta_j * grad_j).
    The layer score is the mean parameter importance in that group, which
    avoids always favoring deeper layers only because they have more weights.
    """
    scores = {group: 0.0 for group in candidate_groups}
    if len(candidate_groups) == 0:
        return [], scores

    load_param_dict(model, params)
    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    try:
        x, y = next(iter(loader))
    except StopIteration:
        selected = candidate_groups[0]
        return [selected], scores

    x, y = x.to(device), y.to(device)
    loss = F.cross_entropy(model(x), y)
    loss.backward()

    named_params = dict(model.named_parameters())
    for group in candidate_groups:
        total_importance = 0.0
        total_params = 0
        for name in layer_groups.get(group, []):
            param = named_params.get(name)
            if param is None or param.grad is None:
                continue
            importance = torch.abs(param.detach() * param.grad.detach())
            total_importance += float(importance.sum().detach().cpu())
            total_params += importance.numel()
        scores[group] = total_importance / max(total_params, 1)

    model.zero_grad(set_to_none=True)
    selected = max(candidate_groups, key=lambda group: scores.get(group, 0.0))
    return [selected], scores


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
    鏈嶅姟鍣ㄥ€欓€夊眰閫夋嫨锛?
    1. 閲嶈鎬у拰涓€鑷存€у厛褰掍竴鍖栵紱
    2. score_l = importance_l * consistency_l锛?
    3. 闄堟棫搴﹁秴杩囬槇鍊肩殑灞備紭鍏堣繘鍏ュ€欓€夐泦鍚堬紱
    4. 鍓╀綑浣嶇疆鐢?score 鏈€楂樼殑灞傚～鍏呫€?
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
    鍒嗗眰鑱氬悎銆?
    姣忓眰鍙仛鍚堝疄闄呬笂浼犺灞傛洿鏂扮殑瀹㈡埛绔€?
    濡傛灉鏌愬眰鍙備笌瀹㈡埛绔繃灏戯紝鍙互闄嶄綆鑱氬悎姝ラ暱銆?
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
            if not torch.is_floating_point(new_global[name]):
                continue

            agg_update = torch.zeros_like(new_global[name])

            for i in available_clients:
                if name in client_updates[i]:
                    w = client_weights[i] / total_weight
                    agg_update += w * client_updates[i][name]

            new_global[name] = new_global[name] + layer_lr * agg_update

    return new_global, layer_client_count


def aggregate_topk_param_updates(global_params, client_uploads, client_weights):
    new_global = copy.deepcopy(global_params)
    sums = {}
    weights = {}

    for client_idx, upload in enumerate(client_uploads):
        for name, payload in upload.items():
            if name not in new_global or not torch.is_floating_point(new_global[name]):
                continue
            if name not in sums:
                sums[name] = torch.zeros_like(new_global[name]).reshape(-1)
                weights[name] = torch.zeros(
                    new_global[name].numel(),
                    dtype=torch.float32,
                )

            indices = payload["indices"].to(torch.long)
            updates = payload["updates"].to(sums[name].dtype)
            client_weight = float(client_weights[client_idx])
            sums[name][indices] += updates * client_weight
            weights[name][indices] += client_weight

    layer_client_count = {}
    for name, flat_sum in sums.items():
        weight = weights[name]
        updated = weight > 0
        if updated.any():
            flat_global = new_global[name].reshape(-1)
            flat_global[updated] = flat_global[updated] + flat_sum[updated] / weight[updated]
            new_global[name] = flat_global.reshape_as(new_global[name])

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
    鏇存柊鏈嶅姟鍣ㄧ淮鎶ょ殑灞傜骇缁熻閲忥細
    - 鍙傛暟閲嶈鎬э細瀹㈡埛绔笂浼犳爣閲忓悗 EMA锛?
    - 鏇存柊涓€鑷存€э細瀹㈡埛绔眰鏇存柊涓庢湰杞叏灞€灞傛洿鏂版柟鍚戠殑浣欏鸡鐩镐技搴︼紱
    """
    groups = list(layer_groups.keys())

    # 鏇存柊閲嶈鎬?
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

    # 鏇存柊涓€鑷存€?
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

            # 褰掍竴鍖栧埌 [0, 1]
            cos01 = (cos + 1.0) / 2.0

            # 骞呭害鍥犲瓙锛氶伩鍏嶆瀬灏忔洿鏂拌幏寰楄繃楂樹竴鑷存€?
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
