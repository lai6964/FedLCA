import torch
import torch.nn.functional as F

from models import (
    get_param_dict,
    load_param_dict,
    load_param_dict_excluding,
    set_trainable_layers,
)
from utils import normalize_dict

def compute_layer_importance(model, loader, device, layer_groups):
    """
    璁＄畻姣忓眰鍙傛暟閲嶈鎬э細
    I_l = mean(|theta_l * grad_l|)
    杩欓噷浣跨敤涓€涓?batch 杩戜技銆?
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
    model,
    loader,
    selected_layers,
    layer_groups,
    device,
    local_epochs=1,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    preserve_local_groups=None,
):
    """
    瀹㈡埛绔湰鍦拌缁冦€?
    鍙缁?selected_layers锛屽叾浣欏眰鍐荤粨銆?
    """
    preserve_local_groups = preserve_local_groups or []
    preserved_names = {
        name
        for group in preserve_local_groups
        for name in layer_groups.get(group, [])
    }
    if preserved_names:
        load_param_dict_excluding(model, global_params, preserved_names)
    else:
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

    # 浠呬笂浼犺閫変腑灞傜殑鏇存柊
    updates = {}
    for g in selected_layers:
        for name in layer_groups[g]:
            if name in new_params and torch.is_floating_point(new_params[name]):
                updates[name] = new_params[name] - global_params[name]

    avg_loss = total_loss / max(total_num, 1)
    return updates, avg_loss


def train_local_head(
    global_params,
    model,
    loader,
    layer_groups,
    device,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
):
    head_names = set(layer_groups.get("head", []))
    load_param_dict_excluding(model, global_params, head_names)
    set_trainable_layers(model, ["head"], layer_groups)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()


def build_topk_param_upload(
    global_params,
    model,
    loader,
    device,
    layer_groups,
    candidate_layers,
    topk_ratio=0.2,
    dense_layers=None,
):
    dense_layers = dense_layers or []
    all_param_names = {name for name, _ in model.named_parameters()}
    dense_names = {
        name
        for group in dense_layers
        for name in layer_groups.get(group, [])
    }
    for group in candidate_layers:
        for name in layer_groups.get(group, []):
            if name not in all_param_names and name in global_params:
                if torch.is_floating_point(global_params[name]):
                    dense_names.add(name)

    selected_names = {
        name
        for group in candidate_layers
        for name in layer_groups.get(group, [])
        if name not in dense_names
    }
    named_params = {
        name: param
        for name, param in model.named_parameters()
        if name in selected_names and param.requires_grad
    }

    new_params = get_param_dict(model)
    upload = {}
    dense_upload_params = 0
    for name in dense_names:
        if name not in new_params or name not in global_params:
            continue
        if not torch.is_floating_point(new_params[name]):
            continue
        upload[name] = {
            "full_update": new_params[name] - global_params[name],
        }
        dense_upload_params += new_params[name].numel()

    if not named_params:
        return upload, dense_upload_params

    model.train()
    model.zero_grad()
    try:
        x, y = next(iter(loader))
    except StopIteration:
        return upload, dense_upload_params

    x, y = x.to(device), y.to(device)
    loss = F.cross_entropy(model(x), y)
    loss.backward()

    importance_parts = []
    names = []
    for name, param in named_params.items():
        if param.grad is None:
            continue
        importance_parts.append(torch.abs(param.detach() * param.grad.detach()).reshape(-1).cpu())
        names.append(name)

    if not importance_parts:
        model.zero_grad()
        return upload, dense_upload_params

    flat_importance = torch.cat(importance_parts)
    total = flat_importance.numel()
    k = min(total, max(1, int(total * topk_ratio)))
    topk_indices = torch.topk(flat_importance, k=k, largest=True).indices

    offset = 0
    topk_indices = topk_indices.sort().values
    for name, scores in zip(names, importance_parts):
        param = named_params[name].detach().cpu().reshape(-1)
        base_param = global_params[name].reshape(-1)
        delta = param - base_param
        next_offset = offset + scores.numel()
        mask = (topk_indices >= offset) & (topk_indices < next_offset)
        local_indices = topk_indices[mask] - offset
        if local_indices.numel() > 0:
            upload[name] = {
                "indices": local_indices.to(torch.long),
                "updates": delta[local_indices].clone(),
                "shape": tuple(named_params[name].shape),
            }
        offset = next_offset

    model.zero_grad()
    return upload, dense_upload_params + k


def estimate_local_consistency(
    local_importance,
    server_consistency,
    candidate_layers,
):
    """
    绠€鍖栧疄鐜帮細
    瀹㈡埛绔娇鐢ㄦ湇鍔″櫒缁存姢鐨勫眰绾т竴鑷存€х粺璁′綔涓哄眬閮ㄤ竴鑷存€у厛楠屻€?
    濡傛灉鍚庣画鎯虫洿涓ユ牸锛屽彲浠ヨ瀹㈡埛绔敤鏈湴姊害鏂瑰悜涓庢渶杩戝叏灞€灞傛洿鏂版柟鍚戣绠椾綑寮︾浉浼煎害銆?
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
    瀹㈡埛绔嚜閫傚簲閫夋嫨灞傘€?
    閫夋嫨鍒嗘暟锛?
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

    # 闃叉瀹㈡埛绔竴灞傞兘涓嶈缁?
    if len(selected) < min_layers and len(items) > 0:
        selected = [items[0][0]]

    return selected, used_budget
