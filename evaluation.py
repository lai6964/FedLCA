import torch
import torch.nn.functional as F

from models import load_param_dict

def evaluate(model, params, test_loader, device, load_params=True):
    model.to(device)
    if load_params:
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


def evaluate_personalized(client_models, global_params, test_loaders, device, client_weights):
    total_correct = 0
    total_num = 0
    total_loss = 0.0

    for cid, model in enumerate(client_models):
        acc, loss = evaluate(
            model,
            global_params,
            test_loaders[cid],
            device,
            load_params=False,
        )
        num_samples = client_weights[cid]
        total_correct += acc * num_samples
        total_loss += loss * num_samples
        total_num += num_samples

    return total_correct / max(total_num, 1), total_loss / max(total_num, 1)
