import torch
import torch.nn.functional as F

from models import load_param_dict

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
