from collections import defaultdict

import torch
import torch.nn as nn
import torchvision

from utils import flatten_tensors

class CifarCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        feature_size = 64 * (input_size // 4) * (input_size // 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        return self.fc2(x)


class FedAvgCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=32):
        super().__init__()
        feature_size = 64 * (((input_size - 4) // 2 - 4) // 2) ** 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        return self.fc2(x)


def get_input_size(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "tinyimagenet":
        return 64
    return 32


def build_model(num_classes=10, model_name="cnn", dataset_name="cifar10"):
    model_name = model_name.lower()
    input_size = get_input_size(dataset_name)
    if model_name == "cnn":
        return CifarCNN(num_classes=num_classes, input_size=input_size)

    if model_name == "fedavgcnn":
        return FedAvgCNN(num_classes=num_classes, input_size=input_size)

    if model_name != "resnet18":
        raise ValueError(f"Unsupported model: {model_name}")

    model = torchvision.models.resnet18(num_classes=num_classes)

    # 淇敼 ResNet-18 浠ラ€傚簲 CIFAR 杈撳叆
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def get_layer_groups(model):
    """
    灏嗘ā鍨嬪垝鍒嗕负灞傜骇妯″潡銆?
    瀵?ResNet-18:
    conv1/bn1, layer1, layer2, layer3, layer4, fc
    """
    groups = defaultdict(list)

    for name in model.state_dict().keys():
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


def load_param_dict_excluding(model, param_dict, excluded_names):
    local_state = model.state_dict()
    for name, value in param_dict.items():
        if name not in excluded_names:
            local_state[name] = value
    model.load_state_dict(local_state, strict=True)


def get_group_tensors(param_dict, group_names, layer_groups):
    tensors = []
    for g in group_names:
        for name in layer_groups[g]:
            if name in param_dict and torch.is_floating_point(param_dict[name]):
                tensors.append(param_dict[name])
    return tensors


def get_single_group_tensor(param_dict, group, layer_groups):
    tensors = []
    for name in layer_groups[group]:
        if name in param_dict and torch.is_floating_point(param_dict[name]):
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
    state = model.state_dict()

    for g, names in layer_groups.items():
        total = 0
        for name in names:
            if name in state and torch.is_floating_point(state[name]):
                total += state[name].numel()
        param_counts[g] = total

    return param_counts
