import os

import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset


def get_cifar_transforms(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, test_transform


def build_cifar_dataset(dataset_name, data_root="./data", download=False, transform=None):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        dataset_cls = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset_cls = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_set = dataset_cls(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )
    test_set = dataset_cls(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )
    return ConcatDataset([train_set, test_set]), num_classes


def get_targets(dataset):
    targets = []
    for subset in dataset.datasets:
        targets.extend(subset.targets)
    return np.array(targets)


def load_dataset(dataset_name, data_root="./data", download=False):
    train_transform, test_transform = get_cifar_transforms(dataset_name)
    train_dataset, num_classes = build_cifar_dataset(
        dataset_name,
        data_root=data_root,
        download=download,
        transform=train_transform,
    )
    test_dataset, _ = build_cifar_dataset(
        dataset_name,
        data_root=data_root,
        download=False,
        transform=test_transform,
    )
    return train_dataset, test_dataset, num_classes


def dirichlet_partition(targets, num_clients, alpha, num_classes):
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        splits = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, splits)

        for cid, idx in enumerate(class_splits):
            client_indices[cid].extend(idx.tolist())

    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

    return client_indices


def split_client_indices(client_indices, train_ratio=0.75):
    train_indices = []
    test_indices = []
    for indices in client_indices:
        indices = list(indices)
        np.random.shuffle(indices)
        train_size = int(len(indices) * train_ratio)
        train_indices.append(indices[:train_size])
        test_indices.append(indices[train_size:])
    return train_indices, test_indices


def pfl_partition(dataset, num_clients, alpha, num_classes, train_ratio=0.75):
    targets = get_targets(dataset)
    client_indices = dirichlet_partition(
        targets=targets,
        num_clients=num_clients,
        alpha=alpha,
        num_classes=num_classes,
    )
    return split_client_indices(client_indices, train_ratio=train_ratio)


def partition_file_name(dataset_name, num_clients, alpha, train_ratio, seed):
    alpha_tag = str(alpha).replace(".", "p")
    ratio_tag = str(train_ratio).replace(".", "p")
    return (
        f"{dataset_name}_clients{num_clients}_alpha{alpha_tag}_"
        f"ratio{ratio_tag}_seed{seed}.npz"
    )


def save_partition(path, client_train_indices, client_test_indices):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for cid, indices in enumerate(client_train_indices):
        payload[f"train_{cid}"] = np.array(indices, dtype=np.int64)
    for cid, indices in enumerate(client_test_indices):
        payload[f"test_{cid}"] = np.array(indices, dtype=np.int64)
    payload["num_clients"] = np.array([len(client_train_indices)], dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_partition(path):
    payload = np.load(path, allow_pickle=False)
    num_clients = int(payload["num_clients"][0])
    client_train_indices = [
        payload[f"train_{cid}"].astype(np.int64).tolist()
        for cid in range(num_clients)
    ]
    client_test_indices = [
        payload[f"test_{cid}"].astype(np.int64).tolist()
        for cid in range(num_clients)
    ]
    return client_train_indices, client_test_indices


def load_or_create_pfl_partition(
    dataset,
    dataset_name,
    num_clients,
    alpha,
    num_classes,
    train_ratio=0.75,
    seed=0,
    partition_dir="./data/partitions",
    regenerate=False,
):
    file_name = partition_file_name(
        dataset_name=dataset_name,
        num_clients=num_clients,
        alpha=alpha,
        train_ratio=train_ratio,
        seed=seed,
    )
    partition_path = os.path.join(partition_dir, file_name)

    if os.path.exists(partition_path) and not regenerate:
        print(f"Loaded client partition from: {partition_path}")
        return load_partition(partition_path)

    client_train_indices, client_test_indices = pfl_partition(
        dataset=dataset,
        num_clients=num_clients,
        alpha=alpha,
        num_classes=num_classes,
        train_ratio=train_ratio,
    )
    save_partition(partition_path, client_train_indices, client_test_indices)
    print(f"Saved client partition to: {partition_path}")
    return client_train_indices, client_test_indices


def build_client_loaders(dataset, client_indices, batch_size, num_workers=2, shuffle=True):
    loaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders
