import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def load_dataset(dataset_name, data_root="./data", download=False):
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
            root=data_root, train=True, download=download, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=download, transform=test_transform
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
            root=data_root, train=True, download=download, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=download, transform=test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_set, test_set, num_classes


def dirichlet_partition(dataset, num_clients, alpha, num_classes):
    """
    鎸夋爣绛句娇鐢?Dirichlet 鍒嗗竷鏋勯€?Non-IID 瀹㈡埛绔暟鎹€?
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
