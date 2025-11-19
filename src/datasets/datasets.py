import torch
import torchvision
from torchvision import transforms

from src.utils import get_project_root

# Constant for shared dataset directory
DATASETS_DIR = get_project_root() / "data"


def get_dataset_by_name(
    dataset_name: str,
    augment: bool | str = False,
    batch_size: int = 64,
    eval_batch_size: int = 512,
    num_workers: int = 0,
    download: bool = True,
):
    """
    Get train and test data loaders for a specified dataset.

    Datasets are cached in a shared project directory to avoid re-downloading.

    Args:
        dataset_name: Name of the dataset ('mnist' or 'cifar-10')
        augment (bool | str): Whether to use augmentation for trainset. If "True", a default augmentation setup will be used. Certain strings can specify other augmentation options
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading

    Returns:
        tuple: (train_loader, test_loader)

    Raises:
        ValueError: If dataset_name is not supported
    """
    # Use shared datasets directory
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = DATASETS_DIR

    if dataset_name == "mnist":
        if augment:
            raise NotImplementedError("Augmentation not implemented for MNIST")

        try:
            train_dataset = torchvision.datasets.MNIST(
                data_dir,
                train=True,
                download=download,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )

            test_dataset = torchvision.datasets.MNIST(
                data_dir,
                train=False,
                download=download,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"MNIST not found in '{data_dir}'. Please run prefetch_datasets() before spawning workers. Original error: {e}"
            )

    elif dataset_name == "cifar-10":
        base_augment = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),
        ]

        if not augment:
            train_augment = []
        elif augment == True:  # noqa: E712
            train_augment = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            raise ValueError(f"Invalid augmentation type: {augment}")

        train_augment.extend(base_augment)

        try:
            train_dataset = torchvision.datasets.CIFAR10(
                data_dir,
                train=True,
                download=download,
                transform=torchvision.transforms.Compose(train_augment),
            )

            test_dataset = torchvision.datasets.CIFAR10(
                data_dir,
                train=False,
                download=download,
                transform=torchvision.transforms.Compose(base_augment),
            )
        except Exception as e:
            raise RuntimeError(
                f"CIFAR-10 not found in '{data_dir}'. Please run prefetch_datasets() before spawning workers. Original error: {e}"
            )

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Supported datasets: 'mnist', 'cifar-10'"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
