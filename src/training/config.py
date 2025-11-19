from typing import Optional

from src.utils import JsonMixin

from dataclasses import dataclass


@dataclass
class TrainingConfig(JsonMixin):
    """
    Config class for training
    """

    # experiment metadata
    name: str = "default_experiment"
    notes: Optional[str] = None

    # model/data parameters and information
    dataset_name: str = "cifar-10"
    augment: bool | str = True
    batch_size: int = 128
    eval_batch_size: int = 1024
    num_workers: int = 0

    # training hyperparameters
    lr: float = 1e-3
    lr_warmup_steps: int = 1000
    lr_scheduler: str | None = "cosine"

    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    num_train_steps: int = 10000

    # Optimizer
    optimizer_name: str = "adamw"
    optimizer_kwargs: dict | None = None

    # evaluation & logging
    log_interval: int = 10
    eval_interval: int = 200
