from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def get_lr_scheduler(
    optimizer: Optimizer,
    scheduler_name: str | None,
    warmup_steps: int,
    total_steps: int,
) -> LRScheduler:
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min((step + 1) / (warmup_steps or 1), 1.0)
    )

    steps_after_warmup = total_steps - warmup_steps
    if steps_after_warmup < 0:
        raise ValueError(
            f"Total steps must be greater than warmup steps. Total steps: {total_steps}, Warmup steps: {warmup_steps}"
        )

    if scheduler_name is None:
        main_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0
        )
    elif scheduler_name == "cosine":
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps_after_warmup
        )
    elif scheduler_name == "linear":
        main_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=steps_after_warmup
        )
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    return scheduler
