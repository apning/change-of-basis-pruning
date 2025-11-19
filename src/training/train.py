import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.modeling.vgg import COB_VGG
from src.training.utils import get_lr_scheduler
from src.utils import (
    get_project_root,
    save_as_json,
    select_best_device,
    create_run_identifier,
)
from src.datasets.datasets import get_dataset_by_name
from src.training.config import TrainingConfig


def train(config: TrainingConfig, model: COB_VGG, device: torch.device | None = None):
    """ """
    # Setup device
    device = device or select_best_device()
    print("Using device: ", device)
    # Move model to device
    model.to(device)
    # Create checkpoint dir
    checkpoint_dir = (
        get_project_root()
        / "checkpoints"
        / config.dataset_name
        / config.name
        / create_run_identifier()
    )
    if checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} already exists")
    checkpoint_dir.mkdir(parents=True)
    # Tensorboard logging
    writer = SummaryWriter(log_dir=checkpoint_dir / "runs")
    # Define criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Load datasets (uses shared cached directory)
    train_loader, test_loader = get_dataset_by_name(
        dataset_name=config.dataset_name,
        augment=config.augment,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        download=True,
    )

    # Setup optimizer
    optimizer_kwargs = config.optimizer_kwargs or {}
    if config.optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            **optimizer_kwargs,
        )
    elif config.optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            **optimizer_kwargs,
        )
    else:
        raise ValueError(f"Invalid optimizer name: {config.optimizer_name}")

    # Setup scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_name=config.lr_scheduler,
        warmup_steps=config.lr_warmup_steps,
        total_steps=config.num_train_steps,
    )

    # Tracking variables
    start_time = time.time()

    # Initial evaluation before training
    validation_loss, accuracy = evaluate(model, test_loader, device)
    writer.add_scalar("eval/loss", validation_loss, 0)
    writer.add_scalar("eval/accuracy", accuracy, 0)
    best_eval_accuracy = accuracy
    best_eval_step = 0
    model.save_pretrained(checkpoint_dir / "best_model")

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    step = 0

    while step < config.num_train_steps:
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)

        data = data.to(device)
        target = target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        if config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        scheduler.step()
        step += 1

        # Log training statistics
        if step % config.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        # Evaluate
        if step % config.eval_interval == 0 or step == config.num_train_steps:
            validation_loss, accuracy = evaluate(model, test_loader, device)
            writer.add_scalar("eval/loss", validation_loss, step)
            writer.add_scalar("eval/accuracy", accuracy, step)
            if accuracy > best_eval_accuracy:
                best_eval_accuracy = accuracy
                best_eval_step = step
                model.save_pretrained(checkpoint_dir / "best_model")
            model.train()

    writer.close()

    # Calculate total time
    total_time = time.time() - start_time

    # Save final statistics
    final_stats = {
        "name": config.name,
        "best_eval_accuracy": best_eval_accuracy,
        "best_eval_step": best_eval_step,
        "total_time_seconds": total_time,
    }

    config.save_as_json(checkpoint_dir / "training_config.json")
    save_as_json(final_stats, checkpoint_dir / "final_stats.json")

    return final_stats


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """ """
    original_training_mode = model.training
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        total_loss += F.cross_entropy(output, target, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += target.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    model.train(original_training_mode)
    return avg_loss, accuracy
