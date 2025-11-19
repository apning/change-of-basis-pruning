import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models

from src.datasets.datasets import get_dataset_by_name
from src.training.train import train, evaluate
from src.training.config import TrainingConfig
from src.utils import (
    save_as_json,
    select_best_device,
    get_project_root,
    str_formatted_datetime,
)


def apply_pruning(model: nn.Module, amount: float = 0.3, modules: str = "conv+linear"):
    """
    Here we're applying standard PyTorch global unstructured pruning to the user specified modules

    Args:
        model: nn.Module - model to prune
        amount: float - fraction of total weights to prune
        modules: str - which types of modules to prune ('conv', 'linear', or 'conv+linear')

    Returns:
        the in place pruned model
    """
    parameters_to_prune = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and "conv" in modules:
            parameters_to_prune.append((module, "weight"))
        elif isinstance(module, nn.Linear) and "linear" in modules:
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        raise ValueError(f"No matching layers found for pruning modules='{modules}'")

    print(
        f"[INFO: ] Applying global L1 unstructured pruning to {len(parameters_to_prune)} layers (amount={amount})..."
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Log sparsity layer-wise
    total_params, total_zeros = 0, 0
    for module, _ in parameters_to_prune:
        mask = module.weight_mask
        zeros = torch.sum(mask == 0).item()
        total = mask.numel()
        layer_sparsity = 100.0 * zeros / total
        total_params += total
        total_zeros += zeros
        print(f"  → Layer {module.__class__.__name__}: {layer_sparsity:.2f}% zeros")

    total_sparsity = 100.0 * total_zeros / total_params
    print(f"[INFO] Global sparsity after pruning: {total_sparsity:.2f}%")

    return model


def load_model(
    model_name: str = "vgg16", num_classes: int = 10, pretrained: bool = False
):
    """
    Load a torchvision model for pruning baseline.
    Automatically adjusts classifier head assuming CIFAR-10 (10 classes).
    """
    if model_name.lower() == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.lower() == "vgg19":
        model = models.vgg19(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def run_standard_pruning(
    config: TrainingConfig,
    model_name: str = "vgg16",
    dataset_name: str = "cifar-10",
    prune_amount: float = 0.3,
    prune_modules: str = "conv+linear",
    pretrained: bool = False,
    fine_tune_epochs: int = 1,
):
    """
    this the full pruning + fine-tuning pipeline.
    """

    device = select_best_device()
    project_root = get_project_root()
    checkpoint_dir = (
        project_root / f"pruning_runs/{model_name}_{str_formatted_datetime()}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name} model...")
    model = load_model(model_name, num_classes=10, pretrained=pretrained)
    model.to(device)

    print(f"Loading dataset: {dataset_name}")
    train_loader, test_loader = get_dataset_by_name(
        dataset_name=dataset_name,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        download=True,
    )

    print("[INFO] Evaluating baseline model before pruning...")
    baseline_loss, baseline_acc = evaluate(model, test_loader, device, step=0)
    print(f"[Baseline] Accuracy: {baseline_acc:.2f}%")

    model = apply_pruning(model, amount=prune_amount, modules=prune_modules)

    print(f"[INFO] Fine-tuning pruned model for {fine_tune_epochs} epochs...")
    # Temporarily override number of steps for shorter fine-tune
    fine_tune_config = TrainingConfig(
        **{**config.__dict__, "num_train_steps": fine_tune_epochs * 1000}
    )
    final_stats = train(fine_tune_config, model, device)

    pruned_loss, pruned_acc = evaluate(
        model, test_loader, device, step=fine_tune_config.num_train_steps
    )
    print(f"[After Pruning] Accuracy: {pruned_acc:.2f}%")

    torch.save(model.state_dict(), checkpoint_dir / "pruned_model.pt")

    summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "method": "global_L1_unstructured",
        "amount": prune_amount,
        "baseline_accuracy": baseline_acc,
        "pruned_accuracy": pruned_acc,
        "checkpoint_dir": str(checkpoint_dir),
    }

    save_as_json(summary, checkpoint_dir / "pruning_summary.json")
    print(f"[INFO] Pruning run complete. Summary saved to {checkpoint_dir}")

    return model, summary


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run standard pruning baseline using VGG16.")
    parser.add_argument(
        "--model", type=str, default="vgg16", help="torchvision model name"
    )
    parser.add_argument("--dataset", type=str, default="cifar-10", help="dataset name")
    parser.add_argument(
        "--amount", type=float, default=0.3, help="fraction of weights to prune"
    )
    parser.add_argument(
        "--modules", type=str, default="conv+linear", help="module types to prune"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="use pretrained weights"
    )
    parser.add_argument("--fine_tune_epochs", type=int, default=1)
    args = parser.parse_args()

    # Default training config (can override if needed)
    config = TrainingConfig(name=f"prune_{args.model}")

    run_standard_pruning(
        config=config,
        model_name=args.model,
        dataset_name=args.dataset,
        prune_amount=args.amount,
        prune_modules=args.modules,
        pretrained=args.pretrained,
        fine_tune_epochs=args.fine_tune_epochs,
    )
