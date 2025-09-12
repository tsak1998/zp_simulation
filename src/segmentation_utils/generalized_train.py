"""Generalized training module for segmentation tasks."""
from typing import Optional, Dict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, JaccardLoss
from tqdm import tqdm
import math

from torch.utils.tensorboard import SummaryWriter

from ..config import SegmentationConfig
from .device_utils import (
    get_autocast_context,
    is_mps_device,
    supports_mixed_precision,
    print_device_info
)


def get_model(config: SegmentationConfig) -> nn.Module:
    """Create model based on configuration."""
    model_args = {
        "encoder_name": config.model.encoder_name,
        "encoder_weights": config.model.encoder_weights,
        "in_channels": config.model.in_channels,
        "classes": config.model.classes,
    }
    
    if config.model.activation:
        model_args["activation"] = config.model.activation
    
    # Get model architecture from smp
    if hasattr(smp, config.model.architecture):
        model_class = getattr(smp, config.model.architecture)
        return model_class(**model_args)
    else:
        raise ValueError(f"Unknown architecture: {config.model.architecture}")


def get_loss_function(config: SegmentationConfig) -> nn.Module:
    """Create loss function based on configuration."""
    loss_args = {
        "mode": config.loss.mode,
        "from_logits": config.loss.from_logits,
    }
    
    if config.loss.name == "DiceLoss":
        return DiceLoss(**loss_args)
    elif config.loss.name == "FocalLoss":
        return FocalLoss(**loss_args)
    elif config.loss.name == "JaccardLoss":
        return JaccardLoss(**loss_args)
    elif config.loss.name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(weight=config.loss.weights)
    elif config.loss.name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    else:
        # Composite loss
        if "+" in config.loss.name:
            losses = []
            weights = []
            for loss_component in config.loss.name.split("+"):
                loss_name, weight = loss_component.strip().split("*")
                weight = float(weight)
                temp_config = config
                temp_config.loss.name = loss_name
                losses.append(get_loss_function(temp_config))
                weights.append(weight)
            
            class CompositeLoss(nn.Module):
                def __init__(self, losses, weights):
                    super().__init__()
                    self.losses = nn.ModuleList(losses)
                    self.weights = weights
                
                def forward(self, pred, target):
                    total_loss = 0
                    for loss_fn, weight in zip(self.losses, self.weights):
                        total_loss += weight * loss_fn(pred, target)
                    return total_loss
            
            return CompositeLoss(losses, weights)
        else:
            raise ValueError(f"Unknown loss function: {config.loss.name}")


def get_optimizer(
    model: nn.Module,
    config: SegmentationConfig
) -> torch.optim.Optimizer:
    """Create optimizer with proper weight decay handling."""
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases and normalization layers
        if "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": config.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    return torch.optim.AdamW(
        param_groups,
        lr=config.training.learning_rate,
        betas=(0.9, 0.999)
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SegmentationConfig,
    steps_per_epoch: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    total_steps = config.training.num_epochs * steps_per_epoch
    warmup_steps = config.training.warmup_epochs * steps_per_epoch
    
    if config.training.scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = (
                float(step - warmup_steps) /
                float(max(1, total_steps - warmup_steps))
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif config.training.scheduler == "constant":
        return None
    
    else:
        raise ValueError(f"Unknown scheduler: {config.training.scheduler}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_metric: float,
    config: SegmentationConfig,
    is_best: bool = False
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "config": config
    }
    
    # Save latest checkpoint
    save_path = config.training.save_dir / f"{config.task_name}_latest.pth"
    torch.save(checkpoint, save_path)
    
    # Save best checkpoint
    if is_best:
        best_path = config.training.save_dir / f"{config.task_name}_best.pth"
        torch.save(checkpoint, best_path)
    
    # Save epoch checkpoint periodically
    if epoch % 10 == 0:
        epoch_path = (
            config.training.save_dir /
            f"{config.task_name}_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, epoch_path)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[GradScaler],
    config: SegmentationConfig,
    epoch: int,
    writer: SummaryWriter,
    global_step: int
) -> tuple[float, int]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    desc = f"Epoch {epoch}/{config.training.num_epochs}"
    progress_bar = tqdm(dataloader, desc=desc)
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(config.device)
        masks = masks.to(config.device)
        
        optimizer.zero_grad()
        
        use_amp = (config.training.precision == "mixed" and
                   supports_mixed_precision(config.device))
        
        if use_amp and scaler is not None:
            with get_autocast_context(config.device, enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        running_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log to tensorboard
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)
        writer.add_scalar("Learning_Rate", current_lr, global_step)
        global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{current_lr:.2e}"
        })
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, global_step


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: SegmentationConfig,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    
    # Additional metrics
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(config.device)
            masks = masks.to(config.device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            # Calculate additional metrics
            if config.dataset.problem_type == "multiclass":
                preds = torch.argmax(outputs, dim=1)
                # Calculate per-class dice
                for class_idx in range(1, config.model.classes):
                    pred_mask = (preds == class_idx).float()
                    true_mask = (masks == class_idx).float()
                    
                    intersection = (pred_mask * true_mask).sum()
                    union = pred_mask.sum() + true_mask.sum()
                    
                    if union > 0:
                        dice = 2 * intersection / union
                        iou = intersection / (union - intersection)
                        total_dice += dice.item()
                        total_iou += iou.item()
            else:
                # Binary or multilabel
                preds = torch.sigmoid(outputs) > 0.5
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                
                if union > 0:
                    dice = 2 * intersection / union
                    iou = intersection / (union - intersection)
                    total_dice += dice.item()
                    total_iou += iou.item()
            
            num_batches += 1
    
    metrics = {
        "loss": running_loss / len(dataloader),
        "dice": total_dice / (
            num_batches * (
                config.model.classes - 1
                if config.dataset.problem_type == "multiclass"
                else 1
            )
        ),
        "iou": total_iou / (
            num_batches * (
                config.model.classes - 1
                if config.dataset.problem_type == "multiclass"
                else 1
            )
        )
    }
    
    return metrics


def train(
    train_dataset,
    val_dataset,
    config: SegmentationConfig,
    resume_from: Optional[str] = None
) -> None:
    """Main training function."""
    # Create directories
    config.training.save_dir.mkdir(parents=True, exist_ok=True)
    config.training.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    # Create model, loss, optimizer
    model = get_model(config).to(config.device)
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision training
    use_amp = (config.training.precision == "mixed" and
               supports_mixed_precision(config.device))
    
    if use_amp and is_mps_device(config.device):
        # MPS doesn't use GradScaler
        scaler = None
        print("Note: MPS device detected. "
              "Mixed precision will use autocast only.")
    elif use_amp:
        scaler = GradScaler()
    else:
        scaler = None
        if config.training.precision == "mixed":
            print(f"Warning: Mixed precision requested but "
                  f"not supported on {config.device}")
    
    # Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = config.training.log_dir / f"{config.task_name}_{timestamp}"
    writer = SummaryWriter(log_path)
    
    # Resume from checkpoint
    start_epoch = 1
    best_metric = float('inf')
    global_step = 0
    
    if resume_from:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric = checkpoint["best_metric"]
        global_step = start_epoch * len(train_loader)
        print(f"Resumed from epoch {start_epoch}, "
              f"best metric: {best_metric:.4f}")
    
    # Print device info before training
    print_device_info()
    
    # Training loop
    for epoch in range(start_epoch, config.training.num_epochs + 1):
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, config, epoch, writer, global_step
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, config, epoch
        )
        
        # Log metrics
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Metrics/dice", val_metrics["dice"], epoch)
        writer.add_scalar("Metrics/iou", val_metrics["iou"], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics["loss"] < best_metric
        if is_best:
            best_metric = val_metrics["loss"]
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_metric,
            config, is_best
        )
    
    writer.close()
    print(f"\nTraining completed! Best validation loss: {best_metric:.4f}")