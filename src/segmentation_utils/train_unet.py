import argparse
from os import write
from typing import Callable, Literal
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from datetime import datetime
from tqdm import tqdm

from segmentation.segmentation_utils.dataloader import ImageDataset
from torch.utils.tensorboard import SummaryWriter


# Global settings
device = "cuda" if torch.cuda.is_available() else "mps"
smooth = 1e-15
base_model_weight_dir = Path(
    "/home/tsakalis/ntua/phd/cellforge/cellforge/model_weights"
)
# Path(
#     '/Users/tsakalis/ntua/cellforge/cellforge/segmentation/model_weights')


def dice_coef(y_pred, y_true):
    intersection = torch.sum(y_true.flatten() * y_pred.flatten())
    return (2.0 * intersection + smooth) / (
        torch.sum(y_true).flatten() + torch.sum(y_pred).flatten() + smooth
    )


def dice_loss(y_pred, y_true):
    return 1.0 - dice_coef(y_pred, y_true)


def validate(model, val_dataloader, output_last_fn: Callable, loss_fn):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, total=len(val_dataloader))
        progress_bar.set_description("Validating...")
        for img_batch, gt_msk_batch in progress_bar:
            img_batch = img_batch.to(device)
            gt_msk_batch = gt_msk_batch.to(device)
            pred_mask = model(img_batch)
            loss = loss_fn(output_last_fn(pred_mask), gt_msk_batch.long())
            val_loss += loss.item()
    return val_loss / len(val_dataloader)


from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.cuda.amp import autocast, GradScaler


def train(
    train_dataset,
    val_dataset,
    task_name,
    lr,
    n_epochs,
    batch_size,
    model,
    loss_fn,
    output_last_fn=lambda x: x,
    precision="full",
    weights_path=None,
    warmup_epochs: int = 5,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2)
    model.to(device)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    model.train()

    # 1. optimizer with weight decay (no decay on biases / norm layers)
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in ["bias", "norm"])
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    total_steps = n_epochs * len(train_loader)

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
    total_steps   = n_epochs * len(train_loader)
    warmup_steps  = warmup_epochs * len(train_loader)

    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) /
                                    (total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if precision == "mixed":
        scaler = GradScaler()

    writer = SummaryWriter(f"runs/{task_name}_{datetime.now():%Y%m%d_%H%M%S}")

    global_step = 0
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0

        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}"):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()

            if precision == "full":
                preds = model(img)
                loss = loss_fn(output_last_fn(preds), mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            else:  # mixed
                with autocast():
                    preds = model(img)
                    loss = loss_fn(output_last_fn(preds), mask.long())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # one step every batch
            train_loss += loss.item()

            # log per‐batch metrics
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            global_step += 1
            scheduler.step()
        
        # end of epoch: validation
        val_loss = validate(model, val_loader, output_last_fn, loss_fn)

        avg_train = train_loss / len(train_loader)
        # writer.add_scalar("Loss/train_epoch", avg_train, epoch)
        # writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalars(
            "Loss",
            {
                "train_epoch": avg_train,
                "val": val_loss,
            },
            epoch,
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), base_model_weight_dir / f"{task_name}.pt")

        print(f"Epoch {epoch:02d}: train={avg_train:.4f}  val={val_loss:.4f}")

    writer.close()


class DiceLossModule(nn.Module):

    def forward(self, y_pred, y_true):
        return dice_loss(y_pred, y_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation training pipeline")
    parser.add_argument(
        "--segmentation_task",
        help="Segmentation task (full_embryo, inner_embryo, pronuclei)",
        type=str,
        required=True,
    )
    parser.add_argument("--lr", help="Learning rate", default=1e-4, type=float)
    parser.add_argument("--n_epochs", help="Number of epochs", default=15, type=int)
    parser.add_argument("--batch_size", help="Batch size", default=8, type=int)
    parser.add_argument(
        "--pretrained_weights",
        help="Path to pretrained weights",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    task_name = args.segmentation_task
    pretrained_weights = args.pretrained_weights

    data_pth = (
        Path("/home/tsakalis/ntua/phd/cellforge/cellforge/data/segmentation_data")
        / task_name
    )

    print("... Loading images ...")
    images = []
    masks = []
    match task_name:
        case "full_embryo":
            for embry_pth in (data_pth / "masks").glob("*"):
                for mask_pth in embry_pth.glob("*.png"):
                    msk_img = Image.open(mask_pth)
                    try:
                        raw_img = Image.open(
                            (data_pth / "images")
                            / f"{embry_pth.name.upper()}/{mask_pth.stem}.jpg"
                        )
                        images.append(raw_img)
                        masks.append(msk_img)
                    except Exception as e:
                        print(e)
                        continue
        case "inner_embryo" | "pronuclei":
            image_file_paths = sorted(
                list((data_pth / "images").glob("*.jpg")), key=lambda x: x.stem
            )
            mask_file_paths = sorted(
                list((data_pth / "masks").glob("*.png")), key=lambda x: x.stem
            )
            for img_path, msk_path in tqdm(
                zip(image_file_paths, mask_file_paths), total=len(image_file_paths)
            ):
                images.append(Image.open(img_path).copy())
                masks.append(Image.open(msk_path).copy())
        case _:
            raise ValueError("Unsupported segmentation task provided.")

    # Simple 80/20 train-validation split
    split_idx = int(len(images) * 0.8)
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_masks, val_masks = masks[:split_idx], masks[split_idx:]

    # Create dataset instances
    train_dataset = ImageDataset(train_images, train_masks, transform=True)
    val_dataset = ImageDataset(val_images, val_masks, transform=True)

    # Initialize model and loss function

    model = smp.Unet(
        encoder_name="resnext101_32x16d",
        encoder_weights="instagram",
        in_channels=3,
        classes=1,
    )

    from segmentation_models_pytorch.losses import DiceLoss

    loss_fn = DiceLoss(mode="binary", from_logits=False)

    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_name=task_name,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        output_last_fn=torch.sigmoid,
        model=model,
        loss_fn=loss_fn,
        weights_path=Path(pretrained_weights) if pretrained_weights else None,
    )
