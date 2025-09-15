"""Training script for blastocyst segmentation (ZP, ICM, TE)."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Local imports after path setup
from src.config import (  # noqa: E402
    SegmentationConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
    AugmentationConfig,
)
from src.segmentation_utils.generalized_dataset import create_data_splits  # noqa: E402
from src.segmentation_utils.generalized_train import train  # noqa: E402
from src.segmentation_utils.device_utils import get_best_device, print_device_info  # noqa: E402


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_blastocyst_config(args) -> SegmentationConfig:
    """Create configuration for blastocyst segmentation."""
    # Dataset configuration
    dataset_config = DatasetConfig(
        name="blastocyst",
        data_dir=Path(args.data_dir),
        image_dir="Images",
        mask_dirs={"ZP": "GT_ZP", "ICM": "GT_ICM", "TE": "GT_TE"},
        image_format="BMP",
        mask_format="bmp",
        image_size=args.image_size,
        num_classes=4,  # background + 3 structures
        class_names=["background", "zona_pellucida", "inner_cell_mass", "trophectoderm"],
        problem_type=args.problem_type,
    )

    # Model configuration
    model_config = ModelConfig(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=4 if args.problem_type == "multiclass" else 3,
        activation=None if args.problem_type == "multiclass" else "sigmoid",
    )

    # Training configuration
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler="cosine",
        warmup_epochs=5,
        precision=args.precision,
        num_workers=args.num_workers,
        save_dir=Path(args.save_dir),
        log_dir=Path(args.log_dir),
    )

    # Loss configuration
    if args.problem_type == "multiclass":
        loss_name = args.loss if args.loss != "auto" else "DiceLoss"
        loss_mode = "multiclass"
    else:
        loss_name = args.loss if args.loss != "auto" else "DiceLoss"
        loss_mode = "multilabel"

    loss_config = LossConfig(name=loss_name, mode=loss_mode, from_logits=True)

    # Augmentation configuration
    augmentation_config = AugmentationConfig(
        use_augmentation=not args.no_augmentation,
        rotation_limit=180,
        elastic_transform=True,
        elastic_alpha=1.0,
        elastic_sigma=50.0,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_contrast=True,
        gaussian_blur=True,
        use_clahe=True,
    )

    # Create main configuration
    config = SegmentationConfig(
        task_name=f"blastocyst_{args.problem_type}_{args.encoder}",
        dataset=dataset_config,
        model=model_config,
        training=training_config,
        loss=loss_config,
        augmentation=augmentation_config,
        device=get_best_device(),
        seed=args.seed,
    )

    return config


def main():
    parser = argparse.ArgumentParser(description="Train blastocyst segmentation model")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/BLASTOCYST",
        help="Path to blastocyst data directory",
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="multiclass",
        choices=["multiclass", "multilabel"],
        help="Problem type: multiclass or multilabel",
    )

    # Model arguments
    parser.add_argument("--architecture", type=str, default="Unet", help="Model architecture")
    parser.add_argument("--encoder", type=str, default="efficientnet-b4", help="Encoder backbone")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--loss", type=str, default="auto", help="Loss function (auto, DiceLoss, FocalLoss, etc.)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="mixed",
        choices=["full", "mixed"],
        help="Training precision",
    )

    # Other arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Data split arguments
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test data ratio")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create configuration
    config = create_blastocyst_config(args)

    print("\n" + "=" * 50)
    print("BLASTOCYST SEGMENTATION TRAINING")
    print("=" * 50)
    print(f"Task: {config.task_name}")
    print(f"Problem type: {config.dataset.problem_type}")
    print(f"Classes: {config.dataset.class_names}")
    print(f"Model: {config.model.architecture} - {config.model.encoder_name}")
    print(f"Image size: {config.dataset.image_size}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Device: {config.device}")
    print("=" * 50 + "\n")

    # Print device information
    print_device_info()

    # Create datasets
    print("Creating datasets...")
    try:
        datasets = create_data_splits(
            config.dataset,
            config.augmentation,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=config.seed,
        )

        print(f"Train samples: {len(datasets['train'])}")
        print(f"Validation samples: {len(datasets['val'])}")
        if "test" in datasets:
            print(f"Test samples: {len(datasets['test'])}")
        print()

    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("\nEnsure your data directory has this structure:")
        print(f"{args.data_dir}/")
        print("├── Images/         # Input images")
        print("├── GT_ZP/          # Zona pellucida masks")
        print("├── GT_ICM/         # Inner cell mass masks")
        print("└── GT_TE/          # Trophectoderm masks")
        return

    # Start training
    print("Starting training...")
    train(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        config=config,
        resume_from=args.resume,
    )

    print("\nTraining completed!")
    print(f"Checkpoints saved to: {config.training.save_dir}")
    print(f"Logs saved to: {config.training.log_dir}")

    # Save test dataset indices for later evaluation
    if "test" in datasets:
        test_indices_file = config.training.save_dir / f"{config.task_name}_test_indices.txt"
        with open(test_indices_file, "w") as f:
            for idx, path in enumerate(datasets["test"].image_paths):
                f.write(f"{idx}\t{path}\n")
        print(f"Test indices saved to: {test_indices_file}")


if __name__ == "__main__":
    main()
