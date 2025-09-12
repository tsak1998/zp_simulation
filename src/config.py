"""Configuration system for segmentation tasks."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Literal
from pathlib import Path

from segmentation_utils.device_utils import get_best_device


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    name: str
    data_dir: Path
    image_dir: str = "Images"
    mask_dirs: Dict[str, str] = field(default_factory=dict)
    image_format: str = "bmp"
    mask_format: str = "bmp"
    image_size: int = 224
    num_classes: int = 1
    class_names: List[str] = field(default_factory=list)
    problem_type: Literal["binary", "multiclass", "multilabel"] = "binary"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    architecture: str = "Unet"
    encoder_name: str = "resnet50"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 1
    activation: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    precision: Literal["full", "mixed"] = "mixed"
    num_workers: int = 4
    save_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)
        self.log_dir = Path(self.log_dir)


@dataclass
class LossConfig:
    """Configuration for loss functions."""

    name: str = "DiceLoss"
    mode: str = "binary"
    from_logits: bool = True
    smooth: float = 1e-5
    weights: Optional[List[float]] = None


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    use_augmentation: bool = True
    rotation_limit: int = 90
    elastic_transform: bool = True
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    horizontal_flip: bool = True
    vertical_flip: bool = True
    brightness_contrast: bool = True
    gaussian_blur: bool = True
    use_clahe: bool = True


@dataclass
class SegmentationConfig:
    """Main configuration combining all components."""

    task_name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    augmentation: AugmentationConfig
    device: str = field(default_factory=get_best_device)
    seed: int = 42


# Predefined configurations for different tasks
PRONUCLEI_CONFIG = SegmentationConfig(
    task_name="pronuclei_segmentation",
    dataset=DatasetConfig(
        name="pronuclei",
        data_dir=Path("/home/tsakalis/ntua/phd/cellforge/" "cellforge/data/segmentation_data"),
        mask_dirs={"pronuclei": "masks"},
        num_classes=2,
        class_names=["background", "pronuclei"],
        problem_type="multilabel",
    ),
    model=ModelConfig(
        architecture="Unet",
        encoder_name="resnext101_32x16d",
        encoder_weights="instagram",
        classes=2,
    ),
    training=TrainingConfig(batch_size=32, num_epochs=120, learning_rate=1e-4),
    loss=LossConfig(name="DiceLoss", mode="multilabel"),
    augmentation=AugmentationConfig(),
)


BLASTOCYST_CONFIG = SegmentationConfig(
    task_name="blastocyst_segmentation",
    dataset=DatasetConfig(
        name="blastocyst",
        data_dir=Path("./data/BLASTOCYST"),
        image_dir="Images",
        mask_dirs={"ZP": "GT_ZP", "ICM": "GT_ICM", "TE": "GT_TE"},
        num_classes=4,  # background + 3 structures
        class_names=["background", "zona_pellucida", "inner_cell_mass", "trophectoderm"],
        problem_type="multiclass",
    ),
    model=ModelConfig(
        architecture="Unet",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        classes=4,
        activation=None,  # For multiclass with CE loss
    ),
    training=TrainingConfig(
        batch_size=16, num_epochs=150, learning_rate=5e-4, scheduler="cosine", warmup_epochs=5
    ),
    loss=LossConfig(name="DiceLoss", mode="multiclass", from_logits=True),
    augmentation=AugmentationConfig(
        rotation_limit=180,  # Blastocysts can be in any orientation
        elastic_transform=True,
        use_clahe=True,
    ),
)
