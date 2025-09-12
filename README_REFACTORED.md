# Refactored Segmentation Pipeline

This repository contains a generalized segmentation pipeline that has been refactored from the original pronuclei detection code to support multiple segmentation tasks, including blastocyst segmentation.

## Features

- **Configuration-based approach**: All training parameters are managed through configuration objects
- **Multi-task support**: Easily switch between binary, multiclass, and multilabel segmentation
- **Flexible data loading**: Generalized dataset class that works with various folder structures
- **Modern training features**: Mixed precision training, cosine annealing, warmup, tensorboard logging
- **Multiple architectures**: Support for various segmentation architectures from segmentation_models_pytorch

## Project Structure

```
src/
├── config.py                    # Configuration dataclasses
├── train_blastocyst.py         # Main training script for blastocyst
├── inference_blastocyst.py     # Inference script
├── segmentation_utils/
│   ├── generalized_dataset.py  # Generalized dataset class
│   └── generalized_train.py    # Training utilities
```

## Blastocyst Segmentation

The pipeline is configured to detect three structures in blastocyst images:
- **Zona Pellucida (ZP)**: The outer protective layer
- **Inner Cell Mass (ICM)**: Cells that will form the embryo
- **Trophectoderm (TE)**: Cells that will form the placenta

### Data Structure

Your blastocyst data should be organized as follows:

```
data/BLASTOCYST/
├── Images/         # Input images (jpg format)
├── GT_ZP/          # Zona pellucida masks (png format)
├── GT_ICM/         # Inner cell mass masks (png format)
└── GT_TE/          # Trophectoderm masks (png format)
```

### Training

Basic training command:

```bash
python src/train_blastocyst.py --data_dir ./data/BLASTOCYST
```

Advanced training with custom parameters:

```bash
python src/train_blastocyst.py \
    --data_dir ./data/BLASTOCYST \
    --epochs 200 \
    --batch_size 16 \
    --lr 5e-4 \
    --encoder efficientnet-b4 \
    --architecture Unet \
    --problem_type multiclass \
    --image_size 256 \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

#### Key Parameters:

- `--problem_type`: Choose between `multiclass` (exclusive classes) or `multilabel` (overlapping classes)
- `--encoder`: Backbone encoder (e.g., resnet50, efficientnet-b4, resnext101_32x8d)
- `--architecture`: Model architecture (Unet, UnetPlusPlus, DeepLabV3Plus, etc.)
- `--loss`: Loss function (auto, DiceLoss, FocalLoss, or combinations like "DiceLoss*0.5+FocalLoss*0.5")
- `--precision`: Training precision (`full` or `mixed`)
- `--no_augmentation`: Disable data augmentation

### Inference

Run inference on new images:

```bash
python src/inference_blastocyst.py \
    ./checkpoints/blastocyst_best.pth \
    ./test_images \
    --output ./predictions \
    --save_overlay
```

This will generate:
- Individual mask predictions for each class
- Overlay visualizations (if `--save_overlay` is used)

## Adding New Segmentation Tasks

To add a new segmentation task:

1. **Create a configuration** in `src/config.py`:

```python
NEW_TASK_CONFIG = SegmentationConfig(
    task_name="new_task",
    dataset=DatasetConfig(
        name="new_dataset",
        data_dir=Path("./data/NEW_TASK"),
        image_dir="images",
        mask_dirs={"class1": "masks_class1", "class2": "masks_class2"},
        num_classes=3,  # background + 2 classes
        class_names=["background", "class1", "class2"],
        problem_type="multiclass"
    ),
    model=ModelConfig(
        architecture="Unet",
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=3
    ),
    # ... other configurations
)
```

2. **Create a training script** (optional, or modify train_blastocyst.py):

```python
# Copy train_blastocyst.py and modify the configuration creation
```

## Key Improvements from Original Code

1. **Removed hardcoded paths**: All paths are now configurable
2. **Generalized dataset**: No longer specific to pronuclei/circles
3. **Configuration system**: Easy to experiment with different settings
4. **Better augmentation**: More augmentation options with Albumentations
5. **Improved training**: Warmup, better logging, checkpoint management
6. **Multi-task ready**: Easy to switch between different segmentation tasks

## Requirements

```
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python
pillow
tqdm
tensorboard
numpy
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./logs
```

## Tips for Best Results

1. **Data Quality**: Ensure masks are properly aligned with images
2. **Class Balance**: For multiclass problems, check if classes are balanced
3. **Augmentation**: Enable augmentation for better generalization
4. **Learning Rate**: Start with 5e-4 and adjust based on validation loss
5. **Model Selection**: EfficientNet-B4 or ResNeXt backbones often work well
6. **Mixed Precision**: Use mixed precision training for faster training

## Troubleshooting

1. **Out of Memory**: Reduce batch size or image size
2. **Poor Performance**: Check data quality, try different encoders
3. **Slow Training**: Enable mixed precision, reduce image size
4. **Overfitting**: Add more augmentation, reduce model size