"""Generalized dataset class for various segmentation tasks."""
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Literal
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from ..config import DatasetConfig, AugmentationConfig


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if len(image.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    elif len(image.shape) == 3:  # RGB
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge and convert back to RGB
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


class GeneralizedSegmentationDataset(Dataset):
    """A generalized dataset for various segmentation tasks."""
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        augmentation_config: AugmentationConfig,
        split: Literal["train", "val", "test"] = "train",
        transform_override: Optional[A.Compose] = None
    ):
        self.config = dataset_config
        self.aug_config = augmentation_config
        self.split = split
        self.use_augmentation = split == "train" and augmentation_config.use_augmentation
        
        # Load image and mask paths
        self.image_paths = self._load_image_paths()
        self.mask_paths = self._load_mask_paths()
        
        # Verify alignment
        assert len(self.image_paths) == len(list(self.mask_paths.values())[0]), \
            "Number of images and masks must match"
        
        # Setup transforms
        if transform_override is not None:
            self.transform = transform_override
        else:
            self.transform = self._build_transform()
            
        # Normalization is always applied
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def _load_image_paths(self) -> List[Path]:
        """Load all image file paths."""
        image_dir = self.config.data_dir / self.config.image_dir
        if not image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
            
        pattern = f"*.{self.config.image_format}"
        image_paths = sorted(list(image_dir.glob(pattern)))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
            
        return image_paths
    
    def _load_mask_paths(self) -> Dict[str, List[Path]]:
        """Load mask paths for each class."""
        mask_paths = {}
        
        for class_name, mask_dir_name in self.config.mask_dirs.items():
            mask_dir = self.config.data_dir / mask_dir_name
            if not mask_dir.exists():
                raise ValueError(f"Mask directory not found: {mask_dir}")
                
            pattern = f"*.{self.config.mask_format}"
            paths = sorted(list(mask_dir.glob(pattern)))
            
            if not paths:
                raise ValueError(f"No masks found in {mask_dir} with pattern {pattern}")
                
            mask_paths[class_name] = paths
            
        return mask_paths
    
    def _build_transform(self) -> A.Compose:
        """Build augmentation pipeline based on configuration."""
        transforms = []
        
        if self.use_augmentation:
            if self.aug_config.rotation_limit > 0:
                transforms.append(
                    A.Rotate(limit=self.aug_config.rotation_limit, p=0.5)
                )
            
            if self.aug_config.elastic_transform:
                transforms.append(
                    A.ElasticTransform(
                        p=0.3,
                        alpha=self.aug_config.elastic_alpha,
                        sigma=self.aug_config.elastic_sigma,
                        alpha_affine=50
                    )
                )
            
            if self.aug_config.horizontal_flip:
                transforms.append(A.HorizontalFlip(p=0.5))
                
            if self.aug_config.vertical_flip:
                transforms.append(A.VerticalFlip(p=0.5))
                
            if self.aug_config.brightness_contrast:
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5
                    )
                )
                
            if self.aug_config.gaussian_blur:
                transforms.append(
                    A.GaussianBlur(blur_limit=(3, 7), p=0.3)
                )
        
        # Always resize
        transforms.append(
            A.Resize(self.config.image_size, self.config.image_size)
        )
        
        return A.Compose(transforms)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image."""
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        
        if self.aug_config.use_clahe:
            image = apply_clahe(image)
            
        return image
    
    def _load_mask(self, idx: int) -> np.ndarray:
        """Load and combine masks based on problem type."""
        if self.config.problem_type == "binary":
            # Single class segmentation
            class_name = list(self.config.mask_dirs.keys())[0]
            mask_path = self.mask_paths[class_name][idx]
            mask = np.array(Image.open(mask_path).convert("L"))
            # Binary threshold
            mask = (mask > 128).astype(np.uint8)
            return mask
            
        elif self.config.problem_type == "multiclass":
            # Multi-class segmentation (exclusive classes)
            # First, load any mask to get original dimensions
            first_class = list(self.mask_paths.keys())[0]
            first_mask = np.array(Image.open(self.mask_paths[first_class][idx]).convert("L"))
            h, w = first_mask.shape
            
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Load each class mask and assign class indices
            for class_idx, (class_name, paths) in enumerate(self.mask_paths.items(), 1):
                class_mask = np.array(Image.open(paths[idx]).convert("L"))
                # Assign class index where mask is positive
                mask[class_mask > 128] = class_idx
                
            return mask
            
        elif self.config.problem_type == "multilabel":
            # Multi-label segmentation (non-exclusive classes)
            masks = []
            
            for class_name, paths in self.mask_paths.items():
                class_mask = np.array(Image.open(paths[idx]).convert("L"))
                # Binary threshold
                class_mask = (class_mask > 128).astype(np.float32)
                masks.append(class_mask)
                
            # Stack masks along channel dimension
            return np.stack(masks, axis=0)
        
        else:
            raise ValueError(f"Unknown problem type: {self.config.problem_type}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(idx)
        
        # Apply spatial transforms
        if self.transform is not None:
            if self.config.problem_type == "multilabel":
                # For multilabel, we need to handle multiple masks
                # Transpose mask to HWC for albumentations
                mask_hwc = mask.transpose(1, 2, 0)
                transformed = self.transform(image=image, mask=mask_hwc)
                image = transformed["image"]
                mask = transformed["mask"].transpose(2, 0, 1)  # Back to CHW
            else:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
        
        # Apply normalization and convert to tensor
        normalized = self.normalize(image=image)
        image_tensor = normalized["image"]
        
        # Convert mask to tensor
        if self.config.problem_type == "multiclass":
            mask_tensor = torch.from_numpy(mask).long()
        else:
            mask_tensor = torch.from_numpy(mask).float()
            
        return image_tensor, mask_tensor


def create_data_splits(
    dataset_config: DatasetConfig,
    augmentation_config: AugmentationConfig,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, GeneralizedSegmentationDataset]:
    """Create train/val/test splits from configuration."""
    np.random.seed(random_seed)
    
    # Create a temporary dataset to get total length
    temp_dataset = GeneralizedSegmentationDataset(
        dataset_config, augmentation_config, split="train"
    )
    total_samples = len(temp_dataset)
    
    # Calculate split indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    datasets = {}
    
    for split_name, split_indices in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices)
    ]:
        if len(split_indices) > 0:
            dataset = GeneralizedSegmentationDataset(
                dataset_config,
                augmentation_config,
                split=split_name
            )
            # Override with subset
            dataset.image_paths = [dataset.image_paths[i] for i in split_indices]
            for class_name in dataset.mask_paths:
                dataset.mask_paths[class_name] = [
                    dataset.mask_paths[class_name][i] for i in split_indices
                ]
            datasets[split_name] = dataset
            
    return datasets