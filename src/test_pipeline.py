"""Test script to verify the refactored pipeline components."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import shutil

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import SegmentationConfig, DatasetConfig, ModelConfig, TrainingConfig, LossConfig, AugmentationConfig
from src.segmentation_utils.generalized_dataset import GeneralizedSegmentationDataset, create_data_splits


def create_synthetic_data(data_dir: Path, num_samples: int = 10):
    """Create synthetic data for testing."""
    print("Creating synthetic test data...")
    
    # Create directories
    (data_dir / "Images").mkdir(parents=True)
    (data_dir / "GT_ZP").mkdir(parents=True)
    (data_dir / "GT_ICM").mkdir(parents=True)
    (data_dir / "GT_TE").mkdir(parents=True)
    
    # Create synthetic images and masks
    for i in range(num_samples):
        # Create a synthetic image (random grayscale converted to RGB)
        img_array = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
        img = Image.fromarray(img_array)
        img.save(data_dir / "Images" / f"sample_{i:03d}.jpg")
        
        # Create synthetic masks
        # Zona Pellucida - outer ring
        mask_zp = np.zeros((256, 256), dtype=np.uint8)
        center = (128, 128)
        radius_outer = 100
        radius_inner = 80
        y, x = np.ogrid[:256, :256]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask_zp[(dist_from_center >= radius_inner) & (dist_from_center <= radius_outer)] = 255
        Image.fromarray(mask_zp).save(data_dir / "GT_ZP" / f"sample_{i:03d}.png")
        
        # Inner Cell Mass - small region inside
        mask_icm = np.zeros((256, 256), dtype=np.uint8)
        icm_center = (128 + np.random.randint(-30, 30), 128 + np.random.randint(-30, 30))
        icm_radius = 20
        dist_from_icm = np.sqrt((x - icm_center[0])**2 + (y - icm_center[1])**2)
        mask_icm[dist_from_icm <= icm_radius] = 255
        Image.fromarray(mask_icm).save(data_dir / "GT_ICM" / f"sample_{i:03d}.png")
        
        # Trophectoderm - rest of the inner area
        mask_te = np.zeros((256, 256), dtype=np.uint8)
        mask_te[(dist_from_center < radius_inner) & (dist_from_icm > icm_radius)] = 255
        Image.fromarray(mask_te).save(data_dir / "GT_TE" / f"sample_{i:03d}.png")
    
    print(f"Created {num_samples} synthetic samples")


def test_dataset_loading():
    """Test the generalized dataset loading."""
    print("\n" + "="*50)
    print("Testing Dataset Loading")
    print("="*50)
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "test_data"
        
        # Create synthetic data
        create_synthetic_data(data_dir, num_samples=5)
        
        # Test multiclass configuration
        print("\nTesting multiclass dataset...")
        dataset_config = DatasetConfig(
            name="test_multiclass",
            data_dir=data_dir,
            image_dir="Images",
            mask_dirs={
                "ZP": "GT_ZP",
                "ICM": "GT_ICM",
                "TE": "GT_TE"
            },
            image_format="jpg",
            mask_format="png",
            image_size=128,
            num_classes=4,
            class_names=["background", "zona_pellucida", "inner_cell_mass", "trophectoderm"],
            problem_type="multiclass"
        )
        
        aug_config = AugmentationConfig(use_augmentation=False)
        
        try:
            dataset = GeneralizedSegmentationDataset(dataset_config, aug_config, split="train")
            print(f"✓ Dataset created successfully")
            print(f"  - Number of samples: {len(dataset)}")
            
            # Test loading a sample
            image, mask = dataset[0]
            print(f"  - Image shape: {image.shape}")
            print(f"  - Mask shape: {mask.shape}")
            print(f"  - Unique mask values: {torch.unique(mask).tolist()}")
            
        except Exception as e:
            print(f"✗ Failed to create dataset: {e}")
            return False
        
        # Test multilabel configuration
        print("\nTesting multilabel dataset...")
        dataset_config.problem_type = "multilabel"
        dataset_config.num_classes = 3
        
        try:
            dataset = GeneralizedSegmentationDataset(dataset_config, aug_config, split="train")
            image, mask = dataset[0]
            print(f"✓ Multilabel dataset created successfully")
            print(f"  - Image shape: {image.shape}")
            print(f"  - Mask shape: {mask.shape}")
            
        except Exception as e:
            print(f"✗ Failed to create multilabel dataset: {e}")
            return False
        
        # Test data splits
        print("\nTesting data splits...")
        try:
            splits = create_data_splits(
                dataset_config, aug_config,
                train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
            )
            print(f"✓ Data splits created successfully")
            for split_name, split_dataset in splits.items():
                print(f"  - {split_name}: {len(split_dataset)} samples")
                
        except Exception as e:
            print(f"✗ Failed to create data splits: {e}")
            return False
    
    return True


def test_configuration():
    """Test configuration system."""
    print("\n" + "="*50)
    print("Testing Configuration System")
    print("="*50)
    
    try:
        # Test creating a configuration
        config = SegmentationConfig(
            task_name="test_task",
            dataset=DatasetConfig(
                name="test",
                data_dir=Path("./test"),
                num_classes=4,
                class_names=["bg", "c1", "c2", "c3"],
                problem_type="multiclass"
            ),
            model=ModelConfig(
                architecture="Unet",
                encoder_name="resnet34",
                classes=4
            ),
            training=TrainingConfig(
                batch_size=8,
                num_epochs=10
            ),
            loss=LossConfig(
                name="DiceLoss",
                mode="multiclass"
            ),
            augmentation=AugmentationConfig()
        )
        
        print("✓ Configuration created successfully")
        print(f"  - Task name: {config.task_name}")
        print(f"  - Model: {config.model.architecture} with {config.model.encoder_name}")
        print(f"  - Classes: {config.dataset.num_classes}")
        print(f"  - Problem type: {config.dataset.problem_type}")
        
    except Exception as e:
        print(f"✗ Failed to create configuration: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation from configuration."""
    print("\n" + "="*50)
    print("Testing Model Creation")
    print("="*50)
    
    try:
        from src.segmentation_utils.generalized_train import get_model, get_loss_function
        
        # Test different architectures
        architectures = ["Unet", "UnetPlusPlus", "DeepLabV3Plus"]
        
        for arch in architectures:
            config = SegmentationConfig(
                task_name="test",
                dataset=DatasetConfig(
                    name="test",
                    data_dir=Path("./test"),
                    num_classes=4,
                    problem_type="multiclass"
                ),
                model=ModelConfig(
                    architecture=arch,
                    encoder_name="resnet18",  # Small encoder for testing
                    classes=4
                ),
                training=TrainingConfig(),
                loss=LossConfig(),
                augmentation=AugmentationConfig()
            )
            
            model = get_model(config)
            print(f"✓ {arch} model created successfully")
            
            # Test forward pass
            import torch
            x = torch.randn(2, 3, 128, 128)
            with torch.no_grad():
                output = model(x)
            print(f"  - Input shape: {x.shape}")
            print(f"  - Output shape: {output.shape}")
            
    except Exception as e:
        print(f"✗ Failed to create models: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING REFACTORED SEGMENTATION PIPELINE")
    print("="*60)
    
    # Import torch to check availability
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run tests
    tests = [
        ("Configuration System", test_configuration),
        ("Dataset Loading", test_dataset_loading),
        ("Model Creation", test_model_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nError in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your blastocyst data in the required format")
        print("2. Run: python src/train_blastocyst.py --data_dir ./data/BLASTOCYST")
        print("3. Monitor training with: tensorboard --logdir ./logs")
    else:
        print("Some tests failed. Please check the errors above.")
    print("="*60)


if __name__ == "__main__":
    main()