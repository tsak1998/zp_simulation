"""Inference script for blastocyst segmentation."""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import SegmentationConfig
from src.segmentation_utils.generalized_dataset import GeneralizedSegmentationDataset
from src.segmentation_utils.generalized_train import get_model
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model
    model = get_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def predict_single_image(
    model: torch.nn.Module, image_path: Path, config: SegmentationConfig, device: str = "cuda"
) -> dict:
    """Predict on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Create preprocessing transform
    transform = A.Compose(
        [
            A.Resize(config.dataset.image_size, config.dataset.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # Apply transform
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)

    # Process output based on problem type
    if config.dataset.problem_type == "multiclass":
        # Get class predictions
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Create separate masks for each class
        masks = {}
        for idx, class_name in enumerate(config.dataset.class_names[1:], 1):
            class_mask = (pred_mask == idx).astype(np.uint8) * 255
            # Resize back to original size
            class_mask = cv2.resize(class_mask, original_size, interpolation=cv2.INTER_NEAREST)
            masks[class_name] = class_mask

    else:  # multilabel
        # Apply sigmoid and threshold
        pred_masks = torch.sigmoid(output).squeeze(0).cpu().numpy()
        pred_masks = (pred_masks > 0.5).astype(np.uint8) * 255

        masks = {}
        for idx, class_name in enumerate(config.dataset.class_names[1:]):
            class_mask = pred_masks[idx]
            # Resize back to original size
            class_mask = cv2.resize(class_mask, original_size, interpolation=cv2.INTER_NEAREST)
            masks[class_name] = class_mask

    return masks


def create_overlay(image: np.ndarray, masks: dict, alpha: float = 0.5) -> np.ndarray:
    """Create overlay visualization of masks on image."""
    # Define colors for each class
    colors = {
        "zona_pellucida": (255, 0, 0),  # Red
        "inner_cell_mass": (0, 255, 0),  # Green
        "trophectoderm": (0, 0, 255),  # Blue
    }

    overlay = image.copy()

    for class_name, mask in masks.items():
        if class_name in colors:
            color = colors[class_name]
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color

            # Blend with original image
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

    return overlay


def main():
    parser = argparse.ArgumentParser(description="Run inference on blastocyst images")

    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("input", type=str, help="Input image or directory")
    parser.add_argument(
        "--output", type=str, default="./predictions", help="Output directory for predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    parser.add_argument("--save_overlay", action="store_true", help="Save overlay visualizations")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency (0-1)")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_checkpoint(args.checkpoint, args.device)
    print(f"Model loaded: {config.model.architecture} with {config.model.encoder_name}")
    print(f"Problem type: {config.dataset.problem_type}")
    print(f"Classes: {config.dataset.class_names}")

    # Prepare input paths
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        # Get all images in directory
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]:
            image_paths.extend(input_path.glob(ext))
        image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in {input_path}")
        return

    print(f"Found {len(image_paths)} images")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each class
    for class_name in config.dataset.class_names[1:]:
        (output_dir / class_name).mkdir(exist_ok=True)

    if args.save_overlay:
        (output_dir / "overlays").mkdir(exist_ok=True)

    # Process images
    print("\nProcessing images...")
    for image_path in tqdm(image_paths):
        # Predict
        masks = predict_single_image(model, image_path, config, args.device)

        # Save individual masks
        for class_name, mask in masks.items():
            mask_path = output_dir / class_name / f"{image_path.stem}.png"
            Image.fromarray(mask).save(mask_path)

        # Save overlay if requested
        if args.save_overlay:
            image = np.array(Image.open(image_path).convert("RGB"))
            overlay = create_overlay(image, masks, args.alpha)
            overlay_path = output_dir / "overlays" / f"{image_path.stem}_overlay.png"
            Image.fromarray(overlay).save(overlay_path)

    print(f"\nPredictions saved to: {output_dir}")
    print("Structure:")
    print(f"{output_dir}/")
    for class_name in config.dataset.class_names[1:]:
        print(f"├── {class_name}/     # {class_name} masks")
    if args.save_overlay:
        print(f"└── overlays/    # Overlay visualizations")


if __name__ == "__main__":
    main()
