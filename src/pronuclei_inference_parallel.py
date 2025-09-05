import multiprocessing
from pathlib import Path
import torch
import cv2
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
from skimage.morphology import binary_dilation
from torch.utils.data import DataLoader
from .segmentation_utils.dataloader import ImageDataset
from torch.cuda.amp import autocast
from tqdm import tqdm

DEVICE = "cuda"
MAX_FRAME = 200

def inference_whole_slide(model, slide_pth: Path, max_frame: int):
    image_file_paths = sorted(list(slide_pth.glob("*.jpg")), key=lambda x: int(x.stem))[:max_frame]
    images = [Image.open(img_path) for img_path in tqdm(image_file_paths)]
    val_dataset = ImageDataset(images=images, masks=images)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    model.eval()
    all_masks = []
    for inpt_images, _ in val_dataloader:
        with torch.no_grad():
            with autocast():
                pred_mask = model(inpt_images.to(DEVICE))
                masks = torch.sigmoid(pred_mask).cpu().numpy()
                all_masks.extend(masks)

    final_images = []
    upscaled_masks = []
    for pil_img, mask in zip(images, all_masks):
        image_ar = np.stack(3 * [np.array(pil_img)])
        upscaled_mask = cv2.resize(mask[1].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST)
        upscaled_masks.append(upscaled_mask)
        image_ar[2, upscaled_mask.astype(bool)] = 240
        final_images.append(Image.fromarray(image_ar.transpose(1, 2, 0)))
        
    return final_images, upscaled_masks, None

def process_timelapse(timelapse_pth_str):
    timelapse_pth = Path(timelapse_pth_str)
    slide_id = timelapse_pth.name

    # Load the model within each worker so each process has its own CUDA context
    model = smp.Unet(
        encoder_name="resnext101_32x48d",
        encoder_weights="instagram",
        in_channels=3,
        classes=3,
    )
    model.load_state_dict(
        torch.load(
            "/home/tsakalis/ntua/phd/cellforge/cellforge/model_weights/pronuclei_simple.pt",
            weights_only=True,
        )
    )
    model.to(DEVICE)
    model.eval()

    try:
        final_images, upscaled_masks, _ = inference_whole_slide(model, timelapse_pth, MAX_FRAME)
        # Save the npz file; the process returns None.
        save_dir = Path('/media/tsakalis/STORAGE/phd/pronuclei_tracking/masks')
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_dir / f'{slide_id}.npz', all_masks=np.stack(upscaled_masks))
    except Exception as e:
        print(f"Error processing {timelapse_pth}: {e}")
        
    return None  # Explicitly return None

if __name__ == "__main__":
    path_timelapses = Path("/media/tsakalis/STORAGE/phd/raw_timelapses/")
    all_timelapses = [str(p) for p in list(path_timelapses.glob("*"))]

    # Use multiprocessing to process each timelapse; results are not returned.
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(process_timelapse, all_timelapses)
