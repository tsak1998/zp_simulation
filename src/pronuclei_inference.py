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

    image_file_paths = sorted(list(slide_pth.glob("*.jpg")), key=lambda x: int(x.stem))[
        :max_frame
    ]

    images = [Image.open(img_path) for img_path in tqdm(image_file_paths)]

    val_dataset = ImageDataset(images=images, masks=images)

    val_dataloader = DataLoader(val_dataset, batch_size=64)

    model.eval()
    from torch.cuda.amp import autocast

    all_masks = []
    for inpt_images, _ in val_dataloader:
        with torch.no_grad():
            with autocast():
                pred_mask = model(inpt_images.to(DEVICE))
                masks = torch.sigmoid(pred_mask).cpu().numpy()
                all_masks.extend([msk for msk in masks])

    pn_size = []
    final_images = []
    upscaled_masks = []
    isolated_pns = []
    for pil_img, mask in zip(images[:], all_masks[:]):
        # Ensure the mask is 2D by removing extra dimensions
        # pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        image_ar = np.stack(3 * [np.array(pil_img)])

        upscaled_mask = cv2.resize(
            mask[1].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )
        pn_size.append(upscaled_mask.sum())

        upscaled_masks.append(upscaled_mask)
        image_pn_isolated = image_ar.copy()
        image_pn_isolated[:, ~upscaled_mask.astype(bool)] = 0
        isolated_pns.append(image_pn_isolated.transpose(1, 2, 0))
        image_ar[2, upscaled_mask.astype(bool)] = 240

        final_images.append(Image.fromarray(image_ar.transpose(1, 2, 0)))

    return final_images, upscaled_masks, pn_size


if __name__ == "__main__":

    model = smp.Unet(
        encoder_name="resnext101_32x48d",  # "resnext101_32x48d",
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
    model.eval()

    model.to(DEVICE)

    path_timelapses = Path(
        "/media/tsakalis/STORAGE/phd/raw_timelapses/"
    )

    all_timelapses = list(path_timelapses.glob("*"))
    save_data_pth = Path('/media/tsakalis/STORAGE/phd/pronuclei_tracking')

    for timelapse_pth in tqdm(all_timelapses):
        slide_id = str(timelapse_pth).split("/")[-1]
        # print(slide_id)

        try:

            final_images, upscaled_masks, pn_size = inference_whole_slide(
                model, timelapse_pth, MAX_FRAME


            )
            
            np.savez_compressed(save_data_pth / f'masks/{slide_id}.npz',
                    all_masks=np.stack(upscaled_masks))

            # np.savez_compressed("stacked_compressed.npz", stacked=stacked)

         
        except Exception as e:
            print(e)
            print(timelapse_pth)
