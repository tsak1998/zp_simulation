import os
from typing import Literal

from src.complete_pronuclei_train import BATCH_SIZE


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from .segmentation_utils.dataloader import (
    ImageCircleDatasetV2,
    ImageCircleDatasetSeperate,
)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import contextmanager

from PIL import Image
device = "cuda"


from enum import StrEnum

BATCH_SIZE: int = 32

class TypeOfProblem(StrEnum):

    MULTILABEL = 'multilabel'
    MULTICLASS = 'multiclass'



def inference_list_of_slides(model, images: list[Image.Image], max_frame: int, type_of_problem: TypeOfProblem):

    val_dataset = ImageCircleDatasetV2(images, images, images, images, predict=True)

    val_dataloader = DataLoader(val_dataset, batch_size=32)

    model.eval()

    all_masks = []
    for inpt_images, _ in val_dataloader:
        with torch.no_grad():
            # with autocast():

            pred_mask = model(inpt_images.to(device))
            #


            # masks = torch.softmax(pred_mask,axis=1).cpu().numpy()>0.05
            masks = torch.sigmoid(pred_mask).cpu().numpy() > 0.05

            
            all_masks.extend([msk for msk in masks])
            

    pn_size = []
    final_images = []
    upscaled_masks = []
    isolated_pns = []
    for pil_img, mask in zip(images[:], all_masks[:]):
        # Ensure the mask is 2D by removing extra dimensions
        # pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        image_ar = np.stack(3 * [np.array(pil_img)])

        upscaled_mask1 = cv2.resize(
            mask[1].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )
        upscaled_mask2 = cv2.resize(
            mask[0].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )
        # upscaled_mask3 = cv2.resize(
        #     mask[3].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        # )

        # pn_size.append(upscaled_mask.sum())

        upscaled_masks.append((upscaled_mask1, upscaled_mask2))
        image_pn_isolated = image_ar.copy()
        image_pn_isolated[:, ~upscaled_mask1.astype(bool)] = 0
        isolated_pns.append(image_pn_isolated.transpose(1, 2, 0))
        image_ar[0, upscaled_mask1.astype(bool)] = 1
        image_ar[1, upscaled_mask2.astype(bool)] = 1
        # image_ar[2, upscaled_mask3.astype(bool)] = 1

        final_images.append(Image.fromarray(image_ar.transpose(1, 2, 0)))

    return (
        final_images,
        upscaled_masks,
    )

def inference_whole_slide(model, slide_pth: Path, max_frame: int, type_of_problem: TypeOfProblem):

    image_file_paths = sorted(list(slide_pth.glob("*.jpg")), key=lambda x: int(x.stem))[
        :max_frame
    ]

    images = [Image.open(img_path) for img_path in tqdm(image_file_paths)]

    val_dataset = ImageCircleDatasetV2(images, images, images, images, predict=True)

    val_dataloader = DataLoader(val_dataset, batch_size=32)

    model.eval()

    all_masks = []
    for inpt_images, _ in val_dataloader:
        with torch.no_grad():
            # with autocast():

            pred_mask = model(inpt_images.to(device))
            #


            # masks = torch.softmax(pred_mask,axis=1).cpu().numpy()>0.05
            masks = torch.sigmoid(pred_mask).cpu().numpy() > 0.05

            
            all_masks.extend([msk for msk in masks])
            

    pn_size = []
    final_images = []
    upscaled_masks = []
    isolated_pns = []
    for pil_img, mask in zip(images[:], all_masks[:]):
        # Ensure the mask is 2D by removing extra dimensions
        # pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        image_ar = np.stack(3 * [np.array(pil_img)])

        upscaled_mask1 = cv2.resize(
            mask[1].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )
        upscaled_mask2 = cv2.resize(
            mask[0].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )
        # upscaled_mask3 = cv2.resize(
        #     mask[3].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        # )

        # pn_size.append(upscaled_mask.sum())

        upscaled_masks.append((upscaled_mask1, upscaled_mask2))
        image_pn_isolated = image_ar.copy()
        image_pn_isolated[:, ~upscaled_mask1.astype(bool)] = 0
        isolated_pns.append(image_pn_isolated.transpose(1, 2, 0))
        image_ar[0, upscaled_mask1.astype(bool)] = 1
        image_ar[1, upscaled_mask2.astype(bool)] = 1
        # image_ar[2, upscaled_mask3.astype(bool)] = 1

        final_images.append(Image.fromarray(image_ar.transpose(1, 2, 0)))

    return (
        final_images,
        upscaled_masks,
    )