import enum
import os
from shutil import ExecError
from sys import breakpointhook

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from random import shuffle
from typing import Callable, Literal
import torch

torch.set_float32_matmul_precision("high")
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, dataloader
from torchvision import models
from tqdm import tqdm
from .segmentation_utils.train_unet import train
from .segmentation_utils.dataloader import (
    ImageDataset,
    ImageCircleDatasetSeperate,
    ImageCircleDatasetV2,
)
from PIL import Image

from enum import StrEnum

BATCH_SIZE = 32
DEVICE = "cuda"
MASK_THRESHOLD = 0.5
IMAGE_SIZE = 500


class InferencePrecision(StrEnum):
    FULL = "full"
    HALF = "half"
    MIXED = "mixed"


data_path = Path("/home/tsakalis/ntua/phd/cellforge/cellforge/data/segmentation_data")


def load_image_folder(
    folder_data_pth: Path,
    image_type: Literal["jpg", "png"] = "jpg",
    sort_fn: Callable = lambda x: x.stem,
) -> list:
    image_file_paths = sorted(
        list((folder_data_pth).glob(f"*.{image_type}")), key=sort_fn
    )

    progress_bar = tqdm(image_file_paths)
    progress_bar.set_description("Loading Images...")

    return [Image.open(img_path) for img_path in progress_bar]


def inference(model, X, precision: InferencePrecision = InferencePrecision.FULL, *args):

    match precision:
        case InferencePrecision.MIXED:
            with torch.no_grad():
                with autocast():
                    return model(X, *args)

        case InferencePrecision.FULL:
            with torch.no_grad():
                return model(X, *args)

    return model(X, *args)


import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# from cellforge.segmentation.segmentation_utils.data_utils import ImageDataset, load_image_folder
# from cellforge.segmentation.segmentation_utils.inference import inference, InferencePrecision


def load_negative_examples(whole_embryo_segmentation_model):

    counter_pronuclei_examples_pth = data_path / "negative_pn"
    counter_example_images = load_image_folder(counter_pronuclei_examples_pth)
    dataset = ImageDataset(images=counter_example_images, masks=counter_example_images)
    dataloader = DataLoader(dataset, BATCH_SIZE)

    counter_example_masks = []
    model = whole_embryo_segmentation_model.eval().to(DEVICE)

    for img_batch, _ in tqdm(dataloader, desc="Generating counter-example masks"):
        img_batch = img_batch.to(DEVICE)

        with torch.autocast(device_type="cuda", enabled=True):
            output = (
                torch.sigmoid(
                    inference(model, img_batch, precision=InferencePrecision.MIXED)
                )
                > MASK_THRESHOLD
            )

        output = output.cpu().numpy()  # shape: (B, 1, H, W)

        for mask in output:
            binary_np = (mask[0] * 255).astype(np.uint8)
            three_class_mask = np.zeros((224, 224, 3), dtype=np.uint8)
            three_class_mask[:, :, 0] = binary_np  # embryo channel
            img = Image.fromarray(three_class_mask)
            counter_example_masks.append(img)

    return counter_example_images, counter_example_masks


def create_all_masks(whole_embryo_segmentation_model: torch.nn.Module):
    """
    Generates three-class masks by combining:
    - Predictions from the whole embryo segmentation model
    - Ground truth pronuclei masks
    - Counter-examples (without pronuclei)

    Returns:
        Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]: Full image and mask lists.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    MASK_THRESHOLD = 0.9
    IMAGE_SIZE = 224

    transform_to_tensor = transforms.ToTensor()

    # Load counter-example images (no pronuclei)
    counter_pronuclei_examples_pth = data_path / "negative_pn"
    counter_example_images = load_image_folder(counter_pronuclei_examples_pth)
    dataset = ImageDataset(images=counter_example_images, masks=counter_example_images)
    dataloader = DataLoader(dataset, BATCH_SIZE)

    counter_example_masks = []
    model = whole_embryo_segmentation_model.eval().to(DEVICE)

    for img_batch, _ in tqdm(dataloader, desc="Generating counter-example masks"):
        img_batch = img_batch.to(DEVICE)

        with torch.autocast(device_type="cuda", enabled=True):
            output = (
                torch.sigmoid(
                    inference(model, img_batch, precision=InferencePrecision.MIXED)
                )
                > MASK_THRESHOLD
            )

        output = output.cpu().numpy()  # shape: (B, 1, H, W)

        for mask in output:
            binary_np = (mask[0] * 255).astype(np.uint8)
            three_class_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            three_class_mask[:, :, 0] = binary_np  # embryo channel
            img = Image.fromarray(three_class_mask)
            counter_example_masks.append(img)

    # Load pronuclei samples
    pronuclei_sample_images = load_image_folder(data_path / "pronuclei/images")
    pronuclei_sample_masks = load_image_folder(
        data_path / "pronuclei/masks", image_type="png"
    )
    pronuclei_simple_dataset = ImageDataset(
        images=pronuclei_sample_images, masks=pronuclei_sample_masks
    )
    dataloader = DataLoader(pronuclei_simple_dataset, BATCH_SIZE)

    pronuclei_sample_masks_3_class = []

    for img_batch, mask_batch in tqdm(dataloader, desc="Generating pronuclei masks"):
        img_batch = img_batch.to(DEVICE)

        with torch.no_grad():
            model_output = torch.sigmoid(inference(model, img_batch)) > 0.95

        model_output = model_output.cpu().numpy()  # shape: (B, 1, H, W)

        for idx in range(img_batch.size(0)):
            embryo_mask = model_output[idx][0]  # shape: (H, W)
            pn_mask = mask_batch[idx][0].cpu().numpy() > 0.5  # shape: (H, W)

            three_class_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            three_class_mask[:, :, 0] = (embryo_mask * 255).astype(np.uint8)
            three_class_mask[:, :, 1] = (pn_mask * 255).astype(np.uint8)

            img = Image.fromarray(three_class_mask)
            pronuclei_sample_masks_3_class.append(img)

    # Combine
    full_images = pronuclei_sample_images + counter_example_images
    full_masks = pronuclei_sample_masks_3_class + counter_example_masks

    # Final assertions to catch future bugs
    for i, mask in enumerate(full_masks):
        arr = np.array(mask)
        assert arr.shape == (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ), f"Mask at idx {i} has shape {arr.shape}"

    return full_images, full_masks

    # carefulll with the aligment of masks and images. is not guaranteed here.
    # counter_examples_dataset = ImageDataset(images=counter_example_images, masks=counter_example_masks, transform=True)


def create_all_masks_separate(whole_embryo_segmentation_model: nn.Module):
    """
    This function will take the pronuclei masks and create the other 2 classes.

    We will need some extra samples to be used as counter examples
    (when pronuclei are not showing).

    """
    DEVICE = "cuda"
    BATCH_SIZE = 32
    MASK_THRESHOLD = 0.9
    IMAGE_SIZE = 224

    base_cicle_pth = Path("/media/tsakalis/STORAGE/phd/pronuclei_tracking")
    timelapse_pth = Path(
        "/home/tsakalis/ntua/phd/cellforge/cellforge/data/raw_timelapses"
    )

    all_circle_data = list((base_cicle_pth / "fitted_circles_samples").glob("*.json"))
    import json

    images = []
    masks = []
    for circle_file_pth in tqdm(all_circle_data[:1000]):
        slide_id = str(circle_file_pth).split("/")[-1][:-5]
        with open(circle_file_pth) as f:
            circles = json.load(f)

        for circle in circles:
            full_frame_pth = timelapse_pth / f"{slide_id}/{circle['frame']}_0.jpg"
            frame_img = Image.open(full_frame_pth)
            mask = np.zeros((500, 500, 3), dtype=np.uint8)

            y_grid, x_grid = np.ogrid[:500, :500]

            # Full blob for pn1 on channel 0
            center1 = (int(circle["pn1"]["x"]), int(circle["pn1"]["y"]))
            radius1 = int(circle["pn1"]["r"])
            blob1 = (x_grid - center1[0]) ** 2 + (
                y_grid - center1[1]
            ) ** 2 <= radius1**2
            mask[..., 1][blob1] = 255

            # Full blob for pn2 on channel 1, if available
            if circle["pn2"]:
                center2 = (int(circle["pn2"]["x"]), int(circle["pn2"]["y"]))
                radius2 = int(circle["pn2"]["r"])
                blob2 = (x_grid - center2[0]) ** 2 + (
                    y_grid - center2[1]
                ) ** 2 <= radius2**2
                mask[..., 2][blob2] = 255

            mask_image = Image.fromarray(mask)

            images.append(frame_img)
            masks.append(mask_image)

    # dataset = ImageDataset(images=images, masks=masks, transform=False)

    # dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=BATCH_SIZE)

    # whole_embryo_segmentation_model.eval()
    # whole_embryo_segmentation_model.to(DEVICE)

    # complete_masks = []
    # for batch_im, batch_mask in tqdm(dataloader):

    #     batch_im = batch_im.to(DEVICE)
    #     with torch.no_grad():
    #         pred_masks = (
    #             torch.sigmoid(
    #                 inference(
    #                     whole_embryo_segmentation_model,
    #                     batch_im,
    #                     precision=InferencePrecision.MIXED,
    #                 )
    #             )
    #             > 0.1
    #         )

    #     pred_masks = pred_masks.cpu().numpy()

    #     for prd_msk, msk in zip(pred_masks, batch_mask.cpu().numpy()):

    #         # breakpoint()

    #         msk *= 255

    #         msk[0, ...] = prd_msk.astype(int) * 255

    #         complete_masks.append(
    #             Image.fromarray(msk.astype(np.uint8).transpose(1, 2, 0))
    #         )
    #         # breakpoint()/
    #         # breakpoint()

    # # final_images = []
    # # final_masks = []

    # # for im, msk in zip(images, complete_masks):

    # #     # msk_ar = np.array(msk)[:, :, 0]

    # #     # if msk_ar.sum() < 190190:
    # #     #     continue

    # #     final_images.append(im)
    # #     final_masks.append(msk)

    return images, masks


def zoom_in_pil(img: Image.Image, zoom_factor: float = 1.25) -> Image.Image:
    w, h = img.size
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def create_all_masks_separate_circles():
    """
    This function will take the pronuclei masks and create the other 2 classes.

    We will need some extra samples to be used as counter examples
    (when pronuclei are not showing).

    """
    DEVICE = "cuda"
    BATCH_SIZE = 32
    MASK_THRESHOLD = 0.9
    IMAGE_SIZE = 224

    base_cicle_pth = Path("/media/tsakalis/STORAGE/phd/pronuclei_tracking")
    timelapse_pth = Path(
        "/home/tsakalis/ntua/phd/cellforge/cellforge/data/raw_timelapses"
    )
    timelapse_pth_cold = Path("/media/tsakalis/STORAGE/phd/raw_timelapses")

    all_timelapses = list(timelapse_pth.glob("*")) + list(timelapse_pth_cold.glob("*"))

    all_timelapses_map = {str(pth).split("/")[-1]: pth for pth in all_timelapses}

    all_circle_data = list((base_cicle_pth / "fitted_circles_samples").glob("*.json"))

    all_masks = list((base_cicle_pth / "masks").glob("*"))
    all_masks_map = {
        ".".join(str(pth).split("/")[-1].split(".")[:-1]): pth for pth in all_masks
    }
    import json

    images = []
    full_circles = []
    pn_whole_masks = []
    paths = []

    import random

    for circle_file_pth in tqdm(all_circle_data):
        try:
            slide_id = str(circle_file_pth).split("/")[-1][:-5]

            with open(circle_file_pth) as f:
                circles = json.load(f)

            for circle in circles:
                try:

                    frame_offset = 0

                    slide_path = all_timelapses_map[slide_id]

                    mask_path = all_masks_map[slide_id]

                    # masks = np.load(mask_path)
                    # if 'npz' in str(mask_path):

                    #     masks = masks["all_masks"]

                    if not circle.get("pn2"):
                        continue

                    if "STORAGE" in str(slide_path):
                        frame_offset = 1

                    full_frame_pth = (
                        slide_path / f"{circle['frame']+frame_offset}_0.jpg"
                    )
                except Exception as e:
                    # breakpoint()
                    continue

                frame_img = Image.open(full_frame_pth).copy()

                mask_np = 1

                pn_whole_masks.append(1)

                images.append(frame_img)
                # frame_img.close()
                full_circles.append(circle)
                paths.append(slide_id)
        except Exception as e:
            print(e)

    return images, full_circles, pn_whole_masks, paths


def create_all_masks_full_(
    whole_model: nn.Module,
    pronuclei_model: nn.Module,
    image_size: int = 224,
    batch_size: int = 32,
    thresh_whole: float = 0.5,
    thresh_pn: float = 0.5,
    precision: InferencePrecision = InferencePrecision.FULL,
) -> tuple[
    list[Image.Image],  # original frames
    list[dict],  # circle data per frame
    list[Image.Image],  # whole‑embryo masks
    list[Image.Image],  # pronuclei masks
]:
    """
    1) reuses create_all_masks_separate_circles() to get (images, circle‑dicts);
    2) runs both whole_model and pronuclei_model over every image (batched);
    3) thresholds each output to two binary masks;
    4) returns (images, circles, whole_masks, pronuc_masks).
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) get raw frames + circle metadata
    images, circles, pn_whole_masks, paths = create_all_masks_separate_circles()

    # 2) build a simple image‑only dataset
    from torchvision import transforms

    class _ImgOnlyDS(Dataset):
        def __init__(self, pil_list, size):
            self.imgs = pil_list
            self.tf = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                ]
            )

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            return self.tf(self.imgs[i]), i

    ds = ImageDataset(images, images)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    whole_model = whole_model.eval().to(DEVICE)
    pronuclei_model = pronuclei_model.eval().to(DEVICE)

    whole_masks = []
    pronuc_masks = []

    # 3) batched inference
    for xb, idxs in tqdm(loader, desc="🖥️ Full+PN inference"):

        xb = xb.to(DEVICE)
        with torch.no_grad():
            # whole‑embryo
            if precision == InferencePrecision.MIXED:
                with autocast():
                    out1 = whole_model(xb)
            else:
                out1 = whole_model(xb)
            # pronuclei
            if precision == InferencePrecision.MIXED:
                with autocast():
                    out2 = pronuclei_model(xb)
            else:
                out2 = pronuclei_model(xb)

        # sigmoid → threshold
        m1 = (torch.sigmoid(out1) > thresh_whole).cpu().numpy()  # (B,1,H,W)
        # m2 = (torch.sigmoid(out2) > thresh_pn).cpu().numpy()

        # convert each back to PIL
        for b in range(len(idxs)):
            # breakpoint()
            mask1 = (m1[b, 0] * 255).astype(np.uint8)
            # mask2 = (m2[b,0]*255).astype(np.uint8)
            whole_masks.append(Image.fromarray(mask1))
            # pronuc_masks.append(Image.fromarray(mask2))

    counter_images, counter_masks = load_negative_examples(whole_model)
    counter_example_circles = [None for _ in counter_images]

    all_images = images + counter_images
    all_masks = whole_masks + counter_masks

    all_circles = circles + counter_example_circles
    return all_images, all_circles, all_masks, paths


model_weights = Path("/home/tsakalis/ntua/phd/cellforge/cellforge/model_weights")

if __name__ == "__main__":

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnext101_32x16d",
        encoder_weights="instagram",
        in_channels=3,
        classes=1,
    )

    model.load_state_dict(
        torch.load(model_weights / "inner_embryo.pt", weights_only=True)
    )
    model.eval()

    n_classes = 3
    model_pronuclei = smp.DPT(
        encoder_name="tu-vit_base_patch16_224.augreg_in21k",
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes,
    )

    # model_pronuclei = smp.Unet(
    #     encoder_name="resnext101_32x16d",
    #     encoder_weights="instagram",
    #     in_channels=3,
    #     classes=5
    # )
    # smp.UnetPlusPlus(
    #     encoder_name="se_resnext101_32x4d",  # "resnext101_32x48d",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=3,
    # )

    # dict =  torch.load(model_weights / "inner_embryo.pt", weights_only=True)

    # breakpoint()
    # model_pn.load_state_dict(
    #     torch.load(model_weights / "pronuclei.pt", weights_only=True)
    # )
    # model_pn.eval()

    # full_images, full_masks = create_all_masks(model)
    # full_images, full_masks = create_all_masks_separate_circles()
    full_images, full_circles, full_masks, paths = create_all_masks_full_(model, model)
    # breakpoint
    c = list(zip(full_images, full_circles, full_masks, paths))

    import random

    random.seed(42)  # set a fixed seed
    random.shuffle(c)

    full_images, full_circles, full_masks, paths = zip(*c)

    # breakpoint()

    # full_dataset = ImageDataset(images=full_images, masks=full_masks)
    # generator = torch.Generator().manual_seed(42)/s

    split_idx = int(len(full_images) * 0.8)
    type_of_problem = "multilabel"
    # type_of_problem = "multiclass"
    import json

    with open("test_ids.json", "w+") as f:
        json.dump({"ids": paths[split_idx:]},f)

    match type_of_problem:
        case "multiclass":
            dataset_class = ImageCircleDatasetSeperate
        case "multilabel":
            dataset_class = ImageCircleDatasetV2

    train_dataset = dataset_class(
        images=full_images[:split_idx],
        circles=full_circles[:split_idx],
        whole_embryo_masks=full_masks[:split_idx],
        # pn_masks=pronuc_masks[:split_idx],
        transform=True,
    )
    # full_images[:split_idx], full_masks[:split_idx], transform=True

    val_dataset = dataset_class(
        images=full_images[split_idx:],
        circles=full_circles[split_idx:],
        whole_embryo_masks=full_masks[split_idx:],
        # pn_masks=pronuc_masks[split_idx:]
    )
    # breakpoint()
    # import matplotlib.pyplot as plt

    # img, msk = train_dataset[0]
    # img, msk = val_dataset[2]  # assuming this index works

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # # Plot image
    # axes[0].imshow(img.numpy().transpose(1, 2, 0)[:,:,0])
    # axes[0].set_title("Image")
    # axes[0].axis('off')

    # # Plot mask
    # axes[1].imshow(msk.numpy(),cmap='gray')  # adjust squeeze/transposing if needed
    # axes[1].set_title("Mask")
    # axes[1].axis('off')

    # plt.tight_layout()
    # plt.show()
    # breakpoint()

    # val_dataset[0]

    from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss
    import torch.nn.functional as F

    class DiceBCELoss(torch.nn.Module):
        def __init__(self, dice_kwargs=None, bce_weight=0.5, dice_weight=0.5):
            super().__init__()
            self.dice = DiceLoss(**(dice_kwargs or {}))
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight

        def forward(self, preds, targets):
            dice_loss = self.dice(preds, targets)
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets.float())
            return self.dice_weight * dice_loss + self.bce_weight * bce_loss

    del model
    torch.cuda.empty_cache()
    # model_pronuclei = torch.compile(model_pronuclei)
    loss_fn = DiceLoss(mode=type_of_problem, log_loss=False, from_logits=True)
    # loss_fn = DiceBCELoss(
    #     dice_kwargs=dict(mode=type_of_problem, log_loss=True, from_logits=True),
    #     bce_weight=0.5,
    #     dice_weight=0.5,
    # )

    # loss_fn = TverskyLoss(mode=type_of_problem, log_loss=False, from_logits=True)
    # model_name = f"{type_of_problem}_{model_pronuclei.__dict__['name']}_{n_classes}_other_other"

    # print(f"Training {model_name}")
    # train(
    #     train_dataset,
    #     val_dataset,
    #    model_name,
    #     lr=1e-4,
    #     n_epochs=120,
    #     batch_size=64,
    #     model=model_pronuclei,
    #     loss_fn=loss_fn,
    #     output_last_fn=lambda x: x,
    #     precision="full",
    # )
