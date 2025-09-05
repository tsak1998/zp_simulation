
from typing import Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


import torch
import numpy as np
import polars as pl

from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from PIL import Image
from PIL.ImageFile import ImageFile

from tqdm import tqdm

device = 'cpu'

data_pth = Path('../data/')

model_weights_pth =Path('../model_weights/')

image_size = 224
normalize_tensor = T.Compose([
    T.Lambda(lambda x: x.resize(
        (image_size, image_size), Image.Resampling.LANCZOS)),
    T.Lambda(lambda x: x.convert("RGB")),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Lambda(lambda x: x[:1, :]),
])

def load_image(pth: Path) -> Image.Image:
    
    return normalize_tensor(Image.open(pth))

def load_images(df_row: dict) -> dict:
    
    all_images = list(((data_pth/'raw_timelapses')/df_row['id']).glob('*.jpg'))

    all_images = sorted(all_images, key=lambda x: int(x.stem.split('_')[0]))[df_row['peak']:]

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(load_image, img): img for img in all_images}
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    return {"id": df_row['id'], "images": results}

    

if __name__=='__main__':

    inner_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )

    whole_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
    inner_model.load_state_dict(
    torch.load(
        model_weights_pth/'inner_embryo.pt', map_location='cpu'
    ))

    whole_model.load_state_dict(
        torch.load(
            model_weights_pth/'full_embryo.pt', map_location='cpu'
        ))

        
    inner_model.eval()
    whole_model.eval()
    
    inner_model_compiled = torch.compile(inner_model)
    whole_model_compiled = torch.compile(whole_model)

    

    relevant_embryo_and_their_peaks = pl.read_csv(data_pth/'area_peaks.csv')

    relevant_embryo_and_their_peaks.to_dicts()

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(load_images, row): row for row in relevant_embryo_and_their_peaks.to_dicts()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    

    for embryo in tqdm(results, total=len(results)):

        
        embryo_id = embryo['id']
        images = embryo['images']
        base_embryo_msk_pth = (data_pth/'extracted_masks')/embryo_id

        base_embryo_msk_pth.mkdir(parents=True, exist_ok=True)

        if len(images)==0:
            continue

        batch = torch.stack(images)

        with torch.no_grad():

            mask_inner = (torch.sigmoid(inner_model_compiled(batch.to(device)))>0.5).double()
            mask_whole = (torch.sigmoid(whole_model_compiled(batch.to(device)))>0.5).double()
            
        zps = mask_whole-mask_inner

        np.save(base_embryo_msk_pth/'mask_inner.npy', mask_inner.numpy())
        np.save(base_embryo_msk_pth/'mask_whole.npy', mask_whole.numpy())
        np.save(base_embryo_msk_pth/'zps.npy', zps.numpy())



    # dataloader = DataLoader(
    # [normalize_tensor(Image.open(img)) for img in image_file_paths[-300:]],
    # shuffle=False,
    # batch_size=12)