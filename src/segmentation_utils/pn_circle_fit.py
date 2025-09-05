#!/usr/bin/env python
# coding: utf-8

import json
from typing import Optional, cast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import binary_closing, binary_dilation, binary_erosion
from torchvision.transforms.functional import rotate
from tqdm import tqdm
from multiprocessing import Pool

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert


def find_circle_centers(smoothed_img):

    # Invert the horse image
    image = invert(smoothed_img)

    # perform skeletonization
    skeleton = skeletonize(image, method='lee')

    coords = np.column_stack(np.where(skeleton))

    # Find the leftmost and rightmost points
    leftmost = coords[np.argmin(coords[:, 1])]
    rightmost = coords[np.argmax(coords[:, 1])]

    return leftmost, rightmost

def mask_orientation_centroid(image: np.ndarray):
    label_img = label(image)
    regions = regionprops(label_img)
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
    return props.centroid, orientation, props.axis_major_length, props.axis_minor_length


def rotate_image(image_arr: Image.Image, angle: Optional[float] = None):
    if angle is not None:
        return np.array(rotate(image_arr, angle)), angle
    _, orientation, _, _ = mask_orientation_centroid(np.array(image_arr))
    rotation_angle = -np.rad2deg(orientation) + 90

    return np.array(rotate(image_arr, rotation_angle)), rotation_angle


def find_signal(arr: np.ndarray) -> tuple[int, int]:
    max_len = max_start = curr_len = curr_start = 0
    for i, val in enumerate(arr):
        if val:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
        else:
            if curr_len > max_len:
                max_len, max_start = curr_len, curr_start
            curr_len = 0
    if curr_len > max_len:
        max_len, max_start = curr_len, curr_start
    return max_start, max_len


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    x0, y0 = c[0], c[1]
    r = np.sqrt(c[2] + x0**2 + y0**2)
    return x0, y0, r


def get_circle_pts(x0, y0, r, npts=100, tmin=0, tmax=2 * np.pi):
    t = np.linspace(tmin, tmax, npts)
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)
    return x, y


# --- Paths & Load ---

base_pth = Path("/Users/tsakalis/downloads")
COUNTER_SPLIT_LEFT = 2.5
COUNTER_SPLIT_RIGHT = 3.5
FITTED_RADIUS_INCREASE = 1.05

class PnCircle(BaseModel):
    x: float
    y: float
    r: float


class CirclesFit(BaseModel):
    pn1: PnCircle
    pn2: Optional[PnCircle] = None

    @classmethod
    def randomize_pns(cls): ...


def inverse_rotate_point(x, y, rad, center):
    cx, cy = center
    x_shifted = x - cx
    y_shifted = y - cy
    x_new = cx + np.cos(rad) * x_shifted - np.sin(rad) * y_shifted
    y_new = cy + np.sin(rad) * x_shifted + np.cos(rad) * y_shifted
    return x_new, y_new


def fit_pn_circles(mask_img: Image.Image) -> CirclesFit:
    rotated_mask, angle = rotate_image(mask_img)
    centroid, _, major_len, minor_len = mask_orientation_centroid(rotated_mask == 1)
    y0, x0 = centroid
    a1, b1 = int(x0 - major_len * 0.6), int(x0 + major_len * 0.65)
    a2, b2 = int(y0 - minor_len * 0.6), int(y0 + minor_len * 0.65)

    smoothed_img = cv2.GaussianBlur(
        binary_dilation(rotated_mask[a2:b2, a1:b1]).astype(np.uint8), (9, 9), 0
    )

    contours = find_contours(smoothed_img, None)

    for contour in contours:
        half1 = contour[contour[:, 1] < smoothed_img.shape[1] // COUNTER_SPLIT_LEFT]
        half2 = contour[
            contour[:, 1] > 2 * smoothed_img.shape[1] // COUNTER_SPLIT_RIGHT
        ]

    x = half1[:, 1]
    y = -half1[:, 0]
    x0, y0, r = fit_circle(x, y)
    rotated_x, rotated_y = inverse_rotate_point(
        x0 + a1, -y0 + a2, np.deg2rad(angle), (250, 250)
    )
    pn_circle1 = PnCircle(x=float(rotated_x), y=float(rotated_y), r=FITTED_RADIUS_INCREASE*float(r))

    if major_len / minor_len < 1.5:
        return CirclesFit(pn1=pn_circle1)

    x = half2[:, 1]
    y = -half2[:, 0]
    x0, y0, r = fit_circle(x, y)
    rotated_x, rotated_y = inverse_rotate_point(
        x0 + a1, -y0 + a2, np.deg2rad(angle), (250, 250)
    )
    pn_circle2 = PnCircle(x=float(rotated_x), y=float(rotated_y), r=FITTED_RADIUS_INCREASE*float(r))

    return CirclesFit(pn1=pn_circle1, pn2=pn_circle2)


from typing import TypeVar, Generic

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    pass


def sample_frames(area: Array, n_samples: int = 5) -> Array:
    max_start, max_len = find_signal(area > 10)
    q_70 = np.quantile(area[max_start : max_start + max_len], 0.5)
    q_10 = np.quantile(area[max_start : max_start + max_len], 0.25)
    low_samples = max_start + np.argwhere(area[max_start : max_start + max_len] < q_10)
    high_samples = max_start + np.argwhere(area[max_start : max_start + max_len] > q_70)
    return cast(
        Array,
        np.random.choice(np.vstack([low_samples, high_samples]).flatten(), n_samples),
    )


def process_mask_file(mask_pth: Path):
    try:
        sample_id = mask_pth.stem

        all_masks = np.load(mask_pth)["all_masks"]
        area = np.sum(all_masks, axis=(1, 2))
        sampled_frames = sample_frames(area)
        fitted_circles = []
    except Exception as e:
        print(e, mask_pth)
        return None
    for frame_idx in sampled_frames:
        mask_img = Image.fromarray(
            binary_closing(all_masks[frame_idx]).astype(np.uint8)
        )
        try:
            circles_fit = fit_pn_circles(mask_img)
        except Exception as e:
            print(f"Error processing frame {frame_idx} in {sample_id}: {e}")
            continue
        fitted_circles.append({"frame": int(frame_idx), **circles_fit.model_dump()})
    with open(
        pronuclei_stuff_pth / f"fitted_circles_samples/{sample_id}.json", "w"
    ) as f:
        json.dump(fitted_circles, f)


if __name__ == "__main__":
    sample_ids = ["D2016.01.23_S1202_I149_7", "D2016.01.11_S1183_I149_1"]

    pronuclei_stuff_pth = Path("/media/tsakalis/STORAGE/phd/pronuclei_tracking")

    masks_pths = list(((pronuclei_stuff_pth / "masks").glob("*.npz")))
    with Pool(32) as pool:
        # Wrap imap with tqdm for a progress bar
        list(tqdm(pool.imap(process_mask_file, masks_pths), total=len(masks_pths)))
