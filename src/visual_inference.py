import os


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

device = "cuda"


@contextmanager
def video_writer_context(output_path, frame_height, frame_width, fps=15):
    # Create a dummy figure to get graph size
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.canvas.draw()
    graph_h, graph_w = fig.canvas.get_width_height()
    plt.close(fig)

    scale = frame_height / graph_h
    graph_resized_width = int(graph_w * scale)
    output_size = (frame_width + graph_resized_width, frame_height)

    output = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"XVID"), fps, output_size
    )

    try:
        yield output, graph_resized_width
    finally:
        output.release()
        cv2.destroyAllWindows()


def generate_tensorboard_video(slide_images, slide_masks, max_frames=20):
    video = []
    for i in range(min(len(slide_images), max_frames)):
        frame = np.array(slide_images[i])
        mask1, mask2 = slide_masks[i]

        overlay = frame.copy()
        overlay[mask1.astype(bool)] = [255, 0, 0]  # Red for PN1
        overlay[mask2.astype(bool)] = [0, 255, 0]  # Green for PN2

        video.append(torch.tensor(overlay).permute(2, 0, 1))  # C, H, W

    video_tensor = torch.stack(video) / 255.0  # Normalize to [0, 1]
    return video_tensor


def generate_video(
    slide_images, slide_masks, output_path, frame_height=500, frame_width=500
):
    pn_size1 = []
    pn_size2 = []

    with video_writer_context(output_path, frame_height, frame_width) as (
        output,
        graph_resized_width,
    ):
        for frame_idx, frame in enumerate(slide_images):

            if len(slide_masks[frame_idx]) == 2:

                pn_size1.append(slide_masks[frame_idx][0].sum())
                pn_size2.append(slide_masks[frame_idx][1].sum())

            x = np.arange(start=0, stop=frame_idx + 1, step=1)

            fig, ax = plt.subplots()
            ax.plot(x, pn_size1)
            ax.plot(x, pn_size2)
            ax.legend(["PN 1", "PN 2"])
            ax.set_title(f"Accumulated PN Size (Frame {frame_idx})")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Accumulated Area")
            fig.tight_layout()
            fig.canvas.draw()

            plot_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)

            plot_resized = cv2.resize(plot_img, (graph_resized_width, frame_height))
            plot_bgr = cv2.cvtColor(plot_resized, cv2.COLOR_RGB2BGR)

            combined = np.hstack((np.array(frame), plot_bgr))

            output.write(combined)
            cv2.imshow("output", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def inference_whole_slide(model, slide_pth: Path, max_frame: int):

    image_file_paths = sorted(list(slide_pth.glob("*.jpg")), key=lambda x: int(x.stem))[
        :max_frame
    ]

    images = [Image.open(img_path) for img_path in tqdm(image_file_paths)]

    val_dataset = ImageCircleDatasetV2(images, images, images, images, predict=True)

    val_dataloader = DataLoader(val_dataset, batch_size=32)

    model.eval()
    from torch.cuda.amp import autocast

    all_masks = []
    for inpt_images, _ in val_dataloader:
        with torch.no_grad():
            # with autocast():

            pred_mask = model(inpt_images.to(device))
            #
            masks = torch.softmax(pred_mask,axis=1).cpu().numpy()>0.05
            # masks = torch.sigmoid(pred_mask).cpu().numpy() > 0.05

            
            all_masks.extend([msk for msk in masks])
            # breakpoint()

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


if __name__ == "__main__":

    model_pronuclei = smp.DPT(
        encoder_name="tu-vit_base_patch16_224.augreg_in21k",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )
    # model_pronuclei = smp.Unet(
    #     encoder_name="mit_b5",
    #     encoder_weights="imagenet",
    #     in_channels=4,
    #     classes=4,
    # )


    type_of_problem = "multilabel"
    model_name = "multiclass_dpt-vit_base_patch16_224.augreg_in21k_3_classes_WHOLE_SINGLE_MASK_FINAL2"#"multiclass_ub5"-mit_
    # f"{type_of_problem}_{model_pronuclei.__dict__['name']}",
    model_pronuclei.load_state_dict(
        torch.load(
            f"/home/tsakalis/ntua/phd/cellforge/cellforge/model_weights/{model_name}.pt",
            weights_only=True,
        )
    )
    model_pronuclei.eval()

    model_pronuclei.to("cuda")
    sample_ids = [
        # "D2016.07.08_S1366_I149_11",
        # "D2016.10.18_S1418_I149_6",
        # "D2016.10.18_S1418_I149_8",
        # "D2016.10.18_S1418_I149_11",
        # #
        #  "D2016.07.08_S1366_I149_11",
        # "D2016.10.18_S1418_I149_6",
        # "D2016.10.18_S1418_I149_8",
        # "D2016.10.18_S1418_I149_11",
        # #
        # "D2017.02.20_S1506_I149_1",
        #
        #  "D2012.10.19_S0346_I149_2",
        # "D2016.03.07_S1251_I149_4",
        # "D2017.02.01_S1488_I149_5",
        # "D2014.06.13_S0785_I149_6",
        "D2013.01.21_S0435_I149_2",
        # "D2018.10.27_S01889_I0149_D_7",
        # "D2016.12.16_S1455_I149_3",
        # "D2013.04.03_S0525_I149_11",
        # "D2013.10.05_S0618_I149_6",
    ]
    sample_path = Path(
        "/home/tsakalis/ntua/phd/cellforge/cellforge/data/raw_timelapses"
    )

    sample_hdd_path = Path('/media/tsakalis/STORAGE/phd/raw_timelapses')

    all_pn_areas = []
    for sample_idx, sample_id in enumerate(sample_ids):

        full_path = sample_path / sample_id

        if not full_path.is_file():
            full_path = sample_hdd_path / sample_id
            

        slide_images, slide_masks = inference_whole_slide(
            model_pronuclei, full_path, 200
        )


        # plt.plot(pn_area)
        output_path = Path(
            f"/home/tsakalis/pn_samples_all/seperate_pn_{sample_idx}_{sample_id}_{model_name}.mp4"
        )
        generate_video(slide_images, slide_masks, output_path)
