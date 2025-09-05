import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import io

base_slide_pth = Path('/media/tsakalis/STORAGE/synology/SynologyDrive/extra_slides')
base_extracted_images_pth = Path('/media/tsakalis/STORAGE/phd/raw_timelapses')

def save_image(image, image_path):
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
    except Exception as e:
        print(f"Error saving image to {image_path}: {e}", flush=True)

def load_images_from_db(slide_id: str) -> None:
    print(f"Processing slide: {slide_id}", flush=True)
    pdb_file = base_slide_pth / f'{slide_id}.pdb'
    try:
        with sqlite3.connect(pdb_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT Well, Run, Focal, Time, Image FROM IMAGES
                WHERE Focal = 0 AND Run > 1
            """)
            rows = cursor.fetchall()
            tasks = []
            # Process images with threads concurrently.
            with ThreadPoolExecutor(max_workers=20) as file_io_executor:
                for row in rows:
                    well, run, focal, time_val, img_blob = row
                    try:
                        image = Image.open(io.BytesIO(img_blob))
                    except Exception as e:
                        print(f"Error opening image from slide {slide_id}: {e}", flush=True)
                        continue
                    output_folder = base_extracted_images_pth / f"{slide_id}_{well}"
                    image_filename = f"{run-1}_{focal}.jpg"
                    image_path = output_folder / image_filename
                    tasks.append(file_io_executor.submit(save_image, image, image_path))
                for task in tqdm(tasks, total=len(tasks), desc=f"Slide {slide_id}", leave=False):
                    task.result()
    except Exception as e:
        print(f"Error processing slide {slide_id}: {e}", flush=True)

if __name__ == '__main__':
    all_slide_ids = [slide.stem for slide in base_slide_pth.glob('*')][::-1]

    # last_idx = all_slide_ids.index('D2014.11.21_S0891_I149')/

    
    print(f"Total slides to process: {len(all_slide_ids)}", flush=True)
    # Process each slide sequentially.
    for slide_id in tqdm(all_slide_ids, desc="Processing slides"):
        load_images_from_db(slide_id)
