import os
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageFile, PngImagePlugin
import numpy as np
from tqdm import tqdm
import pickle

Image.MAX_IMAGE_PIXELS = None 
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
PngImagePlugin.MAX_TEXT_MEMORY = 100 * 1024 * 1024 # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_mostly_white(image_path, whiteness_thresh=240, white_ratio_thresh=0.8):
    """
    Check if an image is mostly white.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    white_mask = np.all(arr >= whiteness_thresh, axis=2)
    white_ratio = white_mask.mean()
    return white_ratio >= white_ratio_thresh

def find_white_images(folder, whiteness_thresh=240, white_ratio_thresh=0.8):
    """
    Walk `folder` recursively, show progress with tqdm, and print
    any PNGs that are mostly white.
    """
    list_of_invalid_images = []

    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: {folder!r} is not a directory.")
        return

    # Gather all PNGs so tqdm knows the total count
    png_paths = list(folder.rglob("*.png"))
    if not png_paths:
        print("No PNGs found.")
        return

    pbar = tqdm(png_paths, desc='Scanning PNGs', unit='img')
    for png_path in pbar:
        try:
            if is_mostly_white(png_path, whiteness_thresh, white_ratio_thresh):
                # use tqdm.write so it doesn't disrupt the progress bar
                # tqdm.write(str(png_path))
                list_of_invalid_images.append(png_path)

                new_description = f"Scanning PNGs {len(list_of_invalid_images)} found"
                pbar.set_description(new_description) 

        except Exception as e:
            tqdm.write(f"Skipping {png_path}: {e}")

    return list_of_invalid_images

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python find_white_images.py /path/to/folder [whiteness_thresh] [white_ratio_thresh]")
#         sys.exit(1)

#     folder = sys.argv[1]
#     # Optional overrides
#     wt = int(sys.argv[2]) if len(sys.argv) > 2 else 225
#     wr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7

#     white_imgs = find_white_images(folder, whiteness_thresh=wt, white_ratio_thresh=wr)
#     print(f"Find {len(white_imgs)} white images")

#     out_file = f"{folder.replace('/', '.')}.pkl"
#     with open(out_file, "wb") as f:
#         # convert Paths to strings for simpler unpickling
#         pickle.dump([str(p) for p in white_imgs], f)

def get_child_dir(parent_dir, full_path):
    full = Path(full_path)
    parent = Path(parent_dir)

    try:
        rel = full.relative_to(parent)
        return rel
    except ValueError:
        print("there is a problem when getting the child dir")
        exit(0)
        # return None


# This is the second part, you need to find out the invalid files path first using above commented script
if __name__ == "__main__":
    fn = ".mnt.Z.cuhk_data.HPACG.batch2.pkl"
    parent_dir = "/mnt/Z/cuhk_data/HPACG/batch2"
    relocate_dir = Path("/mnt/Z/cuhk_data/HPACG/batch2/dump")

    if not os.path.exists(relocate_dir):
        os.mkdir(relocate_dir)

    try:
        with open(fn, "rb") as f:
            fp_of_invalid_img = pickle.load(f)
    except Exception as e:
        print(f"[ERROR while loading pickle]: {e}")
        exit(0)

    for fp in tqdm(fp_of_invalid_img):
        old_fp = fp
        new_fp = relocate_dir / get_child_dir(parent_dir, old_fp)

        if not os.path.exists(new_fp.parent):
            new_fp.mkdir(parents=True, exist_ok=True)
        shutil.move(old_fp, new_fp)