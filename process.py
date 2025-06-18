from glob import glob
from PIL import Image
import os

png_paths = glob("./**/*.png", recursive=True)
for png_path in png_paths:
    os.system(f"rm {png_path}")
