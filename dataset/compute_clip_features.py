import os
import json
import sys
import torch
from tqdm import tqdm
from PIL import Image
import cv2

from image_retriever.clip_retrieval import CLIPRetrieval

cache_path = ".cache"
meta_data_path = "meta_data.json"
clip_model_path = "../../models/clip/CLIP-ViT-bigG-14-laion2B-39B-b160k"
retrieval = CLIPRetrieval(clip_model_path, dtype=torch.float32)
retrieval.load_cache(cache_path)
retrieval.autoremove()
data_path = "data"
retrieval.add_images_by_directory_path(data_path)
retrieval.save_cache(cache_path)
