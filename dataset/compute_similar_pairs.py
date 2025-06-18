import os
import json
import sys
import torch
from tqdm import tqdm

from image_retreiver.clip_retrieval import CLIPRetrieval

cache_path = ".cache"
meta_data_path = "meta_data.json"

retrieval = CLIPRetrieval(dtype=torch.float32)

retrieval.load_cache(cache_path)
print("original image feautres " + str(len(retrieval.image_features)))
retrieval.autoremove()

print("image feautres after autoremove " + str(len(retrieval.image_features)))

similar_pairs = retrieval.get_most_similar_images(threshold=90, device="cpu")

print("similar_pairs " + str(len(similar_pairs)))

meta_data = {}
for image_path_1, image_path_2, similarity in tqdm(similar_pairs):
    if not os.path.exists(image_path_1) or not os.path.exists(image_path_2):
        continue
    if image_path_1 not in meta_data:
        meta_data[image_path_1] = {}
        meta_data[image_path_1]["similar_images"] = []
    similar_images = [
        image_path_2
        for image_path_2, similarity in meta_data[image_path_1]["similar_images"]
    ]
    if image_path_2 not in similar_images:
        meta_data[image_path_1]["similar_images"].append(
            (image_path_2, float(similarity))
        )

print("similar_pairs " + str(len(similar_pairs)))

with open(meta_data_path, "w") as f:
    json.dump(meta_data, f, indent=4)
