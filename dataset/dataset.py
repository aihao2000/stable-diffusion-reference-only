import datasets
import os
from PIL import Image
import json
import torch
import cv2
import numpy as np


class ImagesConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ImagesConfig, self).__init__(**kwargs)


class Images(datasets.GeneratorBasedBuilder):
    def __init__(self, **kwargs):
        super(Images, self).__init__(**kwargs)

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        meta_data = {}
        with open(os.path.join(self.config.data_dir, "meta_data.json"), "r") as f:
            meta_data = json.load(f)
        data = []
        if (
            self.config.name == "similar_pairs"
            or self.config.name == "reference_only_for_automatic_coloring"
        ):
            for image1_path in meta_data:
                for image2_path, similarity in meta_data[image1_path]["similar_images"]:
                    data.append(
                        (
                            image1_path,
                            image2_path,
                            similarity,
                        )
                    )
        elif self.config.name == "image_prompt_pairs":
            for image_path in meta_data:
                data.append(image_path, meta_data[image_path]["prompt"])
        print("data size:", len(data))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": datasets.Split.TRAIN, "data": data},
            )
        ]

    BUILDER_CONFIGS = [
        ImagesConfig(
            name="similar_pairs",
            description="simliar pair dataset,item is a pair of similar images",
        ),
        ImagesConfig(
            name="image_prompt_pairs",
            description="image prompt pairs",
        ),
        ImagesConfig(
            name="reference_only_for_automatic_coloring",
            description="reference_only_for_automatic_coloring",
        ),
    ]

    def _info(self):
        if self.config.name == "similar_pairs":
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "image1": datasets.features.Image(),
                        "image1_path": datasets.Value("string"),
                        "image2": datasets.features.Image(),
                        "image2_path": datasets.Value("string"),
                        "similarity": datasets.Value("float32"),
                    }
                )
            )
        elif self.config.name == "image_prompt_pairs":
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "image": datasets.features.Image(),
                        "image_path": datasets.features.Value("string"),
                        "prompt": datasets.Value("string"),
                    }
                )
            )
        elif self.config.name == "reference_only_for_automatic_coloring":
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "prompt": datasets.features.Image(),
                        "blueprint": datasets.features.Image(),  # "image": datasets.features.Image(),
                        "image": datasets.features.Image(),
                    }
                )
            )

    def _generate_examples(self, split, data):
        if self.config.name == "similar_pairs":
            for image1_path, image2_path, similarity in data:
                yield image1_path + ":" + image2_path, {
                    "image1": Image.open(
                        os.path.join(self.config.data_dir, image1_path)
                    ),
                    "image1_path": image1_path,
                    "image2": Image.open(
                        os.path.join(self.config.data_dir, image2_path)
                    ),
                    "image2_path": image2_path,
                    "similarity": similarity,
                }
        elif self.config.name == "reference_only_for_automatic_coloring":
            for image1_path, image2_path, similarity in data:
                try:
                    prompt = Image.open(
                        os.path.join(self.config.data_dir, image1_path)
                    ).convert("RGB")
                    image = Image.open(
                        os.path.join(self.config.data_dir, image2_path)
                    ).convert("RGB")
                    blueprint = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                    blueprint = cv2.adaptiveThreshold(
                        blueprint,
                        255,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY,
                        blockSize=5,
                        C=7,
                    )
                    blueprint = Image.fromarray(blueprint).convert("RGB")
                    blueprint = Image.eval(blueprint, lambda x: 255 - x)
                except Exception as e:
                    continue
                else:
                    yield image1_path + ":" + image2_path, {
                        "prompt": prompt,
                        "blueprint": blueprint,
                        "image": image,
                    }
