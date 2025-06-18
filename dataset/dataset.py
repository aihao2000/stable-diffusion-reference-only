import datasets
import os
from PIL import Image
import json
import jsonlines


class ImagesConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ImagesConfig, self).__init__(**kwargs)


class Images(datasets.GeneratorBasedBuilder):
    def __init__(self, **kwargs):
        super(Images, self).__init__(**kwargs)

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        with jsonlines.open(os.path.join(self.config.data_dir, "meta_data.jsonl"), "r") as meta_data:
            data = []
            if (
                self.config.name == "similar_pairs"
                or self.config.name =="reference_only_for_dwpose"
            ):
                for obj in meta_data:
                    reference_image_path=obj['image_path']
                    if not os.path.exists(os.path.join(self.config.data_dir,reference_image_path)):
                        print(reference_image_path+" not exists")
                    for target_image, similarity in obj["similar_images"]:
                        if not os.path.exists(os.path.join(self.config.data_dir,target_image)):
                            print(target_image+" not exists")
                            continue
                        data.append(
                            (
                                reference_image_path,
                                target_image
                            )
                        )
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
            name="reference_only_for_dwpose",
        )
    ]

    def _info(self):
        if self.config.name == "similar_pairs":
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "reference_image": datasets.features.Image(),
                        "reference_image_path": datasets.Value("string"),
                        "target_image": datasets.features.Image(),
                        "target_image_path": datasets.Value("string"),
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
        elif self.config.name == "reference_only_for_dwpose":
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "reference_image": datasets.features.Image(),
                        "target_image": datasets.features.Image(),
                        "blueprint_image": datasets.features.Image()
                        
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
        elif self.config.name=="reference_only_for_dwpose":
            for image1_path, image2_path in data:
                yield image1_path + ":" + image2_path, {
                    "reference_image": Image.open(
                        os.path.join(self.config.data_dir, image1_path)
                    ),
                    "target_image": Image.open(
                        os.path.join(self.config.data_dir, image2_path)
                    ),
                    "blueprint_image":Image.open(
                        os.path.join(self.config.data_dir,image2_path.replace('data','dwpose'))
                    ),
                }
