import transformers
import torch
import glob
import json
import csv
from PIL import Image
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
import sys
from tqdm import tqdm

prompts = {}
with open(
    "/home/aihao/workspace/DeepLearningContent/datasets/images/data/pixiv/prompts.json",
    "r",
) as f:
    prompts = json.load(f)

similar_pairs = []
with open(
    "/home/aihao/workspace/DeepLearningContent/datasets/images/data/pixiv/similar_pairs.csv",
    "r",
) as f:
    reader = csv.reader(f)
    for row in reader:
        similar_pairs.append(row)


tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="tokenizer",
)
tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="tokenizer_2",
)
text_encoder_one = transformers.CLIPTextModel.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="text_encoder",
).to("cuda:0")
text_encoder_two = transformers.CLIPTextModelWithProjection.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="text_encoder_2",
).to("cuda:0")

image_clip_processor_one = transformers.CLIPImageProcessor.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="image_clip_processor",
)
image_clip_processor_two = transformers.CLIPImageProcessor.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="image_clip_processor_2",
)
image_encoder_one = transformers.CLIPVisionModelWithProjection.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="image_encoder",
).to("cuda:1")
image_encoder_two = transformers.CLIPVisionModelWithProjection.from_pretrained(
    "/home/aihao/workspace/DeepLearningContent/models/sdxl_jc/sdxl_jc-base-1.0",
    subfolder="image_encoder_2",
).to("cuda:1")
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
image_encoder_one.requires_grad_(False)
image_encoder_two.requires_grad_(False)


def encode_prompt(
    text_encoder: transformers.CLIPTextModel,
    text_encoder_2: transformers.CLIPTextModelWithProjection,
    tokenizer: transformers.CLIPTokenizer,
    tokenizer_2: transformers.CLIPTokenizer,
    image_encoder: transformers.CLIPVisionModelWithProjection,
    image_encoder_2: transformers.CLIPVisionModelWithProjection,
    image_clip_processor: transformers.CLIPImageProcessor,
    image_clip_processor_2: transformers.CLIPImageProcessor,
    text_prompt: str,
    text_prompt_2: Optional[str] = None,
    image_prompt: Image = None,
    image_prompt_2: Image = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
):
    if text_prompt is not None and isinstance(text_prompt, str):
        batch_size = 1
    elif text_prompt is not None and isinstance(text_prompt, list):
        batch_size = len(text_prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define imaege processors and image encoders
    image_clip_processors = (
        [image_clip_processor, image_clip_processor_2]
        if image_clip_processor is not None
        else [image_clip_processor_2]
    )
    image_encoders = (
        [image_encoder, image_encoder_2]
        if image_encoder is not None
        else [image_encoder_2]
    )

    # Define tokenizers and text encoders
    tokenizers = [tokenizer, tokenizer_2] if tokenizer is not None else [tokenizer_2]
    text_encoders = (
        [text_encoder, text_encoder_2] if text_encoder is not None else [text_encoder_2]
    )

    if prompt_embeds is None:
        text_prompt_2 = text_prompt_2 or text_prompt
        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [text_prompt, text_prompt_2]
        for text_prompt, tokenizer, text_encoder in zip(
            prompts, tokenizers, text_encoders
        ):
            text_inputs = tokenizer(
                text_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to("cuda:0"),
                output_hidden_states=True,
            )
            if image_prompt is None:
                pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        if image_prompt is not None:
            image_prompt_2 = image_prompt_2 or image_prompt

            image_prompts = [image_prompt, image_prompt_2]

            image_prompt_embeds_list = []
            for image_prompt, image_clip_processor, image_encoder in zip(
                image_prompts, image_clip_processors, image_encoders
            ):
                pixel_values = image_clip_processor(
                    image_prompt,
                    return_tensors="pt",
                ).pixel_values

                image_prompt_embeds = image_encoder(
                    pixel_values.to("cuda:1"),
                ).image_embeds

                pooled_prompt_embeds = image_prompt_embeds

                image_prompt_embeds_list.append(image_prompt_embeds)

            image_prompt_embeds = torch.concat(image_prompt_embeds_list, dim=-1)
            bs, image_embeds_dim = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.reshape((bs, 1, image_embeds_dim))
            image_prompt_embeds = image_prompt_embeds.repeat(
                1, tokenizer.model_max_length, 1
            )
            prompt_embeds = prompt_embeds + image_prompt_embeds.to("cuda:0")
    return (
        prompt_embeds,
        pooled_prompt_embeds,
    )


new_similar_pairs = []

for row in tqdm(similar_pairs):
    image_path_1, image_path_2, _ = row
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoder_one,
        text_encoder_two,
        tokenizer_one,
        tokenizer_two,
        image_encoder_one,
        image_encoder_two,
        image_clip_processor_one,
        image_clip_processor_two,
        text_prompt=prompts[image_path_2],
        image_prompt=Image.open(image_path_1),
    )
    new_similar_pairs.append(
        [
            image_path_1,
            prompts[image_path_1],
            image_path_2,
            prompts[image_path_2],
            _,
            prompt_embeds.tolist(),
            pooled_prompt_embeds.tolist(),
        ]
    )


save_path = "/home/aihao/workspace/DeepLearningContent/datasets/images/data/pixiv/similar_pairs.csv"
with open(save_path, "w") as f:
    csv.writer(f).writerows(new_similar_pairs)
