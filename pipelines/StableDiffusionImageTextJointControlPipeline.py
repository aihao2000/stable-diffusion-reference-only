import diffusers
import transformers
import torch
from diffusers import StableDiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union


class StableDiffusionTextJointControlPipeline:
    def __init__(
        self,
        vae: diffusers.AutoencoderKL,
        text_encoder: transformers.CLIPTextModel,
        image_encoder: transformers.CLIPVisionModel,
        tokenizer: transformers.CLIPTokenizer,
        unet: diffusers.UNet2DConditionModel,
        scheduler: diffusers.schedulers.KarrasDiffusionSchedulers,
    ):
        self.vae: diffusers.AutoencoderKL = vae
        self.text_encoder: transformers.CLIPTextModel = text_encoder
        self.image_encoder: transformers.CLIPVisionModel = image_encoder
        self.tokenizer: transformers.CLIPTokenizer = tokenizer
        self.unet: diffusers.UNet2DConditionModel = unet
        self.scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = scheduler

    def _encode_prompt(
        self,
        prompts,
    ):
        prompt_embeddings = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to("cuda")

        return prompt_embeddings

    def __call__(
        self,
        condition_image,
        prompts: List = None,
        height: int = None,
        width: int = None,
        num_inference_steps: int = 50,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        init_image = torch.randn((1, 3, height, width)).to("cuda")

        condition_image_embeddings = self.image_encoder(condition_image)[0]
        text_embeddings = self.text_encoder(prompts)[0] if prompts is not None else None
