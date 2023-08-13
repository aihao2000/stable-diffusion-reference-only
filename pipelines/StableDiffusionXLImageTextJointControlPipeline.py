from diffusers import StableDiffusionXLPipeline
import diffusers
import transformers
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class StableDiffusionXLImageTextJointControlPipeline:
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
        self.image_encoder: transformers.CLIPTextModelWithProjection = image_encoder
        self.tokenizer: transformers.CLIPTokenizer = tokenizer
        self.unet: diffusers.UNet2DConditionModel = unet
        self.scheduler: diffusers.schedulers.KarrasDiffusionSchedulers = scheduler

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        None
