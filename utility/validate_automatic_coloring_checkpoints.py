import os
import argparse
import diffusers
import sys
from diffusers import UniPCMultistepScheduler
from PIL import Image
from transformers import CLIPVisionModel
import torch

sys.path.append("src")
from stable_diffusion_reference_only.models.unet_2d_dobule_condition import (
    UNet2DDobuleConditionModel,
)
from stable_diffusion_reference_only.pipelines.stable_diffusion_reference_only_pipeline import (
    StableDiffusionReferenceOnlyPipeline,
)


def log_validation(
    vae,
    image_encoder,
    clip_image_processor,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
):
    pipeline: StableDiffusionReferenceOnlyPipeline = (
        StableDiffusionReferenceOnlyPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            image_encoder=image_encoder,
            clip_image_processor=clip_image_processor,
            unet=unet,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_blueprint) == len(args.validation_prompt):
        blueprint_paths = args.validation_blueprint
        prompt_paths = args.validation_prompt
    elif len(args.validation_blueprint) == 1:
        blueprint_paths = args.validation_blueprint * len(args.validation_prompt)
        prompt_paths = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        blueprint_paths = args.validation_blueprint
        prompt_paths = args.validation_prompt * len(args.validation_blueprint)
    else:
        raise ValueError(
            "number of `args.validation_blueprint` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for prompt_path, blueprint_path in zip(prompt_paths, blueprint_paths):
        blueprint = (
            Image.open(blueprint_path)
            .convert("RGB")
            .resize((args.resolution, args.resolution))
        )
        blueprint = Image.eval(blueprint, lambda x: 255 - x)
        prompt = (
            Image.open(prompt_path)
            .convert("RGB")
            .resize((args.resolution, args.resolution))
        )
        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    prompt=prompt,
                    blueprint=blueprint,
                    num_inference_steps=20,
                    generator=generator,
                    train_image_encoder=args.train_image_encoder,
                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "prompt": prompt,
                "blueprint": blueprint,
                "images": images,
            }
        )

        return image_logs


parser = argparse.ArgumentParser(description="validate checkpoints")
parser.add_argument(
    "--checkpoints_path",
    type=str,
    default="../DeepLearningContent/models/sd_reference_only/automatic_coloring_0.1",
)
parser.add_argument("--validation_path", type=str, default="validation_images")
parser.add_argument(
    "--pretrained_model_path",
    type=str,
    default="../DeepLearningContent/models/sd_reference_only/init_0.1",
)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
)
parser.add_argument("--seed", type=int, default=2221101)
args = parser.parse_args()

checkpoint_paths = os.listdir(args.checkpoints_path)
checkpoint_paths = [
    os.path.join(args.checkpoints_path, checkpoint_path)
    for checkpoint_path in checkpoint_paths
    if checkpoint_path.startswith("checkpoint")
]
if args.output_dir is None:
    args.output_dir = os.path.join(args.checkpoints_path, "validation")

os.makedirs(args.output_dir, exist_ok=True)
for checkpoint_path in checkpoint_paths:
    generator = torch.Generator("cuda").manual_seed(args.seed)
    checkpoint = checkpoint_path.split("-")[1]
    pipeline = StableDiffusionReferenceOnlyPipeline.from_pretrained(
        args.pretrained_model_path,
        unet=UNet2DDobuleConditionModel.from_pretrained(
            os.path.join(f"{checkpoint_path}", "unet"),
        ),
        image_encoder=CLIPVisionModel.from_pretrained(
            os.path.join(f"{checkpoint_path}", "image_encoder"),
        ),
    ).to("cuda")
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    for validation_path in os.listdir(args.validation_path):
        validation_path = os.path.join(args.validation_path, validation_path)

        prompt = (
            Image.open(os.path.join(validation_path, "1.png"))
            .convert("RGB")
            .resize((args.resolution, args.resolution))
        )
        blueprint = (
            Image.open(os.path.join(validation_path, "2.png"))
            .convert("RGB")
            .resize((args.resolution, args.resolution))
        )
        blueprint = Image.eval(blueprint, lambda x: 255 - x)
        with torch.no_grad():
            image = pipeline(
                prompt=prompt,
                blueprint=blueprint,
                num_inference_steps=20,
                generator=generator,
                train_image_encoder=True,
            ).images[0]
        image.save(
            os.path.join(
                args.output_dir, f"{checkpoint}-{os.path.basename(validation_path)}.png"
            )
        )

    del pipeline
    torch.cuda.empty_cache()
