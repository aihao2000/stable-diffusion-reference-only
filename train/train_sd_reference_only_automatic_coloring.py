#!/usr/bin/env python
# coding=utf-8

import argparse
import enum
import logging
import math
import os
import shutil
from pathlib import Path
import itertools
import accelerate
from accelerate import DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel
from transformers.utils import ContextManagers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import cv2
import sys

sys.path.append("src")
from stable_diffusion_reference_only.pipelines.stable_diffusion_reference_only_pipeline import (
    StableDiffusionReferenceOnlyPipeline,
)
from stable_diffusion_reference_only.models.unet_2d_dobule_condition import (
    UNet2DDobuleConditionModel,
)

if is_wandb_available():
    import wandb


logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


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
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)
    if args.train_image_encoder:
        image_encoder = accelerator.unwrap_model(image_encoder)

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

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i, log in enumerate(image_logs):
                images = log["images"]
                prompt = log["prompt"]
                blueprint = log["blueprint"]

                formatted_images = []

                formatted_images.append(np.asarray(prompt))
                formatted_images.append(np.asarray(blueprint))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    str(i), formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                prompt = log["prompt"]
                blueprint = log["blueprint"]

                formatted_images.append(wandb.Image(prompt, caption="prompt"))
                formatted_images.append(wandb.Image(blueprint, caption="blueprint"))
                for image in images:
                    image = wandb.Image(image, caption="result")
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    for i, log in enumerate(image_logs):
        for j, image in enumerate(log["images"]):
            image.save(
                os.path.join(args.output_dir, "validations", f"{step}_{i}_{j}.png")
            )

    del pipeline
    torch.cuda.empty_cache()

    return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_blueprint = log["validation_blueprint"]
            validation_blueprint.save(os.path.join(repo_folder, "blueprint.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_blueprint] + images
            image_grid(images, 1, len(images)).save(
                os.path.join(repo_folder, f"images_{i}.png")
            )
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- unet
inference: true
---
    """
    model_card = f"""
# unet-{repo_id}

These are unet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a UNet2DDobuleConditionModel training script."
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stable-diffusion-reference-only-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_image_encoder",
        action="store_true",
        help="Whether to train the image encoder. If set, the image encoder should be float32 precision.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument("--load_dataset_num_proc", type=int, default=None)
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--load_dataset_streaming",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dataset_map",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dataset_map_batch_size",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--dataset_map_writer_batch_size",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="image1",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image2",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_blueprint`s, a single `--validation_blueprint`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_blueprint`s."
        ),
    )
    parser.add_argument(
        "--validation_blueprint",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_blueprint`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_blueprints`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_stable_diffusion_reference_only",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_blueprint is None:
        raise ValueError(
            "`--validation_blueprint` must be set if `--validation_prompt` is set"
        )

    if args.validation_prompt is None and args.validation_blueprint is not None:
        raise ValueError(
            "`--validation_prompt` must be set if `--validation_blueprint` is set"
        )

    if (
        args.validation_blueprint is not None
        and args.validation_prompt is not None
        and len(args.validation_blueprint) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_blueprint) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_blueprint`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_blueprint`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE ."
        )

    return args


def make_train_dataset(args, clip_image_processor, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
            streaming=args.load_dataset_streaming,
            num_proc=args.load_dataset_num_proc,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.prompt_column is None:
        prompt_column = column_names[1]
        logger.info(f"caption column defaulting to {prompt_column}")
    else:
        prompt_column = args.prompt_column
        if prompt_column not in column_names:
            raise ValueError(
                f"`--prompt_column` value '{args.prompt_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    blueprint_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        blueprints = [blueprint.convert("RGB") for blueprint in examples[image_column]]
        blueprints = [
            cv2.cvtColor(np.array(blueprint), cv2.COLOR_RGB2GRAY)
            for blueprint in blueprints
        ]
        blueprints = [
            cv2.adaptiveThreshold(
                blueprint,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                blockSize=5,
                C=7,
            )
            for blueprint in blueprints
        ]
        blueprints = [
            Image.fromarray(blueprint).convert("RGB") for blueprint in blueprints
        ]
        blueprints = [
            Image.eval(blueprint, lambda x: 255 - x) for blueprint in blueprints
        ]
        blueprints = [blueprint_transforms(blueprint) for blueprint in blueprints]
        prompts = [
            clip_image_processor(prompt, return_tensors="pt").pixel_values[0]
            for prompt in examples[prompt_column]
        ]

        return {
            "pixel_values": images,
            "blueprint_pixel_values": blueprints,
            "prompt_pixel_values": prompts,
        }

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=args.seed)
                .select(range(args.max_train_samples))
            )
        # Set the training transforms
        if args.load_dataset_streaming:
            train_dataset = dataset["train"].map(
                preprocess_train,
                batched=True,
            )
            train_dataset = train_dataset.shuffle(seed=args.seed)
        else:
            if args.dataset_map:
                train_dataset = dataset["train"].map(
                    preprocess_train,
                    batch_size=args.dataset_map_batch_size,
                    batched=True,
                    num_proc=args.load_dataset_num_proc,
                    writer_batch_size=args.dataset_map_writer_batch_size,
                )
            else:
                train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack(
            [
                torch.tensor(example["pixel_values"])
                if isinstance(example["pixel_values"], list)
                else example["pixel_values"]
                for example in examples
            ]
        )
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        blueprint_pixel_values = torch.stack(
            [
                torch.tensor(example["blueprint_pixel_values"])
                if isinstance(example["blueprint_pixel_values"], list)
                else example["blueprint_pixel_values"]
                for example in examples
            ]
        )
        blueprint_pixel_values = blueprint_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        prompt_pixel_values = torch.stack(
            [
                torch.tensor(example["prompt_pixel_values"])
                if isinstance(example["prompt_pixel_values"], list)
                else example["prompt_pixel_values"]
                for example in examples
            ]
        )
        prompt_pixel_values = prompt_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        return {
            "pixel_values": pixel_values,
            "blueprint_pixel_values": blueprint_pixel_values,
            "prompt_pixel_values": prompt_pixel_values,
        }

    if args.load_dataset_streaming:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    return train_dataset, train_dataloader


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    if args.ddp_find_unused_parameters:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if not args.train_image_encoder:
            image_encoder = CLIPVisionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="image_encoder",
            )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="clip_image_processor",
    )
    if args.train_image_encoder:
        image_encoder = CLIPVisionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="image_encoder",
        )
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    unet = UNet2DDobuleConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        for model in models:
            sub_dir = (
                "unet"
                if isinstance(model, type(accelerator.unwrap_model(unet)))
                else "image_encoder"
            )
            model.save_pretrained(os.path.join(output_dir, sub_dir))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(image_encoder))):
                # load transformers style into model
                load_model = CLIPVisionModel.from_pretrained(
                    input_dir, subfolder="image_encoder"
                )
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DDobuleConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(True)
    if args.train_image_encoder:
        image_encoder.requires_grad_(True)
    else:
        image_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        itertools.chain(unet.parameters(), image_encoder.parameters())
        if args.train_image_encoder
        else unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset, train_dataloader = make_train_dataset(
        args, clip_image_processor, accelerator
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    if args.train_image_encoder:
        (
            unet,
            image_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, image_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_image_encoder:
        image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_blueprint")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  train image encoder={args.train_image_encoder}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )
            resume_progress_bar = tqdm(
                range(0, resume_step), disable=not accelerator.is_local_main_process
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_image_encoder:
            image_encoder.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    resume_progress_bar.update(1)
                continue
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = image_encoder(
                    batch["prompt_pixel_values"], output_hidden_states=False
                ).last_hidden_state

                blueprint = batch["blueprint_pixel_values"].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    controlnet_cond=blueprint,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), image_encoder.parameters())
                        if args.train_image_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if (
                        args.validation_prompt is not None
                        and global_step % args.validation_steps == 0
                    ):
                        image_logs = log_validation(
                            vae,
                            image_encoder,
                            clip_image_processor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.train_image_encoder:
            image_encoder = accelerator.unwrap_model(image_encoder)
        pipeline = StableDiffusionReferenceOnlyPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            image_encoder=image_encoder,
            clip_image_processor=clip_image_processor,
            unet=unet,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
