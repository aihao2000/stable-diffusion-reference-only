---
title: Anime Fanwork
emoji: ✏️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 3.48.0
app_file: app.py
pinned: true
license: apache-2.0
models:
- AisingioroHao0/stable-diffusion-reference-only-automatic-coloring-0.1.2
---

# Stable Diffusion Reference Only

A general model for secondary painting.[![arXiv](https://camo.githubusercontent.com/9f865a7810c4ab8e0c379528b0cbf222348b44db84823c091f49d8a992462d2d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f61725869762d323330362e31343638352d6233316231622e737667)](https://arxiv.org/abs/2311.02343)

![](./README.assets/9.png)

![](./README.assets/3.png)

![](./README.assets/3x9.png)

## Online Demo

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try it out  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AisingioroHao0/anime-fanwork)

Try it On [![**Stable Diffusion Reference Only Demo**](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aLKbm8YAXsGP2Y3umx0letLIY_rhl8VL?usp=sharing)

## Models

[Stable Diffusion Reference Only Automatic Coloring 0.1.2](https://huggingface.co/AisingioroHao0/stable-diffusion-reference-only-automatic-coloring-0.1.2) is released. Trained for 850,000 steps on anime images at 512 resolution.

*If you are interested in better stable diffusion reference only models and have idle computing resources, feel free to contact me. Due to copyright issues with the dataset images, I'm not sure if I can make them public.*

## Instructions for Use

It is recommended to create and use a torch virtual environment, such as conda.

- Install conda

  ```shell
  wget https://mirrors.ustc.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash  Miniconda3-latest-Linux-x86_64.sh
  ```

- Create a virtual environment

  It is compatible with general torch environments. If you have a torch environment, you can try using it directly.

  ```shell
  conda create -n stable_diffusion_reference_only 
  conda activate stable_diffusion_reference_only python=3.10 
  ```

### By Package

Make sure the torch environment is normal, such as installing through conda:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

The latest diffusers (0.22.3) currently do not support loading pre-trained models directly from huggingfacehub. You may need to fix the diffusers version in advance, e.g. use 0.21.0

```python
pip install diffusers==0.21.0
```

install this package:

```shell
pip install git+https://github.com/aihao2000/stable-diffusion-reference-only.git
```

You may need to clear the background, you can install:

```shell
pip install git+https://github.com/aihao2000/anime-segmentation.git
```

For usage examples, please refer to https://colab.research.google.com/drive/1aLKbm8YAXsGP2Y3umx0letLIY_rhl8VL?usp=sharing#scrollTo=p-a9SsZg3O93.

### By Web-UI

A1111 stable diffusion Web UI is currently not supported. I will support it as soon as possible. If you want to use web UI, you can deploy a version similar to huggingface space locally.

Clone this repository:

```shell
git clone git@github.com:aihao2000/stable-diffusion-reference-only.git
cd stable-diffusion-reference-only
```

Install dependencies:

```shell
pip install -r requirements.txt
```

Run Web-UI:

```shell
python3 app.py
```

Next, let’s use web ui for secondary creation.

- Upload the colored character reference image to ```prompt```. If it contains a background and you do not want the background to be colored, you can click ```character segment``` to clear to enhance the effect.
- Upload the target character line artto ```blueprint```.
- If you expect to perform a style transfer task, you may not have a line art of the target character. You can upload a color drawing of the target character, click ```character segment``` to clear the background (optional), then click ```get line art``` to obtain the line art of the character, and finally click ```color Inversion``` gets blueprint with black background and white lines.
- If you are a professional artist doing secondary creation, you can directly upload the drawn line art to ```blueprint```, and you can still try to click ```get line art``` (recommended). It is slightly different from the line art drawn manually. The result is slightly different. Finally, click ```color inversion``` to get the final blueprint with black background and white lines.
- Finally, click ```inference``` to generate a secondary creation diagram of the target character. According to tests, ```num_inference_steps``` of 20 can achieve good results. You can still try other parameters.
- For the ```style transfer``` task, you can directly upload the color images of the two characters, and then click ```style transfer```, which will automatically execute ```character segment, get line art```, ```color inversion```, ```inference```.
- For line art ```automatic coloring``` tasks, you can directly upload the reference character image to the manual line art, and then click ```automatic coloring```, which will automatically perform ```color inversion``` and ```inference```.

#### Optional: Manually download switching models

The required model will be automatically downloaded (line art automatic coloring) from huggingfacehub. You can also download it manually at https://huggingface.co/AisingioroHao0/stable-diffusion-reference-only-automatic-coloring-0.1.2.

Then change

```python
automatic_coloring_pipeline = StableDiffusionReferenceOnlyPipeline.from_pretrained(

  "AisingioroHao0/stable-diffusion-reference-only-automatic-coloring-0.1.2"

).to(device)
```

 to

```python
automatic_coloring_pipeline = StableDiffusionReferenceOnlyPipeline.from_pretrained(

  "local_model_path"

).to(device)
```

## More Examples

| input                              | input                                 | output                             |
| ---------------------------------- | ------------------------------------- | ---------------------------------- |
| ![](./README.assets/1_prompt.png)  | ![](./README.assets/1_blueprint.png)  | ![](./README.assets/1_result.png)  |
| ![](./README.assets/12_prompt.png) | ![](./README.assets/12_blueprint.png) | ![](./README.assets/12_result.png) |
| ![](./README.assets/13_prompt.png) | ![](./README.assets/13_blueprint.png) | ![](./README.assets/13_result.png) |
| ![](./README.assets/8_prompt.png)  | ![](./README.assets/8_blueprint.png)  | ![](./README.assets/8_result.png)  |
| ![](./README.assets/11_prompt.png) | ![](./README.assets/11_blueprint.png) | ![](./README.assets/11_result.png) |
| ![](./README.assets/15_prompt.png) | ![](./README.assets/15_blueprint.png) | ![](./README.assets/15_result.png) |





