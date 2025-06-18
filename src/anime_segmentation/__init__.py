# Modify from https://github.com/SkyTNT/anime-segmentation.
from .inference import get_mask
from .train import AnimeSegmentation
import numpy as np
import torch


def get_model(
    device="cpu",
    model_path="~/workspace/DeepLearningContent/models/anime-seg/isnetis.ckpt",
    img_size=1024,
):
    model = AnimeSegmentation.try_load(
        "isnet_is",
        model_path,
        device,
        img_size=img_size,
    )
    model.eval()
    return model


@torch.no_grad()
def character_segment(model, img, img_size=1024, use_amp=False):
    """get character seg image

    Args:
        model (AnimeSegmentation): from get_model
        img (cv2 numpy array): RGB model
        img_size (int, optional): _description_. Defaults to 1024.
        use_amp (bool, optional): _description_. Defaults to False.

    Returns:
        numpy array: RGBA
    """
    mask = get_mask(model, img, use_amp=use_amp, s=img_size)
    img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
    return img
