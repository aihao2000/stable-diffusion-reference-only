import os

import argparse
import cv2
import torch
import numpy as np
import glob
import math
from torch.cuda import amp
from tqdm import tqdm
import sys

from anime_segmentation.train import AnimeSegmentation, net_names


def get_mask(model, input_img, use_amp=True, s=640):
    input_img = (input_img / 255).astype(np.float32)
    h, w = h0, w0 = input_img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w] = cv2.resize(
        input_img, (w, h)
    )
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
        return pred

def is_empty(img):
    img_line = cv2.adaptiveThreshold(
        cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=5,
        C=7,
    )
    if np.all(img_line==255):
        return True
    else:
        return False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument(
        "--net", type=str, default="isnet_is", choices=net_names, help="net name"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="../../models/anime-seg/isnetis.ckpt",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=".",
        help="input data dir",
    )
    parser.add_argument("--out", type=str, default="../characters", help="output dir")
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="hyperparameter, input image size of the net",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)

    model = AnimeSegmentation.try_load(
        args.net, args.ckpt, args.device, img_size=args.img_size
    )
    model.eval()
    model.to(device)
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    image_paths = glob.glob(f"{args.data}/**/*.png", recursive=True) + glob.glob(
        f"{args.data}/**/*.jpg", recursive=True
    )
    image_paths = [
        image_path
        for image_path in image_paths
        if not os.path.exists(
            os.path.join(
                args.out,
                os.path.relpath(os.path.splitext(image_path)[0] + ".png", args.data),
            )
        )
    ]
    print(len(image_paths))
    character_empty_list = []
    for path in tqdm(image_paths):
        try:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            mask = get_mask(
                model,
                img,
                use_amp=False,
                s=1024 if abs(1024-max(img.shape[0], img.shape[1])) <= abs(512-max(img.shape[0], img.shape[1])) else 512,
            )
            img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(
                np.uint8
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            if is_empty(img):
                print(path + " hasn't character")
                character_empty_list.append(path)
                continue
            file_name, file_extension = os.path.splitext(path)
            save_path = os.path.join(
                args.out, os.path.relpath(file_name + ".png", args.data)
            )
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            cv2.imwrite(save_path, img)
        except Exception as e:
            print(path + ":" + str(e))
    with open("character_empty_list.txt", "w") as f:
        f.writelines("\n".join(character_empty_list))
