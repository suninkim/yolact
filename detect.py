# import argparse
# import cProfile
# import json
# import os
# import pickle
# import random
# import time
# from collections import OrderedDict, defaultdict
# from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# import numpy as np
# import pycocotools
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from data import COLORS, cfg, set_cfg

# from data import COLORS, MEANS, COCODetection, cfg, get_label_map, set_cfg, set_dataset
# from layers.box_utils import center_size, jaccard, mask_iou
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer
from utils.augmentations import FastBaseTransform

# from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import SavePath

# from utils.functions import MovingAverage, ProgressBar, SavePath
from yolact import Yolact

# from torch.autograd import Variable


class YolactDetection:
    def __init__(self, config):
        model_path = SavePath.from_str(config["trained_model"])
        config_name = model_path.model_name + "_config"
        self.model = Yolact()
        self.model.load_weights(config["trained_model"])
        self.model.eval()

        set_cfg(config_name)

        # config variable
        self.top_k = config["top_k"]
        self.score_threshold = config["score_threshold"]
        self.display_masks = config["display_masks"]
        self.crop = config["crop"]
        self.display_lincomb = config["display_lincomb"]
        self.display_text = config["display_text"]
        self.display_bboxes = config["display_bboxes"]
        self.display_scores = config["display_scores"]

        if config["device"] == "cuda":
            cudnn.fastest = True
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.model = self.model.cuda()

        self.model.detect.use_fast_nms = config["fast_nms"]
        self.model.detect.use_cross_class_nms = config["cross_class_nms"]

    def get_detection_result(self, image):
        with torch.no_grad():
            frame = torch.from_numpy(image).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.model(batch)

            img_numpy = self.prep_display(
                preds, frame, None, None, undo_transform=False
            )

        return img_numpy

    # return color-masked image, dict{"class": [u,v]}
    def prep_display(
        self,
        dets_out,
        img,
        h,
        w,
        undo_transform=True,
        class_color=False,
        mask_alpha=0.45,
        fps_str="",
    ):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env("Postprocess"):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(
                dets_out,
                w,
                h,
                visualize_lincomb=self.display_lincomb,
                crop_masks=self.crop,
                score_threshold=self.score_threshold,
            )
            cfg.rescore_bbox = save

        with timer.env("Copy"):
            idx = t[1].argsort(0, descending=True)[: self.top_k]

            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(self.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.0
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
        if self.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
            # After this, mask is of size [num_dets, h, w, 1]
            masks = masks[:num_dets_to_consider, :, :, None]

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            colors = torch.cat(
                [
                    get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3)
                    for j in range(num_dets_to_consider)
                ],
                dim=0,
            )
            masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = masks * (-mask_alpha) + 1

            # I did the math for this on pen and paper. This whole block should be equivalent to:
            #    for j in range(num_dets_to_consider):
            #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
            masks_color_summand = masks_color[0]
            if num_dets_to_consider > 1:
                inv_alph_cumul = inv_alph_masks[: (num_dets_to_consider - 1)].cumprod(
                    dim=0
                )
                masks_color_cumul = masks_color[1:] * inv_alph_cumul
                masks_color_summand += masks_color_cumul.sum(dim=0)

            img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()

        if num_dets_to_consider == 0:
            return img_numpy

        if self.display_text or self.display_bboxes:
            for j in reversed(range(num_dets_to_consider)):
                x1, y1, x2, y2 = boxes[j, :]
                color = get_color(j)
                score = scores[j]

                if self.display_bboxes:
                    cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                if self.display_text:
                    _class = cfg.dataset.class_names[classes[j]]
                    text_str = (
                        "%s: %.2f" % (_class, score) if self.display_scores else _class
                    )

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(
                        text_str, font_face, font_scale, font_thickness
                    )[0]

                    text_pt = (x1, y1 - 3)
                    text_color = [255, 255, 255]

                    cv2.rectangle(
                        img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1
                    )
                    cv2.putText(
                        img_numpy,
                        text_str,
                        text_pt,
                        font_face,
                        font_scale,
                        text_color,
                        font_thickness,
                        cv2.LINE_AA,
                    )

        return img_numpy


if __name__ == "__main__":
    detection_cfg_path = f"detection.yaml"
    import yaml

    with open(detection_cfg_path, "r") as stream:
        task_cfg = yaml.safe_load(stream)
    detection_system = Yolact(task_cfg)

    camera_num = 0
    cap = cv2.VideoCapture(camera_num)
    if not cap.isOpened():
        print("Failed to open camera.")

    while True:
        ret, frame = cap.read()
        detected_image = detection_system.get_detection_result(frame)
        plt.show(detected_image)
        plt.waitkeys()

    image = plt.imread("../test.png")
    detected_image = detection_system.get_detection_result(image)
    plt.show(detected_image)
    plt.waitkeys()
