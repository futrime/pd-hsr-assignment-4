from collections.abc import Callable
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torchvision.models.detection
import torchvision.transforms.functional
import tqdm
from PIL.Image import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class Detection(NamedTuple):
    x: int
    y: int
    r: int


_image_detectors: list[Callable[[Image], list[Detection]]] = []
_video_detectors: list[Callable[[list[Image]], list[list[Detection]]]] = []


def get_image_detectors() -> list[Callable[[Image], list[Detection]]]:
    return _image_detectors


def get_video_detectors() -> list[Callable[[list[Image]], list[list[Detection]]]]:
    return _video_detectors


def register_image_detector(
    detector: Callable[[Image], list[Detection]],
) -> Callable[[Image], list[Detection]]:
    _image_detectors.append(detector)
    return detector


def register_video_detector(
    detector: Callable[[list[Image]], list[list[Detection]]],
) -> Callable[[list[Image]], list[list[Detection]]]:
    _video_detectors.append(detector)
    return detector


# ============================================================
# YOUR CUSTOM DETECTION LOGIC BELOW
# ============================================================


@register_image_detector
def dummy_image_detector(image: Image) -> list[Detection]:
    """A dummy detection logic for images. You can copy and modify this function.

    Args:
        image: Input image as a PIL Image.

    Returns:
        A list of detections.
    """

    raise NotImplementedError


@register_video_detector
def dummy_video_detector(frames: list[Image]) -> list[list[Detection]]:
    """A dummy detection logic for videos. You can copy and modify this function.

    Args:
        frames: List of frames as PIL Images.

    Returns:
        A list of detections.
    """

    raise NotImplementedError


@register_image_detector
def rgb_mbr_image_detector(image: Image) -> list[Detection]:
    """An example image detector using RGB thresholding and minimum bounding rectangle."""

    image_array = np.array(image)
    mask = cv2.inRange(
        image_array, np.array([240, 240, 240]), np.array([255, 255, 255])
    )
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    return [
        Detection(
            x=int(xs.mean()),
            y=int(ys.mean()),
            r=int((xs.max() - xs.min() + ys.max() - ys.min()) / 4),
        )
    ]


faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
faster_rcnn_model.eval()


@register_image_detector
def faster_rcnn_image_detector(image: Image) -> list[Detection]:
    """An example image detector using Faster R-CNN object detection model."""

    preprocess = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()

    batch = [preprocess(torchvision.transforms.functional.pil_to_tensor(image))]

    with torch.inference_mode():
        prediction = faster_rcnn_model(batch)[0]

    boxes = prediction["boxes"][
        prediction["labels"]
        == FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"].index(
            "sports ball"
        )
    ]

    detections = [
        Detection(
            x=int((box[0] + box[2]) / 2),
            y=int((box[1] + box[3]) / 2),
            r=int((box[2] - box[0] + box[3] - box[1]) / 4),
        )
        for box in boxes
    ]

    return detections


@register_video_detector
def faster_rcnn_video_detector(frames: list[Image]) -> list[list[Detection]]:
    """An example video detector using Faster R-CNN object detection model."""

    return [faster_rcnn_image_detector(frame) for frame in tqdm.tqdm(frames)]
