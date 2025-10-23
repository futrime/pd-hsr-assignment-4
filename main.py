import os
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray


class Detection(NamedTuple):
    x: int
    y: int
    r: int


def detect_image(image: NDArray[np.uint8]) -> list[Detection]:
    """Your custom detection logic for images.

    Args:
        image: Input image as a NumPy array with shape (H, W, C).

    Returns:
        A list of detections.
    """

    # A simple example: detect footballs with RGB thresholding + minimum bounding rectangle.
    lower_bound = np.array([240, 240, 240])
    upper_bound = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower_bound, upper_bound)
    detections = []
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        radius = (max_x - min_x + max_y - min_y) // 4
        detections.append(Detection(center_x, center_y, radius))
    return detections


def detect_video(video: NDArray[np.uint8]) -> list[list[Detection]]:
    """Your custom detection logic for videos.

    Args:
        video: Input video as a NumPy array with shape (T, H, W, C).

    Returns:
        A list of lists of detections for each frame.
    """

    return [detect_image(frame) for frame in video]


def main() -> None:
    ASSETS_DIR = "./assets/"
    OUTPUTS_DIR = "./outputs/"

    IMAGE_FILE_NAMES = [
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png",
        "6.png",
        "7.png",
        "8.png",
    ]
    VIDEO_FILE_NAMES = [
        "9.mp4",
        "10.mp4",
    ]

    def load_image(path: str) -> NDArray[np.uint8]:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        return image.astype(np.uint8)

    def load_video(path: str) -> tuple[NDArray[np.uint8], float]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames: list[NDArray[np.uint8]] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.astype(np.uint8))
        cap.release()
        return np.stack(frames), fps

    def visualize_detections(
        image: NDArray[np.uint8], detections: list[Detection]
    ) -> NDArray[np.uint8]:
        vis_image = image.copy()
        for det in detections:
            cv2.circle(vis_image, (det.x, det.y), det.r, (0, 0, 255), 2)
        return vis_image

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    for file_name in IMAGE_FILE_NAMES:
        image_path = os.path.join(ASSETS_DIR, file_name)
        image = load_image(image_path)

        detections = detect_image(image)

        output_path = os.path.join(OUTPUTS_DIR, f"detected_{file_name}")
        detection_vis = visualize_detections(image, detections)
        cv2.imwrite(output_path, detection_vis)

    for file_name in VIDEO_FILE_NAMES:
        video_path = os.path.join(ASSETS_DIR, file_name)
        video, fps = load_video(video_path)

        detections = detect_video(video)

        output_path = os.path.join(OUTPUTS_DIR, f"detected_{file_name}")
        video_writer = cv2.VideoWriter(
            filename=output_path,
            fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
            fps=fps,
            frameSize=(video.shape[2], video.shape[1]),
        )
        for frame, frame_detections in zip(video, detections):
            detection_vis = visualize_detections(frame, frame_detections)
            video_writer.write(detection_vis)
        video_writer.release()


if __name__ == "__main__":
    main()
