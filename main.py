import os
import timeit

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
from PIL.Image import Image

import detectors
from detectors import Detection

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


def visualize_detections(image: Image, detections: list[Detection]) -> Image:
    image_copy = image.copy()
    draw = PIL.ImageDraw.Draw(image_copy)
    for det in detections:
        draw.circle((det.x, det.y), det.r, outline="red", width=3)
    return image_copy


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    for file_name in IMAGE_FILE_NAMES:
        image_path = os.path.join(ASSETS_DIR, file_name)
        image = PIL.Image.open(image_path).convert("RGB")

        for detector in detectors.get_image_detectors():
            print()
            print(f"Processing image {file_name} with detector {detector.__name__}")

            try:
                start_time = timeit.default_timer()
                detections = detector(image)
                elapsed_time = timeit.default_timer() - start_time
                print(f"Took {elapsed_time} seconds")
            except NotImplementedError:
                print("Not implemented, skipping...")
                continue

            output_path = os.path.join(OUTPUTS_DIR, f"{detector.__name__}_{file_name}")
            detection_vis = visualize_detections(image, detections)
            detection_vis.save(output_path)

    for file_name in VIDEO_FILE_NAMES:
        video_path = os.path.join(ASSETS_DIR, file_name)
        video = cv2.VideoCapture(video_path)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video.release()

        for detector in detectors.get_video_detectors():
            print()
            print(f"Processing video {file_name} with detector {detector.__name__}")

            try:
                start_time = timeit.default_timer()
                detections = detector(frames)
                elapsed_time = timeit.default_timer() - start_time
                print(f"Took {elapsed_time} seconds")
            except NotImplementedError:
                print("Not implemented, skipping...")
                continue

            output_path = os.path.join(OUTPUTS_DIR, f"{detector.__name__}_{file_name}")
            video_writer = cv2.VideoWriter(
                filename=output_path,
                fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
                fps=fps,
                frameSize=(width, height),
            )
            for frame, frame_detections in zip(frames, detections):
                detection_vis = visualize_detections(frame, frame_detections)
                detection_vis_bgr = cv2.cvtColor(
                    np.array(detection_vis), cv2.COLOR_RGB2BGR
                )
                video_writer.write(detection_vis_bgr)
            video_writer.release()


if __name__ == "__main__":
    main()
