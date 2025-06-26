import numpy as np
import os
import pickle
from PIL import Image
from typing import List
from tactopro import TactoFrame
from ..helpers.pose import quat_to_SE3
from tactopro.renderer import Renderer


__available_dir__ = [
    "004_sugar_box/",
    "005_tomato_soup_can/",
    "006_mustard_bottle/",
    "021_bleach_cleanser/",
    "025_mug/",
    "035_power_drill/",
    "037_scissors/",
    "042_adjustable_wrench/",
    "048_hammer/",
    "055_baseball/",
]


def load_ycb_slide_frame(file_path: str, idx: int) -> TactoFrame:
    """
    Loads a single TactoFrame from a YCB slide dataset.
    Args:
        file_path (str): The path to the directory containing the YCB slide data.
        idx (int): The index of the frame to load.
    Returns:
        TactoFrame: A TactoFrame object loaded from the specified index in the YCB slide dataset.
    """
    c_path = os.path.join(file_path, "gt_contactmasks", f"{idx}.jpg")
    h_path = os.path.join(file_path, "gt_heightmaps", f"{idx}.jpg")
    i_path = os.path.join(file_path, "tactile_images", f"{idx}.jpg")
    c = np.array(Image.open(c_path)).astype(bool)
    h = np.array(Image.open(h_path)).astype(np.int64)
    i = np.array(Image.open(i_path)).astype(np.uint8)

    with open(os.path.join(file_path, "tactile_data.pkl"), "rb") as f:
        data = pickle.load(f)
        cam_pose = quat_to_SE3(data["camposes"][idx])
        gel_pose = quat_to_SE3(data["gelposes"][idx])
        f.close()

    return TactoFrame(
        rgbframe=i,
        heightmap=h,
        contactmask=c,
        pointcloud=Renderer.get_ycbslide_renderer().heightmap_to_pointcloud(h),
        campose=cam_pose,
        gelpose=gel_pose,
    )


def load_ycb_slide_path(file_path: str) -> List[TactoFrame]:
    """
    Loads TactoFrames from a directory containing YCB slide data.
    Args:
        file_path (str): The path to the directory containing the YCB slide data.
    Returns:
        List[TactoFrame]: A list of TactoFrame objects loaded from the specified directory.
    """
    with open(os.path.join(file_path, "tactile_data.pkl"), "rb") as f:
        data = pickle.load(f)
        cam_pose = quat_to_SE3(data["camposes"])
        gel_pose = quat_to_SE3(data["gelposes"])
        f.close()

    frames = []
    idx = 0
    while True:
        try:
            c_path = os.path.join(file_path, "gt_contactmasks", f"{idx}.jpg")
            h_path = os.path.join(file_path, "gt_heightmaps", f"{idx}.jpg")
            i_path = os.path.join(file_path, "tactile_images", f"{idx}.jpg")
            c = np.array(Image.open(c_path)).astype(bool)
            h = np.array(Image.open(h_path)).astype(np.int64)
            i = np.array(Image.open(i_path)).astype(np.uint8)
            frame = TactoFrame(
                rgbframe=i,
                heightmap=h,
                contactmask=c,
                pointcloud=Renderer.get_ycbslide_renderer().heightmap_to_pointcloud(h)[
                    c.reshape(-1)
                ],
                campose=cam_pose[idx],
                gelpose=gel_pose[idx],
            )
            frames.append(frame)
            idx += 1
            if idx % 100 == 0:
                print(f"Loaded {idx} frames from {file_path}")
        except FileNotFoundError:
            print("No file found, returning...")
            break
    return frames
