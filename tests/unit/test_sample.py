from tactopro import TactoPro
import os.path as osp
import os
import pickle
import numpy as np


def check_frame(frame):
    assert frame.rgbframe is not None, "RGB frame should be present."
    assert frame.rgbframe.shape == (
        480,
        640,
        3,
    ), "RGB frame should have shape (480, 640, 3)."
    assert frame.heightmap is not None, "Height map should be present."
    assert frame.heightmap.shape == (
        480,
        640,
    ), "Height map should have shape (480, 640)."
    assert frame.contactmask is not None, "Contact mask should be present."
    assert frame.contactmask.shape == (
        480,
        640,
    ), "Contact mask should have shape (480, 640)."
    assert frame.campose is not None, "Camera pose should be present."
    assert frame.campose.shape == (4, 4), "Camera pose should have shape (4, 4)."
    assert frame.gelpose is not None, "Gel pose should be present."
    assert frame.gelpose.shape == (4, 4), "Gel pose should have shape (4, 4)."
    assert frame.rgbframe.dtype == np.uint8, "RGB frame should be of type uint8."
    assert frame.heightmap.dtype == np.float32, "Height map should be of type float32."
    assert frame.contactmask.dtype == np.uint8, "Contact mask should be of type uint8."
    assert frame.campose.dtype == np.float32, "Camera pose should be of type float32."
    assert frame.gelpose.dtype == np.float32, "Gel pose should be of type float32."


def test_sample_uniformly():
    tp = TactoPro("tests/data/digit.STL")
    frames = tp.sample_frames_uniformly(10)
    assert len(frames) == 10, "Should sample 10 frames."
    for frame in frames:
        check_frame(frame)


def test_sample_trajectory():
    tp = TactoPro("tests/data/digit.STL")
    frames = tp.sample_frames_trajectory(10)
    assert len(frames) == 10, "Should sample 10 frames."
    for frame in frames:
        check_frame(frame)
