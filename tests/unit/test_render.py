from tactopro import TactoPro
import numpy as np


def check_frame(frame, config):
    assert frame.rgbframe is not None, "RGB frame should be present."
    assert frame.rgbframe.shape == (
        config.height,
        config.width,
        3,
    ), f"RGB frame should have shape ({config.height}, {config.width}, 3), but got {frame.rgbframe.shape}."
    assert (
        frame.rgbframe.dtype == np.uint8
    ), f"RGB frame should be of type uint8, but got {frame.rgbframe.dtype}."

    assert frame.heightmap is not None, "Height map should be present."
    assert frame.heightmap.shape == (
        config.height,
        config.width,
    ), f"Height map should have shape ({config.height}, {config.width}), but got {frame.heightmap.shape}."
    assert (
        frame.heightmap.dtype == np.float32
    ), f"Height map should be of type float32, but got {frame.heightmap.dtype}."

    assert frame.contactmask is not None, "Contact mask should be present."
    assert frame.contactmask.shape == (
        config.height,
        config.width,
    ), f"Contact mask should have shape ({config.height}, {config.width}), but got {frame.contactmask.shape}."
    assert (
        frame.contactmask.dtype == bool
    ), f"Contact mask should be of type bool, but got {frame.contactmask.dtype}."

    assert frame.pointcloud is not None, "Point cloud should be present."
    assert frame.pointcloud.shape == (
        config.height * config.width,
        3,
    ), f"Point cloud should have shape ({frame.contactmask.sum()}, 3), but got {frame.pointcloud.shape}."

    assert frame.campose is not None, "Camera pose should be present."
    assert frame.campose.shape == (
        4,
        4,
    ), f"Camera pose should have shape (4, 4), but got {frame.campose.shape}."

    assert frame.gelpose is not None, "Gel pose should be present."
    assert frame.gelpose.shape == (
        4,
        4,
    ), f"Gel pose should have shape (4, 4), but got {frame.gelpose.shape}."


def test_render(config):
    tp = TactoPro("tests/data/object.stl", config=config)
    poses = tp.sample_poses_uniformly(10)
    frames = tp.get_frames_from_poses(poses)
    for frame in frames:
        check_frame(frame, config)

    pass
