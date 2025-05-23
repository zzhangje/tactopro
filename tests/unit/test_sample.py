from tactopro import TactoPro
import pytest


def test_sample_uniformly():
    tp = TactoPro("tests/data/object.stl")
    poses = tp.sample_poses_uniformly(10)
    assert len(poses) == 10, f"Should sample 10 poses, but got {len(poses)}."


def test_sample_trajectory():
    tp = TactoPro("tests/data/object.stl")
    poses = tp.sample_poses_trajectory(10)
    assert len(poses) == 10, f"Should sample 10 poses, but got {len(poses)}."
