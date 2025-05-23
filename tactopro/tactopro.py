from dataclasses import dataclass
import numpy as np
import trimesh
from typing import List
import os.path as osp
import os
import pickle
import cv2
from PIL import Image
from .renderer import Renderer
from .vizutils import viz_poses_pointclouds_on_mesh


@dataclass
class TactoFrame:
    """
    Represents a single frame captured by the Tacto sensor, containing image and pose data.
    Attributes:
        rgb_frame (np.ndarray): The RGB image captured by the sensor.
        height_map (np.ndarray): The height map representing surface deformations, units in pixels.
        contact_mask (np.ndarray): A binary mask indicating contact regions.
        camera_pose (np.ndarray): The 4x4 matrix representing the pose of the camera at the time of capture.
        gel_pose (np.ndarray): The 4x4 matrix representing the pose of the gel pad at the time of capture.
    """

    rgbframe: np.ndarray
    heightmap: np.ndarray
    contactmask: np.ndarray
    campose: np.ndarray
    gelpose: np.ndarray


class TactoPro:
    def __init__(self, trimesh_path: str):
        self._trimesh = trimesh.load_mesh(trimesh_path)
        self._renderer = Renderer()

    @property
    def trimesh(self) -> trimesh.Trimesh:
        """
        Returns the loaded trimesh object.
        """
        return self._trimesh

    def sample_frames_uniformly(self, num_samples: int = 1000) -> List[TactoFrame]:
        """
        Randomly samples frames from the mesh.
        Args:
            num_samples (int): The number of points to sample.
        Returns:
            List[TactoFrame]: A list of randomly sampled TactoFrame objects.
        """
        sampled_points = self._trimesh.sample(num_samples)
        return [
            TactoFrame(
                rgbframe=point,
                heightmap=None,
                contactmask=None,
                campose=None,
                gelpose=None,
            )
            for point in sampled_points
        ]

    def sample_frames_trajectory(self, num_samples: int = 1000) -> List[TactoFrame]:
        """
        Samples frames along a trajectory on the mesh.
        Args:
            num_samples (int): The number of points to sample.
        Returns:
            List[TactoFrame]: A list of sampled TactoFrame objects along the trajectory.
        """
        # Placeholder for trajectory sampling logic
        sampled_points = self._trimesh.sample(num_samples)
        return [
            TactoFrame(
                rgbframe=point,
                heightmap=None,
                contactmask=None,
                campose=None,
                gelpose=None,
            )
            for point in sampled_points
        ]

    def sample_frames_manually(self, poses: List[np.ndarray]) -> List[TactoFrame]:
        """
        Samples frames based on manually specified poses.
        Args:
            poses (List[np.ndarray]): A list of 4x4 matrices representing the poses.
        Returns:
            List[TactoFrame]: A list of TactoFrame objects sampled at the specified poses.
        """
        sampled_points = [self._trimesh.sample(1, pose=pose) for pose in poses]
        return [
            TactoFrame(
                rgbframe=point,
                heightmap=None,
                contactmask=None,
                campose=pose,
                gelpose=None,
            )
            for point, pose in zip(sampled_points, poses)
        ]

    def save(self, frames: List[TactoFrame], save_path: str, headless: bool = False):
        """
        Saves the sampled frames to a specified path.
        Args:
            save_path (str): The path to save the frames.
            headless (bool): Whether to run in headless mode (no GUI).
        """
        assert isinstance(frames, list), "must be a list of TactoFrame objects"
        for frame in frames:
            assert isinstance(frame, TactoFrame), "must be a list of TactoFrame objects"

        base_path = save_path
        count = 1
        while osp.exists(save_path):
            save_path = f"{base_path}_{count}"
            count += 1
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        print(f"Saving data to {save_path}")

        rgbframe_path = osp.join(save_path, "rgbframes")
        heightmap_path = osp.join(save_path, "heightmaps")
        contactmask_path = osp.join(save_path, "contactmasks")
        pose_path = osp.join(save_path, "poses.pkl")

        os.makedirs(rgbframe_path)
        os.makedirs(heightmap_path)
        os.makedirs(contactmask_path)

        with open(pose_path, "wb") as f:
            pickle.dump(
                {
                    "camposes": [frame.campose for frame in frames],
                    "gelposes": [frame.gelpose for frame in frames],
                },
                f,
            )

        for i, frame in enumerate(frames):
            cv2.imwrite(
                osp.join(rgbframe_path, f"{i}.png"),
                Image.fromarray(frame.rgbframe.astype(np.uint8), "RGB"),
            )
            cv2.imwrite(
                osp.join(heightmap_path, f"{i}.png"), frame.heightmap.astype(np.float32)
            )
            cv2.imwrite(
                osp.join(contactmask_path, f"{i}.png"),
                255 * frame.contactmask.astype(np.uint8),
            )

        if not headless:
            # prepare point cloud
            pc_all = []
            for i, frame in enumerate(frames[::10]):
                pc_body = self._renderer.heightmap_to_pointcloud(frame.heightmap)
                R = frame.gelpose[:3, :3]
                t = frame.gelpose[:3, 3]
                pc_world = (R @ pc_body.T).T + t
                pc_all.append(pc_world)

            # visualize point cloud
            illustration_path = osp.join(save_path, "illustration.png")
            viz_poses_pointclouds_on_mesh(
                self._trimesh,
                [frame.campose for frame in frames[::10]],
                pc_all,
                save_path=illustration_path,
            )

        pass

    pass
