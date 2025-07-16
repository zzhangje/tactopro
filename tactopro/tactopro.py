from dataclasses import dataclass
import numpy as np
import trimesh
from typing import List
import os.path as osp
import os
import pickle
import cv2
from .renderer import Renderer, RendererConfig
from .helpers.viz import viz_poses_pointclouds_on_mesh
from .helpers.mesh import sample_poses_on_mesh, random_geodesic_poses
from tqdm import tqdm
from PIL import Image
from .helpers.pose import quat_to_SE3
import open3d as o3d


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
    pointcloud: np.ndarray
    campose: np.ndarray
    gelpose: np.ndarray

    def get_world_pcd(self) -> np.ndarray:
        """
        Returns the point cloud in world coordinates.
        """
        Rwb = self.campose[:3, :3]
        twb = self.campose[:3, 3]
        return (Rwb @ self.pointcloud.T).T + twb


class TactoPro:
    def __init__(self, trimesh_path: str, config: RendererConfig = RendererConfig()):
        self._trimesh = trimesh.load_mesh(trimesh_path)
        self._renderer = Renderer(config=config, trimesh_path=trimesh_path)
        self._config = config

    @property
    def trimesh(self) -> trimesh.Trimesh:
        """
        Returns the loaded trimesh object.
        """
        return self._trimesh

    def sample_poses_uniformly(
        self, num_samples: int = 1000, shear_mag: float = 5.0, edges: bool = False
    ) -> np.ndarray:
        """
        Randomly samples poses from the mesh.
        Args:
            num_samples (int): The number of points to sample.
            shear_mag (float): The magnitude of shear to apply to the sampled points.
            edges (bool): Whether to sample from the edges of the mesh.
        Returns:
            np.ndarray: A numpy array of sampled poses.
        """
        return sample_poses_on_mesh(self._trimesh, num_samples, edges)

    def sample_poses_trajectory(
        self, num_samples: int = 1000, traj_length=0.5, shear_mag: float = 5.0
    ) -> np.ndarray:
        """
        Samples poses along a trajectory on the mesh.
        Args:
            num_samples (int): The number of points to sample, the result may be less than this due to mesh topology.
            traj_length (float): The length of the trajectory.
            shear_mag (float): The magnitude of shear to apply to the sampled points.
        Returns:
            np.ndarray: A numpy array of sampled poses along the trajectory.
        """
        poses = random_geodesic_poses(
            self._trimesh, total_length=traj_length, N=num_samples
        )
        if poses is None:
            return np.empty((0, 4, 4))
        return poses

    def sample_frames_uniformly(
        self, num_samples: int = 1000, shear_mag: float = 5.0, edges: bool = False
    ) -> List[TactoFrame]:
        """
        Randomly samples frames from the mesh.
        Args:
            num_samples (int): The number of points to sample.
            shear_mag (float): The magnitude of shear to apply to the sampled points.
            edges (bool): Whether to sample from the edges of the mesh.
        Returns:
            List[TactoFrame]: A list of randomly sampled TactoFrame objects.
        """
        return self.get_frames_from_poses(
            self.sample_poses_uniformly(num_samples, shear_mag, edges)
        )

    def sample_frames_trajectory(
        self, num_samples: int = 1000, traj_length=0.5, shear_mag: float = 5.0
    ) -> List[TactoFrame]:
        """
        Samples frames along a trajectory on the mesh.
        Args:
            num_samples (int): The number of points to sample, the result may be less than this due to mesh topology.
            traj_length (float): The length of the trajectory.
            shear_mag (float): The magnitude of shear to apply to the sampled points.
        Returns:
            List[TactoFrame]: A list of sampled TactoFrame objects along the trajectory.
        """
        return self.get_frames_from_poses(
            self.sample_poses_trajectory(num_samples, traj_length, shear_mag)
        )

    # def sample_frames_manually(self) -> List[TactoFrame]:
    #     """
    #     Samples frames based on manually specified poses.
    #     Args:
    #         poses (List[np.ndarray]): A list of 4x4 matrices representing the poses.
    #     Returns:
    #         List[TactoFrame]: A list of TactoFrame objects sampled at the specified poses.
    #     """
    #     sampled_poses = []
    #     return self._get_frames_from_poses(sampled_poses)

    def save(
        self, frames: List[TactoFrame], save_path: str, ycb_slide_type: bool = False
    ):
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
        os.makedirs(save_path, exist_ok=True)
        print(
            f"Saving data to {save_path}, create {'successfully' if osp.exists(save_path) else 'failed'}"
        )

        if ycb_slide_type:
            rgbframe_path = osp.join(save_path, "rgbframes")
            heightmap_path = osp.join(save_path, "heightmaps")
            contactmask_path = osp.join(save_path, "contactmasks")
            pointcloud_path = osp.join(save_path, "pointclouds")
            pose_path = osp.join(save_path, "poses.pkl")

            os.makedirs(rgbframe_path)
            os.makedirs(heightmap_path)
            os.makedirs(contactmask_path)
            os.makedirs(pointcloud_path)

            with open(pose_path, "wb") as f:
                pickle.dump(
                    {
                        "camposes": [frame.campose for frame in frames],
                        "gelposes": [frame.gelpose for frame in frames],
                    },
                    f,
                )
                f.close()

            for i, frame in enumerate(frames):
                cv2.imwrite(osp.join(rgbframe_path, f"{i}.png"), frame.rgbframe)
                np.save(osp.join(heightmap_path, f"{i}.npy"), frame.heightmap)
                cv2.imwrite(
                    osp.join(contactmask_path, f"{i}.png"),
                    255 * frame.contactmask.astype(np.uint8),
                )
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(frame.pointcloud)
                o3d.io.write_point_cloud(osp.join(pointcloud_path, f"{i}.pcd"), pcd)
        else:
            for i, frame in enumerate(frames):
                with open(osp.join(save_path, f"{i}.pkl"), "wb") as f:
                    pickle.dump(frame, f)
                f.close()

        if not self._config.headless:
            # prepare point cloud
            pc_all = []
            for i, frame in enumerate(frames[::10]):
                pc_world = frame.get_world_pcd()[frame.contactmask.reshape(-1)]
                pc_all.append(pc_world)

            # visualize point cloud
            illustration_path = osp.join(save_path, "illustration.png")
            try:
                viz_poses_pointclouds_on_mesh(
                    self._trimesh,
                    np.array([frame.gelpose for frame in frames[::5]]),
                    pc_all,
                    decimation_factor=20,
                    save_path=illustration_path,
                )
                if osp.exists(illustration_path):
                    print(f"Visualization saved to {illustration_path}, successfully")
                else:
                    print(f"Visualization failed to save to {illustration_path}")
            except Exception as e:
                print(f"Visualization failed to save to {illustration_path}: {e}")

        pass

    def get_frames_from_poses(self, poses: np.ndarray) -> List[TactoFrame]:
        """
        Helper function to get frames from poses.
        Args:
            poses (np.ndarray): An array of 4x4 matrices representing the poses.
        Returns:
            List[TactoFrame]: A list of TactoFrame objects sampled at the specified poses.
        """
        batch_size = 1000
        traj_sz = poses.shape[0]
        num_batches = traj_sz // batch_size
        num_batches = num_batches if (num_batches != 0) else 1

        frames = []

        for idx in tqdm(range(num_batches)):
            idx_range = (
                np.array(range(idx * batch_size, traj_sz))
                if idx == num_batches - 1
                else np.array(range(idx * batch_size, (idx + 1) * batch_size))
            )
            (heightmaps, contactmasks, rgbframes, camposes, gelposes, _) = (
                self._renderer.render_sensor_trajectory(poses[idx_range, :])
            )
            for i in range(len(rgbframes)):
                frame = TactoFrame(
                    rgbframe=rgbframes[i],
                    heightmap=heightmaps[i],
                    contactmask=contactmasks[i],
                    pointcloud=self._renderer.heightmap_to_pointcloud(heightmaps[i]),
                    campose=camposes[i],
                    gelpose=gelposes[i],
                )
                frames.append(frame)

        return frames

    def get_pcd_from_frames(
        self, frames: List[TactoFrame], total_points: int = 100000
    ) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        points = []
        points_per_frame = total_points // len(frames)
        for frame in frames:
            pc = frame.get_world_pcd()[frame.contactmask.reshape(-1)]
            indices = np.random.choice(len(pc), points_per_frame)
            pc = pc[indices]
            points.append(pc)

        pcd.points = o3d.utility.Vector3dVector(np.concatenate(points, axis=0))
        return pcd
