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
        R = self.campose[:3, :3]
        t = self.campose[:3, 3]
        return (R @ self.pointcloud.T).T + t

    @staticmethod
    def load_from_ycbslide(file_path: str, idx: int) -> "TactoFrame":
        try:
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
                pointcloud=Renderer.get_ycbslide_renderer().heightmap_to_pointcloud(h)[
                    c.reshape(-1)
                ],
                campose=cam_pose,
                gelpose=gel_pose,
            )
        except Exception as e:
            print(f"Error loading TactoFrame from YCB slide: {e}")
            return None


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
        return random_geodesic_poses(
            self._trimesh, total_length=traj_length, N=num_samples
        )

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
                f.close()

            for i, frame in enumerate(frames):
                cv2.imwrite(osp.join(rgbframe_path, f"{i}.png"), frame.rgbframe)
                np.save(osp.join(heightmap_path, f"{i}.npy"), frame.heightmap)
                cv2.imwrite(
                    osp.join(contactmask_path, f"{i}.png"),
                    255 * frame.contactmask.astype(np.uint8),
                )
        else:
            for i, frame in enumerate(frames):
                with open(osp.join(save_path, f"{i}.pkl"), "wb") as f:
                    pickle.dump(frame, f)
                f.close()

        if not self._config.headless:
            # prepare point cloud
            pc_all = []
            for i, frame in enumerate(frames[::10]):
                pc_body = frame.pointcloud
                R = frame.campose[:3, :3]
                t = frame.campose[:3, 3]
                pc_world = (R @ pc_body.T).T + t
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

    def get_frames_from_poses(self, poses: List[np.ndarray]) -> List[TactoFrame]:
        """
        Helper function to get frames from poses.
        Args:
            poses (List[np.ndarray]): A list of 4x4 matrices representing the poses.
        Returns:
            List[TactoFrame]: A list of TactoFrame objects sampled at the specified poses.
        """
        batch_size = 1000
        traj_sz = poses.shape[0]
        num_batches = traj_sz // batch_size
        num_batches = num_batches if (num_batches != 0) else 1

        frames = []
        print(poses.shape)

        for idx in tqdm(range(num_batches)):
            idx_range = (
                np.array(range(idx * batch_size, traj_sz))
                if idx == num_batches - 1
                else np.array(range(idx * batch_size, (idx + 1) * batch_size))
            )
            (heightmaps, contactmasks, rgbframes, camposes, gelposes) = (
                self._renderer.render_sensor_trajectory(poses[idx_range, :])
            )
            for i in range(len(rgbframes)):
                frame = TactoFrame(
                    rgbframe=rgbframes[i],
                    heightmap=heightmaps[i],
                    contactmask=contactmasks[i],
                    pointcloud=self._renderer.heightmap_to_pointcloud(heightmaps[i])[
                        contactmasks[i].reshape(-1)
                    ],
                    campose=camposes[i],
                    gelpose=gelposes[i],
                )
                frames.append(frame)

        return frames

    pass
