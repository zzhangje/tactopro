# A modified version of `digit_renderer.py` by Meta Platforms, Inc. and affiliates.
# See: https://github.com/facebookresearch/MidasTouch

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from os import path as osp
import tacto
from tacto.loaders.object_loader import object_loader
from tacto.renderer import euler2matrix
import cv2
from .helpers.pose import pose_from_vertex_normal
from scipy.spatial.transform import Rotation as R

import trimesh
import random

DEBUG = False


@dataclass
class RendererConfig:
    """
    Configuration parameters for the Renderer.
    """

    # default values
    pixmm: float = 0.03
    width: int = 240
    height: int = 320
    cam_dist: float = 0.022
    min_pen: float = 0.0005
    max_pen: float = 0.001
    bg_id: int = 30
    headless: bool = False
    randomize: bool = False


class Renderer:
    def __init__(
        self, config: RendererConfig = RendererConfig(), trimesh_path: str = ""
    ):

        self.render_config = config

        if config.randomize:
            bg_id = random.randint(0, 30)
            print(f"Randomize on, background id: {bg_id}")
            config.bg_id = bg_id
            self.render_config.bg_id = bg_id

        # Create renderer
        self.renderer = tacto.Renderer(
            width=config.width,
            height=config.height,
            background=cv2.imread(tacto.get_background_image_path(config.bg_id)),
            config_path=tacto.get_digit_shadow_config_path(),
            headless=config.headless,
        )
        self.cam_dist = config.cam_dist
        self.pixmm = config.pixmm

        if not DEBUG:
            _, self.bg_depth = self.renderer.render()
            self.bg_depth = self.bg_depth[0]
            self.bg_depth_pix = self.correct_pyrender_height_map(self.bg_depth)

        if trimesh_path != "":
            self.obj_loader = object_loader(trimesh_path)
            obj_trimesh = trimesh.load(trimesh_path)
            self.renderer.add_object(obj_trimesh, "object")

        self.press_depth = 0.001

    @staticmethod
    def get_ycbslide_renderer() -> "Renderer":
        """
        Returns a RendererConfig instance with parameters suitable for YcbSlide.
        """
        return Renderer(
            RendererConfig(
                pixmm=0.03,
                width=240,
                height=320,
                cam_dist=0.022,
                min_pen=0.0005,
                max_pen=0.001,
                bg_id=30,
                headless=False,
                randomize=False,
            )
        )

    def get_background(self, frame="gel"):
        """
        Return cached bg image
        """
        return self.bg_depth_pix if frame == "gel" else self.bg_depth

    def pix2meter(self, pix):
        """
        Convert pixel to meter
        """
        return pix * self.pixmm / 1000.0

    def meter2pix(self, m):
        """
        Convert meter to pixels
        """
        return m * 1000.0 / self.pixmm

    def update_pose_given_point(self, point, press_depth, shear_mag, delta):
        """
        Convert meter to pixels
        """
        dist = np.linalg.norm(point - self.obj_loader.obj_vertices, axis=1)
        idx = np.argmin(dist)

        # idx: the idx vertice, get a new pose
        new_position = self.obj_loader.obj_vertices[idx].copy()
        new_orientation = self.obj_loader.obj_normals[idx].copy()

        delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(1,))[0]
        new_pose = pose_from_vertex_normal(
            new_position, new_orientation, shear_mag, delta
        ).squeeze()
        self.update_pose_given_pose(press_depth, new_pose)

    def update_pose_given_pose(self, press_depth, gel_pose):
        """
        Given tf gel_pose and press_depth, update tacto camera
        """
        self.press_depth = press_depth
        cam_pose = self.gel2cam(gel_pose)
        cam_pose = self.add_press(cam_pose)
        self.renderer.update_camera_pose_from_matrix(self.fix_transform(cam_pose))
        # self.renderer.update_camera_pose_from_matrix(cam_pose)

    def fix_transform(self, pose):
        """
        Inverse of transformation in config_digit_shadow.yml
        """
        switch_axes = euler2matrix(angles=[-90, 0, 90], xyz="zyx", degrees=True)
        return np.matmul(pose, switch_axes)

    def add_press(self, pose):
        """
        Add sensor penetration
        """
        pen_mat = np.eye(4)
        pen_mat[2, 3] = -self.press_depth
        return np.matmul(pose, pen_mat)

    def gel2cam(self, gel_pose):
        """
        Convert gel_pose to cam_pose
        """
        cam_tf = np.eye(4)
        cam_tf[2, 3] = self.cam_dist
        return np.matmul(gel_pose, cam_tf)

    def cam2gel(self, cam_pose):
        """
        Convert cam_pose to gel_pose
        """
        cam_tf = np.eye(4)
        cam_tf[2, 3] = -self.cam_dist
        return np.matmul(cam_pose, cam_tf)

    # input depth is in camera frame here
    def render(self):
        """
        render [tactile image + depth + mask] @ current pose
        """
        color, depth = self.renderer.render()
        color, depth = color[0], depth[0]
        diff_depth = (self.bg_depth) - depth
        contact_mask = diff_depth > np.abs(self.press_depth * 0.2)
        #  pix in gel frame
        gel_depth = self.correct_pyrender_height_map(depth)
        # cam_depth = self.correct_image_height_map(gel_depth)
        # assert np.allclose(cam_depth, depth), "Conversion to pixels is incorrect"
        if self.render_config.randomize:
            self.renderer.randomize_light()
        return color, gel_depth, contact_mask

    def correct_pyrender_height_map(self, height_map):
        """
        Input: height_map in meters, in camera frame
        Output: height_map in pixels, in gel frame
        """
        # move to the gel center
        height_map = (self.cam_dist - height_map) * (1000 / self.pixmm)
        return height_map

    def correct_image_height_map(self, height_map, output_frame="cam"):
        """
        Input: height_map in pixels, in gel frame
        Output: height_map in meters, in camera/gel frame
        """
        height_map = (
            -height_map * (self.pixmm / 1000)
            + float(output_frame == "cam") * self.cam_dist
        )
        return height_map

    def get_cam_pose_matrix(self):
        """
        return camera pose matrix of renderer
        """
        return self.renderer.camera_nodes[0].matrix

    def get_cam_pose(self):
        """
        return camera pose of renderer
        """
        # print(f"Cam pose: {tf_to_xyzquat(self.get_cam_pose_matrix())}")
        return self.get_cam_pose_matrix()

    def get_gel_pose_matrix(self):
        """
        return gel pose matrix of renderer
        """
        return self.cam2gel(self.get_cam_pose_matrix())

    def get_gel_pose(self):
        """
        return gel pose of renderer
        """
        # print(f"Gel pose: {tf_to_xyzquat(self.get_gel_pose_matrix())}")
        return self.get_gel_pose_matrix()

    def heightmap_to_pointcloud(self, depth):
        """
        Convert heightmap to point cloud.
        [Input]  depth: (height, width) in pixels, in gel frame
        [Output] pointcloud: [N, 3] in meters in camera frame
        """
        depth = np.asarray(depth)
        depth = self.correct_image_height_map(depth, output_frame="cam")

        f = self.renderer.f
        w = self.renderer.width / 2.0
        h = self.renderer.height / 2.0

        yvals = np.arange(depth.shape[0])
        xvals = np.arange(depth.shape[1])
        x, y = np.meshgrid(xvals, yvals)

        # x and y in meters
        x = ((x - w)) / f
        y = ((y - h)) / f

        x = x * depth
        y = -y * depth

        z = -depth

        heightmap_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        heightmap_3d = heightmap_3d[heightmap_3d[:, 2] != 0]
        return heightmap_3d

    def render_sensor_trajectory(
        self, p, mNoise=None, pen_ratio=1.0, over_pen=False
    ) -> tuple:
        """
        Render a trajectory of poses p via tac_render
        """
        p = np.atleast_3d(p)

        N = p.shape[0]
        images, heightmaps, contactMasks = (
            np.empty(N, dtype=object),
            np.empty(N, dtype=object),
            np.empty(N, dtype=object),
        )
        gelposes, camposes = np.zeros([N, 4, 4]), np.zeros([N, 4, 4])

        min_press, max_press = (
            self.render_config.min_pen * pen_ratio,
            self.render_config.max_pen * pen_ratio,
        )
        # print(f"min_press: {min_press}, max_press: {max_press}")
        press_depth = np.random.uniform(low=min_press, high=max_press)
        press_range = max_press - min_press

        idx = 0
        for p0 in p:

            delta = np.random.uniform(-press_range / 50.0, press_range / 50.0)
            if press_depth + delta > max_press or press_depth + delta < min_press:
                press_depth -= delta
            else:
                press_depth += delta

            self.update_pose_given_pose(press_depth, p0)

            tactile_img, height_map, contact_mask = self.render()

            if over_pen:
                # Check for over-pen and compensate
                diff_pen = height_map - self.get_background()  # pixels in gel frame
                diff_pen_max = self.pix2meter(np.abs(diff_pen.max())) - max_press
                if diff_pen_max > 0:
                    new_depth = press_depth - diff_pen_max
                    self.update_pose_given_pose(new_depth, p0)
                    tactile_img, height_map, contact_mask = self.render()

            heightmaps[idx], contactMasks[idx], images[idx] = (
                height_map,
                contact_mask,
                tactile_img,
            )
            gelposes[idx, :] = self.get_gel_pose()
            camposes[idx, :] = self.get_cam_pose()
            idx += 1

        # measurement with noise
        if mNoise is not None:
            rotNoise = np.random.normal(loc=0.0, scale=mNoise["sig_r"], size=(N, 3))
            Rn = R.from_euler("zyx", rotNoise, degrees=True).as_matrix()  # (N, 3, 3)
            tn = np.random.normal(loc=0.0, scale=mNoise["sig_t"], size=(N, 3))
            Tn = np.zeros((N, 4, 4))
            Tn[:, :3, :3], Tn[:, :3, 3], Tn[:, 3, 3] = Rn, tn, 1
            gelposes_meas = gelposes @ Tn
        else:
            gelposes_meas = gelposes

        return heightmaps, contactMasks, images, camposes, gelposes, gelposes_meas

    def render_sensor_poses(self, p, num_depths=1, no_contact_prob=0):
        """
        Render an unordered set of poses p via tac_render
        """
        p = np.atleast_3d(p)

        N = p.shape[0] * num_depths
        images, heightmaps, contactMasks = [None] * N, [None] * N, [None] * N
        gelposes, camposes = np.zeros([N, 4, 4]), np.zeros([N, 4, 4])

        idx = 0
        for p0 in p:

            # loop over # press depths
            for _ in range(num_depths):
                # randomly sample no contact for no_contact_prob% of trials
                p_no_contact = random.randrange(100) < no_contact_prob
                if p_no_contact:
                    press_depth = -self.render_config.min_pen
                else:
                    press_depth = np.random.uniform(
                        low=self.render_config.min_pen, high=self.render_config.max_pen
                    )

                self.update_pose_given_pose(press_depth, p0)

                tactile_img, height_map, contact_mask = self.render()
                # Check for over-pen and compensate
                diff_pen = height_map - self.get_background()  # pixels in gel frame
                diff_pen_max = (
                    self.pix2meter(np.abs(diff_pen.max())) - self.render_config.max_pen
                )
                if diff_pen_max > 0:
                    press_depth -= diff_pen_max
                    self.update_pose_given_pose(press_depth, p0)
                    tactile_img, height_map, contact_mask = self.render()

                heightmaps[idx], contactMasks[idx], images[idx] = (
                    height_map,
                    contact_mask,
                    tactile_img,
                )
                gelposes[idx, :] = self.get_gel_pose()
                camposes[idx, :] = self.get_cam_pose()
                idx += 1

        return heightmaps, contactMasks, images, camposes, gelposes


def ycb_slide_heightmap_to_pointcloud(
    heightmap: np.ndarray, renderer: Renderer
) -> np.ndarray:
    """
    Convert a heightmap to a point cloud using the given renderer.
    Args:
        heightmap (np.ndarray): The heightmap to convert.
        renderer (Renderer): The renderer to use for conversion.
    Returns:
        np.ndarray: The resulting point cloud.
    """
    return renderer.heightmap_to_pointcloud(heightmap)
