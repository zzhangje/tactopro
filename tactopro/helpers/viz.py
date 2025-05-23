# A modified version of `helper.py` by Meta Platforms, Inc. and affiliates.
# See: https://github.com/facebookresearch/MidasTouch

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license


import numpy as np
import pyvista as pv
import trimesh
from typing import List
from scipy.spatial.transform import Rotation as R


def viz_poses_pointclouds_on_mesh(
    trimesh: trimesh.Trimesh,
    poses: List[np.ndarray],
    pointclouds: List[np.ndarray],
    save_path: str,
    decimation_factor: int = 10,
) -> None:
    """
    Visualizes a set of 4x4 pose matrices and associated pointclouds on a given mesh using PyVista.

    Args:
        trimesh (trimesh.Trimesh): The mesh to visualize. Shape: (varies by mesh).
        poses (List[np.ndarray]): List of N pose matrices, each of shape (4, 4).
        pointclouds (List[np.ndarray]): List of N pointclouds, each of shape (M_i, 3).
        save_path (str): Path to save the rendered image. If empty, shows interactively.
        decimation_factor (int, optional): Factor to downsample each pointcloud for visualization. Defaults to 10.
    """
    plotter = pv.Plotter(window_size=[2000, 2000], off_screen=True)

    mesh = pv.wrap(trimesh)
    dargs = dict(
        color="grey",
        ambient=0.6,
        opacity=0.5,
        smooth_shading=True,
        specular=1.0,
        show_scalar_bar=False,
        render=False,
    )
    plotter.add_mesh(mesh, **dargs)
    draw_poses(plotter, mesh, poses, quiver_size=0.05)

    if poses.ndim == 2:
        spline = pv.lines_from_points(poses[:, :3])
        plotter.add_mesh(spline, line_width=3, color="k")

    final_pc = np.empty((0, 3))
    for i, pointcloud in enumerate(pointclouds):
        if pointcloud.shape[0] == 0:
            continue
        if decimation_factor is not None:
            downpcd = pointcloud[
                np.random.choice(
                    pointcloud.shape[0],
                    pointcloud.shape[0] // decimation_factor,
                    replace=False,
                ),
                :,
            ]
        else:
            downpcd = pointcloud
        final_pc = np.vstack([final_pc, downpcd])

    if final_pc.shape[0]:
        pc = pv.PolyData(final_pc)
        plotter.add_points(
            pc, render_points_as_spheres=True, color="#26D701", point_size=3
        )

    if save_path:
        plotter.show(screenshot=save_path)
        print(f"Save path: {save_path}")
    else:
        plotter.show()
    plotter.close()
    pv.close_all()


def draw_poses(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    cluster_poses: np.ndarray,
    opacity: float = 1.0,
    quiver_size=0.05,
) -> None:
    """
    Draws RGB coordinate axes for a set of poses in a PyVista visualizer.
    Args:
        plotter (pv.Plotter): The PyVista plotter object to draw on.
        mesh (pv.DataSet): The mesh object used for scaling the quiver size.
        cluster_poses (np.ndarray): Array of shape (N, 4, 4) representing N pose transformation matrices.
        opacity (float, optional): Opacity of the drawn arrows. Defaults to 1.0.
        quiver_size (float, optional): Relative size of the coordinate axes arrows. Defaults to 0.05.
    """
    quivers = poses_to_quivers(cluster_poses, quiver_size * mesh.length)
    quivers = [quivers["xvectors"]] + [quivers["yvectors"]] + [quivers["zvectors"]]
    names = ["xvectors", "yvectors", "zvectors"]
    colors = ["r", "g", "b"]
    cluster_centers = cluster_poses[:, :3, 3]
    for q, c, n in zip(quivers, colors, names):
        plotter.add_arrows(
            cluster_centers,
            q,
            color=c,
            opacity=opacity,
            show_scalar_bar=False,
            render=False,
            name=n,
        )


def poses_to_quivers(poses: np.ndarray, sz: float) -> pv.PolyData:
    """
    Convert a set of poses to quivers for visualization in pyvista.
    Args:
        poses (np.ndarray): Array of shape (N, 4, 4) representing N poses.
        sz (float): Size of the quivers.
    Returns:
        pv.PolyData: PolyData object containing the quivers.
    """
    quivers = pv.PolyData(poses[:, :3, 3])  # (N, 3) [x, y, z]
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    r = R.from_matrix(poses[:, 0:3, 0:3])  # (N, 3, 3)
    quivers["xvectors"], quivers["yvectors"], quivers["zvectors"] = (
        r.apply(x) * sz,
        r.apply(y) * sz,
        r.apply(z) * sz,
    )
    return quivers
