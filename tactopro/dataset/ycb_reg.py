import os
import pickle
from typing import Optional, List, Tuple
import trimesh
from trimesh.base import Trimesh
import open3d as o3d
import numpy as np
from tactopro import load_ycb_object, get_ycb_object_path
from tactopro.helpers.viz import viz_pointclouds_on_mesh


def load_ycb_reg(
    path: str,
) -> Tuple[
    o3d.geometry.TriangleMesh,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    str,
    np.ndarray,
]:
    """
    Load a YCB registration object mesh from the assets directory.
    """
    with open(os.path.join(path, "label.txt"), "rb") as f:
        object_name = f.read(1024).strip().decode("utf-8")
    src_pcd_path = os.path.join(path, "src.pcd")
    tar_pcd_path = os.path.join(path, "tar.pcd")
    gt_path = os.path.join(path, "gt.npy")

    mesh = o3d.io.read_triangle_mesh(get_ycb_object_path(object_name))
    src_pcd = o3d.io.read_point_cloud(src_pcd_path)
    tar_pcd = o3d.io.read_point_cloud(tar_pcd_path)
    gt = np.load(gt_path)

    return mesh, src_pcd, tar_pcd, object_name, gt


def save_ycb_reg(path: str, label: str, pcd: o3d.geometry.PointCloud, gt: np.ndarray):
    """
    Save a YCB registration object mesh to the assets directory.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "label.txt"), "wb") as f:
        f.write(label.encode("utf-8"))
    viz_pointclouds_on_mesh(
        load_ycb_object(label), np.asarray(pcd.points), os.path.join(path, "img.png")
    )
    pcd.transform(gt)
    o3d.io.write_point_cloud(os.path.join(path, "src.pcd"), pcd)
    mesh = o3d.io.read_triangle_mesh(get_ycb_object_path(label))
    o3d.io.write_point_cloud(
        os.path.join(path, "tar.pcd"),
        mesh.sample_points_uniformly(number_of_points=150000),
    )
    np.save(os.path.join(path, "gt.npy"), gt)
