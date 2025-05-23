# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Mesh processing utilities
"""

import numpy as np
import trimesh
import pyvista as pv
from scipy.spatial import KDTree
from .pose import pose_from_vertex_normal
from typing import Tuple
import math
import random
import time
import potpourri3d as pp3d


# mesh utils
def sample_mesh(
    mesh: trimesh.base.Trimesh, num_samples: int, method: str = "even"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample mesh and return point/normals
    """
    sampled_points, faces = np.empty((0, 3)), np.array([], dtype=int)
    # https://github.com/mikedh/trimesh/issues/558 : trimesh.sample.sample_surface_even gives wrong number of samples
    while True:
        if method == "even":
            sP, f = trimesh.sample.sample_surface_even(mesh, count=num_samples)
        else:
            sP, f = trimesh.sample.sample_surface(mesh, count=num_samples)
        sampled_points = np.vstack([sampled_points, sP])
        faces = np.concatenate([faces, f])
        if len(sampled_points) <= num_samples:
            continue
        else:
            sampled_points, faces = sampled_points[:num_samples, :], faces[:num_samples]
            break

    sampled_normals = mesh.face_normals[faces, :]
    sampled_normals = sampled_normals / np.linalg.norm(sampled_normals, axis=1).reshape(
        -1, 1
    )
    return sampled_points, sampled_normals


def extract_edges(
    mesh: trimesh.base.Trimesh, num_samples: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract mesh edges via pyvista
    """
    mesh = pv.wrap(mesh)
    edges = mesh.extract_feature_edges(10)
    edges.compute_normals(inplace=True)  # this activates the normals as well

    tree = KDTree(mesh.points)
    _, ii = tree.query(edges.points, k=1)
    edgePoints, edgeNormals = edges.points, mesh.point_normals[ii, :]

    if edgePoints.shape[0] < num_samples:
        num_samples = edgePoints.shape[0]

    # https://stackoverflow.com/a/14262743/8314238
    indices = np.random.choice(edgePoints.shape[0], num_samples, replace=False)
    edgePoints = edgePoints[indices, :]
    edgeNormals = edgeNormals[indices, :] / np.linalg.norm(
        edgeNormals[indices, :], axis=1
    ).reshape(-1, 1)
    return edgePoints, edgeNormals, num_samples


def sample_mesh_edges(
    mesh: trimesh.base.Trimesh, num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample only mesh edges
    """
    sampled_edge_points, sampled_edge_normals, num_samples = extract_edges(
        mesh, num_samples
    )
    return sampled_edge_points, sampled_edge_normals


def sample_poses_on_mesh(
    mesh: trimesh.base.Trimesh,
    num_samples: int,
    edges: bool = True,
    constraint: np.ndarray = None,
    r: float = None,
    shear_mag: float = 5.0,
) -> np.ndarray:
    """
    Sample mesh and generates candidate sensor poses
    """
    if constraint is not None:
        constrainedSampledPoints, constrainedSampledNormals = np.empty(
            (0, 3)
        ), np.empty((0, 3))
        box = trimesh.creation.box(extents=[2 * r, 2 * r, 2 * r])
        box.apply_translation(constraint)
        constrainedMesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)
        while constrainedSampledPoints.shape[0] < num_samples:
            sP, sN = sample_mesh(constrainedMesh, num_samples * 100, method="even")
            dist = (np.linalg.norm(sP - constraint, axis=1)).squeeze()
            constrainedSampledPoints = np.append(
                constrainedSampledPoints, sP[np.less(dist, r), :], axis=0
            )
            constrainedSampledNormals = np.append(
                constrainedSampledNormals, sN[np.less(dist, r), :], axis=0
            )
        idxs = np.random.choice(constrainedSampledPoints.shape[0], num_samples)
        sampled_points, sampled_normals = (
            constrainedSampledPoints[idxs, :],
            constrainedSampledNormals[idxs, :],
        )
    elif edges:
        numSamplesEdges = int(0.3 * num_samples)
        sampled_edge_points, sampled_edge_normals, numSamplesEdges = extract_edges(
            mesh, numSamplesEdges
        )
        numSamplesEven = num_samples - numSamplesEdges
        sampledPointsEven, sampledNormalsEven = sample_mesh(mesh, numSamplesEven)
        sampled_points, sampled_normals = np.concatenate(
            (sampledPointsEven, sampled_edge_points), axis=0
        ), np.concatenate((sampledNormalsEven, sampled_edge_normals), axis=0)
    else:
        sampled_points, sampled_normals = sample_mesh(
            mesh, num_samples, method="normal"
        )

    # apply random pen into manifold
    shear_mag = np.radians(shear_mag)
    delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(num_samples,))
    T = pose_from_vertex_normal(sampled_points, sampled_normals, shear_mag, delta)
    return T


def sample_poses_on_mesh_minkloc(
    mesh: trimesh.base.Trimesh,
    num_samples: int,
    edges: bool = True,
    num_angles: int = 1,
    shear_mag: float = 5.0,
) -> np.ndarray:
    """
    Sample mesh and generates candidate sensor poses, custom for minkloc data
    """
    if edges:
        numSamplesEdges = int(0.3 * num_samples)
        sampled_edge_points, sampled_edge_normals, numSamplesEdges = extract_edges(
            mesh, numSamplesEdges
        )
        numSamplesEven = num_samples - numSamplesEdges
        sampledPointsEven, sampledNormalsEven = sample_mesh(mesh, numSamplesEven)
        sampled_points, sampled_normals = np.concatenate(
            (sampledPointsEven, sampled_edge_points), axis=0
        ), np.concatenate((sampledNormalsEven, sampled_edge_normals), axis=0)
    else:
        sampled_points, sampled_normals = sample_mesh(mesh, num_samples)

    sampled_points = np.repeat(sampled_points, num_angles, axis=0)
    sampled_normals = np.repeat(sampled_normals, num_angles, axis=0)

    # apply random pen into manifold
    delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(num_samples * num_angles,))
    T = pose_from_vertex_normal(sampled_points, sampled_normals, shear_mag, delta)
    return T


def random_geodesic_poses(mesh, shear_mag=5.0, total_length=0.5, N=2000):
    """Generate random points and compute geodesic trajectory"""

    cumm_length = 0.0
    num_waypoints = 1
    seg_length = total_length / float(num_waypoints)
    while seg_length > 0.25 * mesh.scale:
        num_waypoints += 1
        seg_length = total_length / float(num_waypoints)

    V, F = np.array(mesh.vertices), np.array(mesh.faces)
    print(f"num_waypoints: {num_waypoints}")
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    path_solver = pp3d.EdgeFlipGeodesicSolver(
        V, F
    )  # shares precomputation for repeated solves

    sample_points, sample_normals = np.empty((0, 3)), np.empty((0, 3))
    seg_start = random.randint(0, V.shape[0])

    waypoints = V[seg_start, None]
    start_time = time.time()
    tree = KDTree(mesh.vertices)

    for _ in range(num_waypoints):
        geo_dist = solver.compute_distance(seg_start)
        candidates = np.argsort(np.abs(geo_dist - seg_length))

        waypoint_dist = np.linalg.norm(V[candidates, :, None] - waypoints.T, axis=1)
        waypoint_dist = np.amin(waypoint_dist, axis=1)
        mask = waypoint_dist < 0.01
        candidates = np.ma.MaskedArray(candidates, mask=mask)
        candidates = candidates.compressed()
        print(candidates)
        print(np.shape(candidates))
        seg_end = candidates[0]
        seg_dist = geo_dist[seg_end]
        waypoints = np.concatenate((waypoints, V[seg_end, None]), axis=0)

        seg_points = path_solver.find_geodesic_path(v_start=seg_start, v_end=seg_end)
        # subsample path (spline) to 0.1mm per odom

        # length-based sampling
        # segmentSpline = pv.Spline(seg_points, n_segment)
        # segmentSpline = np.array(segmentSpline.points)

        _, ii = tree.query(seg_points, k=1)
        segmentNormals = mesh.vertex_normals[ii, :]

        sample_points = np.concatenate((sample_points, seg_points), axis=0)
        sample_normals = np.concatenate((sample_normals, segmentNormals), axis=0)

        cumm_length += seg_dist

        seg_start = seg_end
        if time.time() - start_time > 120:
            print("Timeout, trying again!")
            return None

    # interval-based sampling
    n_interval = math.ceil(len(sample_points) / N)
    n_interval = 1 if n_interval == 0 else n_interval
    sample_points = sample_points[::n_interval, :]
    sample_normals = sample_normals[::n_interval, :]

    delta = np.zeros(sample_points.shape[0])

    a = 1
    for i in range(1, sample_points.shape[0]):
        if i % int(sample_points.shape[0] / num_waypoints) == 0:
            a = -a
        delta[i] = delta[i - 1] + np.radians(np.random.normal(loc=a, scale=0.01))
    T = pose_from_vertex_normal(sample_points, sample_normals, shear_mag, delta)
    print(
        f"Dataset path length: {cumm_length:.4f} m, Num poses: {sample_points.shape[0]}, Time taken: {time.time() - start_time}"
    )
    return T
