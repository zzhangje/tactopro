from tactopro.tactopro import TactoPro
import pytest
import trimesh


def test_load_mesh():
    """
    Test the loading of a mesh file.
    """
    tactopro = TactoPro("tests/data/digit.STL")
    assert tactopro.trimesh is not None, "Mesh should be loaded successfully."
    assert isinstance(tactopro.trimesh, trimesh.Trimesh), "Loaded object should be a trimesh object."
    assert len(tactopro.trimesh.vertices) == 14455, f"Mesh should have 14455 vertices, but found {len(tactopro.trimesh.vertices)}."
    assert len(tactopro.trimesh.edges) == 87690, f"Mesh should have 87690 edges, but found {len(tactopro.trimesh.edges)}."
    assert len(tactopro.trimesh.faces) == 29230, f"Mesh should have 29230 faces, but found {len(tactopro.trimesh.faces)}."