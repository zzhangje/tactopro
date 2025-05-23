from tactopro import TactoPro
import trimesh


def test_load_mesh():
    """
    Test the loading of a mesh file.
    """
    tp = TactoPro("tests/data/digit.STL")
    assert tp.trimesh is not None, "Mesh should be loaded successfully."
    assert isinstance(tp.trimesh, trimesh.Trimesh), "Loaded object should be a trimesh object."
    assert len(tp.trimesh.vertices) == 14455, f"Mesh should have 14455 vertices, but found {len(tp.trimesh.vertices)}."
    assert len(tp.trimesh.edges) == 87690, f"Mesh should have 87690 edges, but found {len(tp.trimesh.edges)}."
    assert len(tp.trimesh.faces) == 29230, f"Mesh should have 29230 faces, but found {len(tp.trimesh.faces)}."