from tactopro import TactoPro
import trimesh


def test_load_mesh(config):
    """
    Test the loading of a mesh file.
    """
    tp = TactoPro("tests/data/object.stl", config=config)
    assert tp.trimesh is not None, "Mesh should be loaded successfully."
    assert isinstance(
        tp.trimesh, trimesh.Trimesh
    ), "Loaded object should be a trimesh object."
    assert (
        len(tp.trimesh.vertices) == 10664
    ), f"Mesh should have 10664 vertices, but found {len(tp.trimesh.vertices)}."
    assert (
        len(tp.trimesh.edges) == 63972
    ), f"Mesh should have 63972 edges, but found {len(tp.trimesh.edges)}."
    assert (
        len(tp.trimesh.faces) == 21324
    ), f"Mesh should have 21324 faces, but found {len(tp.trimesh.faces)}."
