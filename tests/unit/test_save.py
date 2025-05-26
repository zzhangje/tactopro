from tactopro import TactoPro
import os.path as osp
import os
import shutil


def check_directory_structure(directory, ycb_slide_type: bool = False):
    """
    Check if the directory structure is correct.
    """
    assert osp.exists(directory), f"Directory {directory} should exist."
    if ycb_slide_type:
        assert osp.exists(
            osp.join(directory, "rgbframes")
        ), "RGB frames directory should exist."
        assert osp.exists(
            osp.join(directory, "heightmaps")
        ), "Height maps directory should exist."
        assert osp.exists(
            osp.join(directory, "contactmasks")
        ), "Contact masks directory should exist."
        assert osp.exists(osp.join(directory, "poses.pkl")), "Poses file should exist."


def test_save_data(config):
    tp = TactoPro("tests/data/object.stl", config=config)
    if osp.exists("tests/tmp/"):
        shutil.rmtree("tests/tmp/")
    os.makedirs("tests/tmp/")

    tp.save([], "tests/tmp/object")
    check_directory_structure("tests/tmp/object")

    tp.save([], "tests/tmp/object", True)
    check_directory_structure("tests/tmp/object_1", True)

    tp.save([], "tests/tmp/object", True)
    check_directory_structure("tests/tmp/object_2", True)
