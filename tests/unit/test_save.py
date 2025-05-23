from tactopro import TactoPro
import os.path as osp
import os
import shutil

def test_save_data():
    tp = TactoPro("tests/data/digit.STL")
    if osp.exists("tests/tmp/"):
        shutil.rmtree("tests/tmp/")
    os.makedirs("tests/tmp/")
    
    tp.save([], "tests/tmp/digit", headless=True)
    assert osp.exists("tests/tmp/digit"), "Directory should be created."
    assert osp.exists("tests/tmp/digit/rgbframes"), "RGB frames directory should be created."
    assert osp.exists("tests/tmp/digit/heightmaps"), "Height maps directory should be created."
    assert osp.exists("tests/tmp/digit/contactmasks"), "Contact masks directory should be created."
    assert osp.exists("tests/tmp/digit/poses.pkl"), "Poses file should be created."

    tp.save([], "tests/tmp/digit", headless=True)
    assert osp.exists("tests/tmp/digit_1"), "Directory should be created."
    assert osp.exists("tests/tmp/digit_1/rgbframes"), "RGB frames directory should be created."
    assert osp.exists("tests/tmp/digit_1/heightmaps"), "Height maps directory should be created."
    assert osp.exists("tests/tmp/digit_1/contactmasks"), "Contact masks directory should be created."
    assert osp.exists("tests/tmp/digit_1/poses.pkl"), "Poses file should be created."

    tp.save([], "tests/tmp/digit", headless=True)
    assert osp.exists("tests/tmp/digit_2"), "Directory should be created."
    assert osp.exists("tests/tmp/digit_2/rgbframes"), "RGB frames directory should be created."
    assert osp.exists("tests/tmp/digit_2/heightmaps"), "Height maps directory should be created."
    assert osp.exists("tests/tmp/digit_2/contactmasks"), "Contact masks directory should be created."
    assert osp.exists("tests/tmp/digit_2/poses.pkl"), "Poses file should be created."