from tactopro import TactoPro
from tactopro import get_ycb_object_path


def test_assets():
    TactoPro(get_ycb_object_path("035_power_drill"))
    pass
