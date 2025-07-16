import os
import pickle
from typing import Optional, List
import trimesh
from trimesh.base import Trimesh


__dataset_path__ = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets/ycb_objects")
)
__available_dir__ = [
    # "002_master_chef_can/",
    "003_cracker_box/",
    "004_sugar_box/",
    "005_tomato_soup_can/",
    "006_mustard_bottle/",
    "007_tuna_fish_can/",
    "008_pudding_box/",
    "009_gelatin_box/",
    "010_potted_meat_can/",
    "011_banana/",
    "012_strawberry/",
    "013_apple/",
    "014_lemon/",
    "015_peach/",
    "016_pear/",
    "017_orange/",
    "018_plum/",
    "019_pitcher_base/",
    "021_bleach_cleanser/",
    "024_bowl/",
    "025_mug/",
    "026_sponge/",
    "029_plate/",
    "030_fork/",
    "031_spoon/",
    "032_knife/",
    "033_spatula/",
    "035_power_drill/",
    "036_wood_block/",
    "037_scissors/",
    "040_large_marker/",
    "042_adjustable_wrench/",
    "044_flat_screwdriver/",
    "048_hammer/",
    "050_medium_clamp/",
    "051_large_clamp/",
    "052_extra_large_clamp/",
    "053_mini_soccer_ball/",
    "054_softball/",
    "055_baseball/",
    "056_tennis_ball/",
    "057_racquetball/",
    "058_golf_ball/",
    "061_foam_brick/",
    # "062_dice/",
    "065-a_cups/",
    "065-b_cups/",
    "070-a_colored_wood_blocks/",
    "072-a_toy_airplane/",
    "073-a_lego_duplo/",
    "073-b_lego_duplo/",
    "073-c_lego_duplo/",
    "073-d_lego_duplo/",
    "073-e_lego_duplo/",
    "073-f_lego_duplo/",
    "073-g_lego_duplo/",
    "077_rubiks_cube/",
]


def load_ycb_object(file_name: str) -> Trimesh:
    """
    Load a YCB object mesh from the assets directory.
    """
    file_path = os.path.join(__dataset_path__, file_name, "nontextured.stl")
    return trimesh.load_mesh(file_path)


def get_ycb_object_path(file_name: str) -> str:
    return os.path.join(__dataset_path__, file_name, "nontextured.stl")
