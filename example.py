from tactopro import TactoPro

tp = TactoPro("tests/data/object.stl")
frames = tp.sample_frames_trajectory(100)
tp.save(frames, "tests/tmp/object")
