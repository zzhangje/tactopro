from tactopro import TactoPro

tp = TactoPro("tests/data/digit.STL")
frames = tp.sample_frames_uniformly(2)
tp.save(frames, "tests/tmp/digit")
