from tactopro import TactoPro

tp = TactoPro("tests/data/object.stl")
frames = tp.sample_frames_uniformly(2000)
tp.save(frames, "public/example")
