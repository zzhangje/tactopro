from tactopro import TactoPro

tp = TactoPro("public/hkust.stl")
frames = tp.sample_frames_uniformly(500)
tp.save(frames, "public/hkust")
