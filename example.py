from tactopro import TactoPro, TactoConfig

tp = TactoPro("public/hkust.stl", config=TactoConfig())
frames = tp.sample_frames_uniformly(500)
tp.save(frames, "public/hkust")
