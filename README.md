With `tactopro`, you don't need to manually operate the `tacto` package or write a lot of code to sample tactile images. All you need are the following:

```py
from tactopro import TactoPro

tp = TactoPro("xxx.STL")
frames = tp.sample_poses_uniformly(10)
tp.save(frames, "dataset")
```
