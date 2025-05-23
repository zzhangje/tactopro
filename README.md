# TACTO Pro:

With `tactopro`, you don't need to manually operate the `tacto` package or write a lot of code to sample tactile images. All you need are the following:

```py
from tactopro import TactoPro

tp = TactoPro("xxx.STL")
frames = tp.sample_frames_uniformly(2)
tp.save(frames, "dataset")
```

Then you would get something like:

## Install

## License

This project is licensed under the MIT License, as detailed in the [LICENSE](./LICENSE) file.

Some visualization utilities and rendering functions are adapted from [MidasTouch](https://github.com/facebookresearch/MidasTouch), which is primarily distributed under the MIT License. Portions of the code are available under separate license terms: FCRN-DepthPrediction is licensed under the BSD 2-Clause License, and pytorch3d is licensed under the BSD 3-Clause License.
