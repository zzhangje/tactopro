# TACTO Pro

TACTO Pro is a lightweight wrapper designed to streamline the creation of TACTO simulations, significantly reducing code complexity and boilerplate. It supports Python versions 3.8 through 3.13.

With `tactopro`, there's no need to manually interact with the `tacto` package or write extensive custom code to sample tactile images. Simply follow this workflow:

```py
from tactopro import TactoPro

tp = TactoPro("xxx.stl")
frames = tp.sample_frames_uniformly(1000)
tp.save(frames, "dataset")
```

After running the code, you'll get results similar to the following:

<p align="center"> 
    <img src="./public/image.png" width=60%>
</p>

## Setup

You can install TACTO Pro using pip:

```sh
pip install tactopro@git+https://github.com/ZhangzrJerry/TactoPro.git
```

Alternatively, clone the repository and install it manually:

```sh
git clone https://github.com/ZhangzrJerry/TactoPro && cd TactoPro
pip install -e .
```

## License

This project is licensed under the MIT License, as detailed in the [LICENSE](./LICENSE) file.

Some visualization utilities and rendering functions are adapted from [MidasTouch](https://github.com/facebookresearch/MidasTouch), which is primarily distributed under the MIT License.
