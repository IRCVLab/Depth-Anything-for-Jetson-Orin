<div align="center">
<h1> Real-time Depth Estimation for Jetson Orin </h1>

This project aims to provide a real-time depth estimation optimized for NVIDIA Jetson Orin devices. Utilizing the power of the Jetson Orin's GPU, it processes live camera feeds to estimate depth maps in real-time, enabling applications in robotics, autonomous vehicles, and augmented reality among others. Thank you to [DepthAnything](https://github.com/LiheYoung/Depth-Anything) team for implementing real-time depth estimation.

<img src="assets/output.gif">
</img>

**Note: This video is unedited, and the frame rate may appear awkward due to the varying inference time of each frame.**

</div>

## Timelines
- **2024-04-04: This repository is public! 🎉**

## Requirements

### Hardware

- NVIDIA Jetson Orin Developer Kit(8GB)
- A CSI camera module(or Webcam)
- ~~Webcam~~\
  **The current code in ```camera.py``` <ins>only supports CSI cameras</ins>, but a simple code modification should enable it to work with USB Cameras as well. 😊**

### Software

- JetPack SDK (version: 5.1.3)
- Python 3.8.10
- CUDA (version: 11.4)
- TensorRT (version: 8.5.2)
- OpenCV (version: 4.5.4, ensure compatibility with GStreamer)

### Installations
- **Clone Repository**
```bash

```
- **Install TensorRT**
```bash
sudo apt install tensorrt
sudo apt update
```
- **Install torch-related packages**
```bash
# torch
pip3 install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl

# torchvision
pip3 install torchvision==0.15.0
```
For more details, please refer to Nvidia's [Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/). (**<ins>Do not install venv or Anaconda</ins>**)

- **Install Jetson Stats**
```bash
sudo pip3 install -U jetson-stats 
```
- You can check the memory usage using the command below:
```bash
jtop
```
### Camera Setup
- Ensure your camera is connected to the Jetson Orin.
- To check if GStreamer is working properly, run the following code:
```python
import cv2

print(cv2.getBuildInformation())
```

### ❌ Non-recommended ❌

- **Do not install Anaconda**
- **Do not reinstall OpenCV**
  - **It is already installed with Jetson SDK** 
- **Do not install PyTorch using ```pip3 install pytorch```**

## Performance
All performance was measured on the Jetson Orin(8GB). The input size means the size of the resized tensor that goes into the model, <ins>not the resolution of the camera</ins>. **<ins>Larger models(Base & Large) are not supported on Jetson Orin due to memory issues.</ins>** 
| Model | Input size | Inference Time | Memory Usage |
|:-:|:-:|:-:|:-:|
| Depth-Anything-Small | 308x308 | 23.5ms | 626MB |
| Depth-Anything-Small | 364x364 | 39.2ms | 640MB |
| Depth-Anything-Small | 406x406 | 47.7ms | 649MB |
| Depth-Anything-Small | 518x518 | 98.0ms | 689MB |

**All of weights files are available [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints).**
## Usage

### Code
**You can run the code in just a few lines.**
- **Export ONNX & TensorRT Engine(<ins>Network required</ins>)**
  - ```input_size``` must <ins>be divisible by 14</ins>.
```python
# export.py
export(
    weights_path="LiheYoung/depth_anything_vits14", # local hub or online
    save_dir="weights", # folder name
    input_size=364, # 308 | 364 | 406 | 518
)
```
**✨ No networks are required from now**
- **Camera Streaming**
```python
from camera import Camera

camera = Camera(sensor_id=0, save=True)
camera.run()
```

- **Depth Estimation**
  - ```input_size``` must <ins>be divisible by 14</ins>.
```python
from depth import Depth

depth = DepthEngine(
    input_size=308
    frame_rate=15,
    stream=True, 
)
depth.run()
```
  
### Running 
- **Convert weights(.pt/.pth) to TensorRT Engine(.trt)**
```bash
python3 export.py
```

- **Real-time depth estimation**
```bash
# Streaming the results 
python3 depth.py --stream

# Streaming and Visualizing the depth map to grayscale
python3 depth.py --stream --grayscale

# Streaming and Saving the depth map
python3 depth.py --stream --save 

# Using only raw depth map(float type)
python3 depth.py --raw

# Recording Results
python3 depth.py --record
```
**Frame rate of recorded video could be unmatched with the camera's frame rate <ins>due to variable inference time</ins>.**

**⚠ Note: Please turn off the stream/save option for faster performance.**

## Dependencies

### Pip
>huggingface-hub:       0.22.2 \
>jetson-stats:          4.2.7 \
>matplotlib:            3.1.2 \
>mpmath:                1.3.0 \
>numpy:                 1.24.4 \
>onnx:                  1.16.0 \
>Pillow:                7.0.0 \
>pip:                   24.0 \
>pycuda:                2024.1 \
>scipy:                 1.3.3 \
>sympy:                 1.12 \
>tensorrt:              8.5.2.2 \
>torch:                 2.0.0a0+8aa34602.nv23.3 \
>torchvision:           0.15.0 \

<details>
<summary>Entire</summary>
<div markdown="1">

>appdirs:               1.4.4\
>apt-clone:             0.2.1\
>apturl:                0.5.2\
>bcrypt:                3.1.7\
>blinker:               1.4\
>Brlapi:                0.7.0\
>certifi:               2019.11.28\
>chardet:               3.0.4\
>Click:                 7.0\
>colorama:              0.4.3\
>cryptography:          2.8\
>cupshelpers:           1.0\
>cycler:                0.10.0\
>dbus-python:           1.2.16\
>decorator:             4.4.2\
>defer:                 1.0.6\
>distro:                1.4.0\
>distro-info:           0.23+ubuntu1.1\
>duplicity:             0.8.12.0\
>entrypoints:           0.3\
>fasteners:             0.14.1\
>filelock:              3.13.3\
>fsspec:                2024.3.1\
>future:                0.18.2\
>graphsurgeon:          0.4.6\
>httplib2:              0.14.0\
>huggingface-hub:       0.22.2\
>idna:                  2.8\
>Jetson.GPIO:           2.1.6\
>jetson-stats:          4.2.7\
>keyring:               18.0.1\
>kiwisolver:            1.0.1\
>language-selector:     0.1\
>launchpadlib:          1.10.13\
>lazr.restfulclient:    0.14.2\
>lazr.uri:              1.0.3\
>lockfile:              0.12.2\
>louis:                 3.12.0\
>macaroonbakery:        1.3.1\
>Mako:                  1.1.0\
>MarkupSafe:            1.1.0\
>matplotlib:            3.1.2\
>monotonic:             1.5\
>mpmath:                1.3.0\
>networkx:              3.1\
>numpy:                 1.24.4\
>oauthlib:              3.1.0\
>olefile:               0.46\
>onboard:               1.4.1\
>onnx:                  1.16.0\
>onnx-graphsurgeon:     0.3.12\
>packaging:             24.0\
>PAM:                   0.4.2\
>pandas:                0.25.3\
>paramiko:              2.6.0\
>pexpect:               4.6.0\
>Pillow:                7.0.0\
>pip:                   24.0\
>platformdirs:          4.2.0\
>protobuf:              5.26.1\
>pycairo:               1.16.2\
>pycrypto:              2.6.1\
>pycuda:                2024.1\
>pycups:                1.9.73\
>PyGObject:             3.36.0\
>PyICU:                 2.4.2\
>PyJWT:                 1.7.1\
>pymacaroons:           0.13.0\
>PyNaCl:                1.3.0\
>pyparsing:             2.4.6\
>pyRFC3339:             1.1\
>python-apt:            2.0.1+ubuntu0.20.4.1\
>python-dateutil:       2.7.3\
>python-dbusmock:       0.19\
>python-debian:         0.1.36+ubuntu1.1\
>pytools:               2024.1.1\
>pytz:                  2019.3\
>pyxdg:                 0.26\
>PyYAML:                5.3.1\
>requests:              2.22.0\
>requests-unixsocket:   0.2.0\
>scipy:                 1.3.3\
>SecretStorage:         2.3.1\
>setuptools:            45.2.0\
>simplejson:            3.16.0\
>six:                   1.14.0\
>smbus2:                0.4.3\
>sympy:                 1.12\
>systemd-python:        234\
>tensorrt:              8.5.2.2\
>torch:                 2.0.0a0+8aa34602.nv23.3\
>torchvision:           0.15.0\
>tqdm:                  4.66.2\
>typing_extensions:     4.10.0\
>ubuntu-drivers-common: 0.0.0\
>ubuntu-pro-client:     8001\
>uff:                   0.6.9\
>urllib3:               1.25.8\
>urwid:                 2.0.1\
>wadllib:               1.3.3\
>wheel:                 0.34.2\
>xkit:                 0.0.0\
</div>
</details>

**⚠ When installing Python packages that depend on opencv, please be cautious or do not install them.**
## Issues
**Known solutions for troubleshootings**
### Gstreamer

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0
```

### OpenCV
**Nothing here yet.**

## References
**✨ Special thanks to these amaizing projects:**
- [DepthAnything](https://github.com/LiheYoung/Depth-Anything)
- [depth-anything-tensorrt](https://github.com/spacewalk01/depth-anything-tensorrt)
- [DepthAnythingTensorrtDeploy](https://github.com/thinvy/DepthAnythingTensorrtDeploy)