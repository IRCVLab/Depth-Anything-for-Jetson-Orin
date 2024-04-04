from __future__ import annotations
from typing import Sequence

import argparse

import logging

import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

import tensorrt as trt
import pycuda.autoinit # Don't remove this line
import pycuda.driver as cuda
from torchvision.transforms import Compose

from camera import Camera
from depth_anything import transform


class DepthEngine:
    """
    Real-time depth estimation using Depth Anything with TensorRT
    """
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        input_size: int = 308,
        frame_rate: int = 15,
        trt_engine_path: str = 'weights/depth_anything_vits14_308.trt', # Must match with the input_size
        save_path: str = None,
        raw: bool = False,
        stream: bool = False,
        record: bool = False,
        save: bool = False,
        grayscale: bool = False,
    ):
        """
        sensor_id: int | Sequence[int] -> Camera sensor id
        input_size: int -> Width and height of the input tensor(e.g. 308, 364, 406, 518)
        frame_rate: int -> Frame rate of the camera(depending on inference time)
        trt_engine_path: str -> Path to the TensorRT engine
        save_path: str -> Path to save the results
        raw: bool -> Use only the raw depth map
        stream: bool -> Stream the results
        save: bool -> Save the results
        grayscale: bool -> Convert the depth map to grayscale
        """
        # Initialize the camera
        self.camera = Camera(sensor_id=sensor_id, frame_rate=frame_rate)
        self.width = input_size
        self.height = input_size
        self._width = self.camera._width
        self._height = self.camera._height
        self.save_path = Path(save_path) if isinstance(save_path, str) else Path("results")
        self.raw = raw
        self.stream = stream
        self.record = record
        self.save = save
        self.grayscale = grayscale
        
        # Initialize the raw data
        # Depth map without any postprocessing -> float32
        # For visualization, change raw to False
        if raw: self.raw_depth = None 
        
        # Load the TensorRT engine
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        self.engine = self.runtime.deserialize_cuda_engine(open(trt_engine_path, 'rb').read())
        self.context = self.engine.create_execution_context()
        print(f"Engine loaded from {trt_engine_path}")
        
        # Allocate pagelocked memory
        self.h_input = cuda.pagelocked_empty(trt.volume((1, 3, self.width, self.height)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume((1, 1, self.width, self.height)), dtype=np.float32)
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Create a cuda stream
        self.cuda_stream = cuda.Stream()
        
        # Transform functions
        self.transform = Compose([
            transform.Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            transform.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transform.PrepareForNet(),
        ])
        
        if record:
            # Recorded video's frame rate could be unmatched with the camera's frame rate due to inference time
            self.video = cv2.VideoWriter(
                'results.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                frame_rate,
                (2 * self._width, self._height),
            )
        
        # Make results directory
        if save:
            os.makedirs(self.save_path, exist_ok=True) # if parent dir does not exist, create it
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image
        """
        image = image.astype(np.float32)
        image /= 255.0
        image = self.transform({'image': image})['image']
        image = image[None]
        
        return image
    
    def postprocess(self, depth: np.ndarray) -> np.ndarray:
        """
        Postprocess the depth map
        """
        depth = np.reshape(depth, (self.width, self.height))
        depth = cv2.resize(depth, (self._width, self._height))
        
        if self.raw:
            return depth # raw depth map
        else:
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if self.grayscale:
                depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            else:
                depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            
        return depth
        
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Infer depth from an image using TensorRT
        """
        # Preprocess the image
        image = self.preprocess(image)
        
        t0 = time.time()
        
        # Copy the input image to the pagelocked memory
        np.copyto(self.h_input, image.ravel())
        
        # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()
        
        print(f"Inference time: {time.time() - t0:.4f}s")
        
        return self.postprocess(self.h_output) # Postprocess the depth map
    
    def run(self):
        """
        Real-time depth estimation
        """
        try:
            while True:
                # frame = self.camera.frame # This causes bad performance
                _, frame = self.camera.cap[0].read()
                depth = self.infer(frame)
                
                if self.raw:
                    self.raw_depth = depth
                else:
                    results = np.concatenate((frame, depth), axis=1)
                    
                    if self.record:
                        self.video.write(results)
                        
                    if self.save:
                        cv2.imwrite(str(self.save_path / f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.png'), results)

                    if self.stream:
                        cv2.imshow('Depth', results) # This causes bad performance
                        
                        if cv2.waitKey(1) == ord('q'):
                            break
        except Exception as e:
            print(e)
        finally:
            if self.record:
                self.video.release()
                
            if self.stream:
                cv2.destroyAllWindows()
            
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--frame_rate', type=int, default=15, help='Frame rate of the camera')
    args.add_argument('--raw', action='store_true', help='Use only the raw depth map')
    args.add_argument('--stream', action='store_true', help='Stream the results')
    args.add_argument('--record', action='store_true', help='Record the results')
    args.add_argument('--save', action='store_true', help='Save the results')
    args.add_argument('--grayscale', action='store_true', help='Convert the depth map to grayscale')
    args = args.parse_args()
    
    depth = DepthEngine(
        frame_rate=args.frame_rate,
        raw=args.raw,
        stream=args.stream, 
        record=args.record,
        save=args.save, 
        grayscale=args.grayscale
    )
    depth.run()
