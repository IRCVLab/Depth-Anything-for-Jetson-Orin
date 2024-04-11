from __future__ import annotations
from typing import Sequence

import logging

import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

# from jtop import jtop # Use this to monitor compute usage (for Jetson Nano)

logging.getLogger().setLevel(logging.INFO)

class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 1920,
        height: int = 1080,
        _width: int = 960,
        _height: int = 540,
        frame_rate: int = 30,
        flip_method: int = 0,
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = False,
        save: bool = False,
        log: bool = True,
    ) -> None:
        """
        sensor_id: int | Sequence[int] -> Camera sensor id
        width: int -> Width of the display window
        height: int -> Height of the display window
        _width: int -> Width of the camera frame
        _height: int -> Height of the camera frame
        frame_rate: int -> Frame rate of the camera
        flip_method: int -> Flip method of the camera
        window_title: str -> Title of the display window
        save_path: str -> Path to save the results
        stream: bool -> Stream the camera feed
        save: bool -> Save the camera feed
        log: bool -> Log informations
        """
        self.sensor_id = sensor_id
        self.width = width # width of the display window
        self.height = height # height of the display window
        self._width = _width # width of the camera frame
        self._height = _height # height of the camera frame
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log
        
        # Check if OpenCV is built with GStreamer support
        # print(cv2.getBuildInformation())
        
        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence) and len(sensor_id) > 1:
            raise NotImplementedError("Multiple cameras are not supported yet")
    
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        self.cap = [cv2.VideoCapture(self.gstreamer_pipeline(sensor_id=id, flip_method=0), \
                    cv2.CAP_GSTREAMER) for id in self.sensor_id]
        
        # Make record directory
        if save:
            assert save_path is not None, "Please provide a save path"
            os.makedirs(self.save_path, exist_ok=True) # if path does not exist, create it
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)
            
            logging.info(f"Save directory: {self.save_path}")
    
    def gstreamer_pipeline(self, sensor_id: int, flip_method: int) -> str:
        """
        Return a GStreamer pipeline for capturing from the CSI camera
        """
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                self.width,
                self.height,
                self.frame_rate,
                flip_method,
                self._width,
                self._height,
            )
        )
    
    def run(self) -> None:
        """
        Streaming camera feed
        """
        if self.cap[0].isOpened():
            try:
                while True:
                    t0 = time.time()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    _, frame = self.cap[0].read()
                    
                    if self.save: 
                        cv2.imwrite(str(self.save_path / f"{timestamp}.jpg"), frame)
                        
                    if self.log:
                        print(f"FPS: {1 / (time.time() - t0):.2f}")
                    
                    if self.stream:
                        cv2.imshow('Camera', frame)
                        
                        if cv2.waitKey(1) == ord('q'):
                            break
                    
            except Exception as e:
                print(e)
            finally:
                self.cap[0].release()
                cv2.destroyAllWindows()
        
    @property
    def frame(self) -> np.ndarray:
        """
        !!! Important: This method is not efficient for real-time rendering !!!
        
        [Example Usage]
        ...
        frame = cam.frame # Get the current frame from camera
        cv2.imshow('Camera', frame)
        ...
        
        """
        if self.cap[0].isOpened():
            return self.cap[0].read()[1]
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
if __name__ == '__main__':
    cam = Camera(sensor_id=0, stream=True)
    cam.run()
