from __future__ import annotations
from typing import Sequence
import logging
import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np
import PySpin

logging.getLogger().setLevel(logging.INFO)


def _convert_image_to_bgr(image_result: PySpin.ImagePtr) -> np.ndarray:
    """
    Convert a PySpin image to an OpenCV BGR image.

    Args:
        image_result (PySpin.ImagePtr): Captured image from FLIR camera.

    Returns:
        frame_bgr (np.ndarray): BGR image ready for OpenCV.
    """
    img = image_result.GetNDArray()
    pixel_format = image_result.GetPixelFormat()

    # Try to get symbolic pixel format name (e.g. "RGB8", "BayerRG8", ...)
    fmt_name = None
    try:
        fmt_name = image_result.GetPixelFormatName()
    except AttributeError:
        pass

    if fmt_name is not None:
        fmt_name_upper = fmt_name.upper()
        # Uncomment once if you want to inspect:
        # print(f"[DEBUG] PixelFormat name: {fmt_name}, enum value: {pixel_format}")

        # --- Mono formats ---
        if "MONO8" in fmt_name_upper:
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # --- RGB formats (your camera: "RGB8") ---
        elif "RGB8" in fmt_name_upper:
            # Image is in RGB order -> convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # --- BGR formats ---
        elif "BGR8" in fmt_name_upper:
            frame_bgr = img

        # --- Bayer formats ---
        elif "BAYERRG8" in fmt_name_upper:
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)
        elif "BAYERGB8" in fmt_name_upper:
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
        elif "BAYERGR8" in fmt_name_upper:
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)
        elif "BAYERBG8" in fmt_name_upper:
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

        # --- Unknown but named format ---
        else:
            print(f"[WARN] Unhandled PixelFormat name '{fmt_name}'. Using generic conversion.")
            if img.ndim == 2:
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    else:
        # No name API, fallback by shape
        print(f"[WARN] PixelFormat ({pixel_format}) without name. Using generic conversion.")
        if img.ndim == 2:
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return frame_bgr


def _set_pixel_format(cam: PySpin.CameraPtr) -> None:
    """
    Set camera PixelFormat to a readable color format.

    Args:
        cam (PySpin.CameraPtr): Initialized camera object.
    """
    nodemap = cam.GetNodeMap()
    pixel_format_enum = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if not PySpin.IsWritable(pixel_format_enum):
        print("[WARN] Cannot write PixelFormat. Using current format.")
        return

    # Prefer named RGB format if available, then BGR8, then Mono8
    for fmt_name in ["RGB8", "RGB8Packed", "BGR8", "Mono8"]:
        entry = pixel_format_enum.GetEntryByName(fmt_name)
        if PySpin.IsReadable(entry):
            pixel_format_enum.SetIntValue(entry.GetValue())
            print(f"[INFO] PixelFormat set to {fmt_name}")
            return

    print("[WARN] No preferred PixelFormat found. Using default.")


def _configure_resolution_fps(
    cam: PySpin.CameraPtr,
    width: int,
    height: int,
    fps: float,
) -> None:
    """
    Configure resolution and frame rate for FLIR camera.

    Args:
        cam (PySpin.CameraPtr): Initialized camera object.
        width (int): Desired image width in pixels.
        height (int): Desired image height in pixels.
        fps (float): Desired frame rate.
    """
    nodemap = cam.GetNodeMap()

    # Width / Height
    width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
    height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))

    if PySpin.IsWritable(width_node) and PySpin.IsWritable(height_node):
        max_w = width_node.GetMax()
        max_h = height_node.GetMax()

        width = min(width, max_w)
        height = min(height, max_h)

        if width % 2 == 1:
            width -= 1
        if height % 2 == 1:
            height -= 1

        width_node.SetValue(width)
        height_node.SetValue(height)
        print(f"[INFO] Resolution set to {width} x {height}")
    else:
        print("[WARN] Width/Height not writable")

    # Frame rate
    enable_fps_node = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    if PySpin.IsWritable(enable_fps_node):
        enable_fps_node.SetValue(True)

    fps_node = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if PySpin.IsWritable(fps_node):
        max_fps = fps_node.GetMax()
        fps = min(fps, max_fps)
        fps_node.SetValue(fps)
        print(f"[INFO] Frame rate set to {fps:.2f} FPS")
    else:
        print("[WARN] Cannot set AcquisitionFrameRate.")

    current_width = width_node.GetValue() if PySpin.IsReadable(width_node) else -1
    current_height = height_node.GetValue() if PySpin.IsReadable(height_node) else -1
    current_fps = fps_node.GetValue() if PySpin.IsReadable(fps_node) else -1.0
    print(f"[CHECK] Applied resolution: {current_width} x {current_height}, FPS: {current_fps:.2f}")


class _SpinnakerCapture:
    """
    Thin wrapper that mimics cv2.VideoCapture.read() using PySpin.

    This allows DepthEngine to continue using:
        ret, frame = self.camera.cap[0].read()
    """

    def __init__(self, cam: PySpin.CameraPtr):
        """
        Args:
            cam (PySpin.CameraPtr): Initialized and acquisition-started camera.
        """
        self.cam = cam

    def isOpened(self) -> bool:
        """
        Returns:
            opened (bool): Always True if camera is initialized.
        """
        return True

    def read(self, timeout_ms: int = 1000) -> tuple[bool, np.ndarray]:
        """
        Grab the next frame from the camera.

        Args:
            timeout_ms (int): Timeout in milliseconds when waiting for image.

        Returns:
            ret (bool): True if frame was captured successfully, False otherwise.
            frame (np.ndarray): BGR image (if ret is True) or empty array.
        """
        try:
            image_result = self.cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException as e:
            print(f"[ERROR] GetNextImage failed: {e}")
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        if image_result.IsIncomplete():
            print(f"[WARN] Incomplete image. Status: {image_result.GetImageStatus()}")
            image_result.Release()
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        frame_bgr = _convert_image_to_bgr(image_result)
        image_result.Release()
        return True, frame_bgr


class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        width: int = 308,
        height: int = 308,
        _width: int = 308,
        _height: int = 308,
        frame_rate: int = 30,
        flip_method: int = 0,
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = False,
        save: bool = False,
        log: bool = True,
    ) -> None:
        """
        FLIR camera wrapper compatible with the original GStreamer-based Camera API.

        Args:
            sensor_id (int | Sequence[int]): Ignored for FLIR GigE (first camera is used).
            width (int): Display width (not strictly used in FLIR pipeline).
            height (int): Display height (not strictly used in FLIR pipeline).
            _width (int): Capture width from camera sensor (resized target).
            _height (int): Capture height from camera sensor (resized target).
            frame_rate (int): Target frame rate.
            flip_method (int): Not used for FLIR (kept for compatibility).
            window_title (str): Window title when streaming.
            save_path (str): Directory to save frames when save=True.
            stream (bool): Whether to show a realtime window (for Camera.run()).
            save (bool): Whether to save raw camera frames (for Camera.run()).
            log (bool): Whether to log FPS in Camera.run().
        """
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self._width = _width
        self._height = _height
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log

        # Only single camera is supported
        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence) and len(sensor_id) > 1:
            raise NotImplementedError("Multiple cameras are not supported yet")

        # Initialize Spinnaker system and camera
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError("No FLIR cameras found")

        # Use the first camera
        self.cam: PySpin.CameraPtr = self.cam_list[0]
        self.cam.Init()

        # Configure pixel format, resolution, and FPS
        _set_pixel_format(self.cam)
        _configure_resolution_fps(self.cam, width=self._width, height=self._height, fps=self.frame_rate)

        # Start acquisition in Continuous mode
        nodemap = self.cam.GetNodeMap()
        acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsWritable(acquisition_mode):
            cont_entry = acquisition_mode.GetEntryByName("Continuous")
            acquisition_mode.SetIntValue(cont_entry.GetValue())
        self.cam.BeginAcquisition()
        print("[INFO] FLIR camera acquisition started (Continuous mode).")

        # Wrap camera with cv2.VideoCapture-like API
        self.cap = [_SpinnakerCapture(self.cam)]

        # Prepare recording directory if needed
        if save:
            os.makedirs(self.save_path, exist_ok=True)
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)
            logging.info(f"Save directory: {self.save_path}")

    def run(self) -> None:
        """
        Simple streaming loop for testing camera only.
        """
        if not self.cap[0].isOpened():
            print("[ERROR] Camera is not opened.")
            return

        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)

        try:
            while True:
                t0 = time.time()
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')

                ret, frame = self.cap[0].read()
                if not ret:
                    print("[WARN] Failed to grab frame.")
                    continue

                if self.save:
                    cv2.imwrite(str(self.save_path / f"{timestamp}.jpg"), frame)

                if self.log:
                    print(f"FPS: {1 / max(time.time() - t0, 1e-6):.2f}")

                if self.stream:
                    cv2.imshow(self.window_title, frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

        except Exception as e:
            print(e)
        finally:
            self.release()

    def release(self) -> None:
        """
        Release camera and Spinnaker resources.
        """
        try:
            self.cam.EndAcquisition()
        except PySpin.SpinnakerException:
            pass

        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.system.ReleaseInstance()
        cv2.destroyAllWindows()
        print("[INFO] FLIR camera and resources released.")

    @property
    def frame(self) -> np.ndarray:
        """
        Return a single frame (BGR) from the camera.

        Returns:
            frame (np.ndarray): BGR frame or black image if failed.
        """
        if self.cap[0].isOpened():
            ret, frame = self.cap[0].read()
            if ret:
                return frame

        return np.zeros((self._height, self._width, 3), dtype=np.uint8)


if __name__ == "__main__":
    cam = Camera(sensor_id=0, stream=True)
    cam.run()
