# depth.py
from __future__ import annotations
from typing import Sequence, Optional

import argparse
import logging
import os
import sys
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

# Camera 래퍼는 기존 것을 사용
from camera import Camera

logging.getLogger().setLevel(logging.INFO)


# ---------- 유틸: 항상 imshow 가능한 BGR8로 ----------
def ensure_bgr8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.dtype in (np.float32, np.float64):
            im = img
            if im.max() <= 1.0:
                im = im * 255.0
            else:
                im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
            img = im.astype(np.uint8)
        elif img.dtype == np.uint16:
            img = cv2.convertScaleAbs(img, alpha=(255.0 / 65535.0))
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]
        return img
    img = np.squeeze(img)
    return ensure_bgr8(img)


class DepthApp:
    """
    GUI 우선 → 카메라 → (옵션) TensorRT 순으로 초기화하는 실시간 Depth 추정 앱
    - 입력/출력 텐서는 NCHW float32 (1,3,H,W) / (1,1,H,W) 기준
    - 엔진 바인딩 이름: 'input' / 'output' 를 우선 사용, 실패 시 첫 input/output으로 fallback
    """

    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        input_size: int = 308,
        frame_rate: int = 15,
        engine_path: str = "weights/depth_anything_vits14_308.trt",
        save_path: Optional[str] = None,
        grayscale: bool = False,
        stream: bool = False,          # 헤드리스 (GUI OFF)
        lazy_trt: bool = True,         # True면 첫 추론 시 TRT/pycuda 초기화
    ):
        # 매개변수만 저장 (무거운 초기화는 run()에서)
        self.sensor_id = sensor_id
        self.width = int(input_size)
        self.height = int(input_size)
        self.frame_rate = frame_rate
        self.engine_path = engine_path
        self.save_path = Path(save_path) if isinstance(save_path, str) else Path("results")
        self.grayscale = grayscale
        self.stream = stream
        self.lazy_trt = lazy_trt

        # 런타임 상태
        self.headless = (os.environ.get("DISPLAY", "") == "")
        self.force_gui = False
        self.cam_only = False
        self.record = False
        self.save = False

        # 런타임 변수(나중 초기화)
        self.cam: Optional[Camera] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

        # TRT 관련 (지연 초기화)
        self.trt_loaded = False
        self.trt = None
        self.cuda = None
        self.runtime = None
        self.engine = None
        self.context = None
        self.input_idx = None
        self.output_idx = None
        self.h_input = None
        self.h_output = None
        self.d_input = None
        self.d_output = None
        self.cuda_stream = None

        # 비디오 레코더 (run에서 초기화)
        self.video = None

    # ---------- 전처리: 명시적 NCHW float32 ----------
    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        x = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))                       # HWC → CHW
        x = np.ascontiguousarray(x[None, :, :, :], np.float32)  # (1,C,H,W)
        return x

    # ---------- 후처리: min/max로 normalize 후 colormap ----------
    def postprocess(self, depth_flat: np.ndarray) -> np.ndarray:
        depth = depth_flat.reshape(self.height, self.width)
        depth = cv2.resize(depth, (self._width, self._height), interpolation=cv2.INTER_LINEAR)

        if self.grayscale:
            # 단순 그레이스케일 normalize
            dmin, dmax = float(depth.min()), float(depth.max())
            if dmax - dmin < 1e-6:
                g = np.zeros_like(depth, np.uint8)
            else:
                g = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        # 컬러맵
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax - dmin < 1e-6:
            g = np.zeros_like(depth, np.uint8)
        else:
            g = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)

    # ---------- TensorRT/pycuda 초기화 ----------
    def _init_trt(self):
        if self.trt_loaded:
            return
        print("[INIT] Loading TensorRT / CUDA context ...")
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401  # CUDA context 생성
        import pycuda.driver as cuda

        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        print(f"[TRT] Engine loaded: {self.engine_path}")

        # 바인딩 인덱스: 이름 우선
        try:
            self.input_idx = self.engine.get_binding_index("input")
            self.output_idx = self.engine.get_binding_index("output")
        except Exception:
            # fallback: 첫 input / 첫 output
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i) and self.input_idx is None:
                    self.input_idx = i
                if (not self.engine.binding_is_input(i)) and self.output_idx is None:
                    self.output_idx = i
        if self.input_idx is None or self.output_idx is None:
            raise RuntimeError("TensorRT bindings not found")

        # 정적 엔진 가정 (이미 1x3xHxW, 1x1xHxW로 만들어진 상태)
        in_shape = self.engine.get_binding_shape(self.input_idx)
        out_shape = self.engine.get_binding_shape(self.output_idx)
        print(f"[TRT] bindings: input shape={tuple(in_shape)}, output shape={tuple(out_shape)}")

        # Host/Device 메모리
        self.h_input = self.cuda.pagelocked_empty(trt.volume((1, 3, self.height, self.width)), dtype=np.float32)
        self.h_output = self.cuda.pagelocked_empty(trt.volume((1, 1, self.height, self.width)), dtype=np.float32)
        self.d_input = self.cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = self.cuda.mem_alloc(self.h_output.nbytes)
        self.cuda_stream = self.cuda.Stream()
        self.trt_loaded = True
        print("[INIT] TensorRT ready.")

    # ---------- 추론 ----------
    def infer(self, bgr: np.ndarray) -> np.ndarray:
        if not self.trt_loaded:
            self._init_trt()

        x = self.preprocess(bgr)  # (1,3,H,W) float32
        # H2D
        np.copyto(self.h_input, x.ravel())
        self.cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)

        # execute
        bindings = [None] * self.engine.num_bindings
        bindings[self.input_idx] = int(self.d_input)
        bindings[self.output_idx] = int(self.d_output)

        ok = self.context.execute_async_v2(bindings=bindings, stream_handle=self.cuda_stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        # D2H
        self.cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()

        # 후처리
        depth_vis = self.postprocess(np.asarray(self.h_output, dtype=np.float32))
        return depth_vis

    # ---------- 실행 ----------
    def run(self):
        print(f"[ENV] DISPLAY={os.environ.get('DISPLAY')}  XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE')}")
        print(f"[FLAGS] stream={self.stream} headless={self.headless} force_gui={self.force_gui} cam_only={self.cam_only} lazy_trt={self.lazy_trt}")

        use_gui = (not self.stream) and (not self.headless)
        if self.force_gui:
            use_gui = True

        # 1) GUI 스레드/창 먼저 확보
        if use_gui:
            try:
                cv2.startWindowThread()
            except Exception:
                pass
            cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth", 1280, 720)
            cv2.moveWindow("Depth", 40, 40)
            # 스모크 표출
            smoke = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(smoke, "SMOKE GUI", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            for _ in range(20):
                cv2.imshow("Depth", smoke)
                cv2.waitKey(20)

        # 2) 이제 카메라 시작
        self.cam = Camera(sensor_id=self.sensor_id, frame_rate=self.frame_rate)
        self._width, self._height = self.cam._width, self.cam._height
        print(f"[CAM] started: {self._width} x {self._height} @ {self.frame_rate} FPS")

        # 3) 저장/레코딩 준비
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            sub = self.save_path / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sub.mkdir(parents=True, exist_ok=True)
            self.save_path = sub
            print(f"[SAVE] dir = {self.save_path}")
        if self.record:
            self.video = cv2.VideoWriter(
                "results.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                self.frame_rate, (2 * self._width, self._height)
            )

        # 4) 메인 루프
        try:
            while True:
                ret, frame = self.cam.cap[0].read()
                if not ret or frame is None or frame.size == 0:
                    print("[WARN] camera read failed"); time.sleep(0.01); continue

                frame = ensure_bgr8(frame)

                if self.cam_only:
                    right = np.zeros_like(frame)
                else:
                    # (필요 시 lazy_trt로 첫 infer에서 초기화)
                    right = self.infer(frame)
                    right = ensure_bgr8(right)

                out = np.concatenate((frame, right), axis=1)
                if out.dtype != np.uint8:
                    out = ensure_bgr8(out)
                if not out.flags.c_contiguous:
                    out = np.ascontiguousarray(out)

                # 화면 맞춤
                h, w = out.shape[:2]
                if w > 1920 or h > 1080:
                    s = min(1920 / w, 1080 / h)
                    out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

                if self.record:
                    self.video.write(out)
                if self.save:
                    fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
                    cv2.imwrite(str(self.save_path / fname), out)

                if use_gui:
                    cv2.imshow("Depth", out)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] 'q' pressed"); break
                else:
                    # 헤드리스 진행 표시
                    print("[LOOP] frame ok (no-GUI)")

        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt")
        finally:
            try:
                if self.video is not None:
                    self.video.release()
                if use_gui:
                    cv2.destroyAllWindows()
                if self.cam is not None:
                    self.cam.release()
            except Exception as e:
                print("[WARN] cleanup:", e)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--input-size", type=int, default=308)
    ap.add_argument("--frame-rate", type=int, default=15)
    ap.add_argument("--engine", type=str, default="weights/depth_anything_vits14_308.trt")
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--stream", action="store_true", help="Headless (no GUI)")
    ap.add_argument("--force-gui", action="store_true", help="Force GUI even if headless detected")
    ap.add_argument("--cam-only", action="store_true", help="Skip TensorRT inference")
    ap.add_argument("--lazy-trt", action="store_true", default=True, help="Defer TRT init until first inference")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    app = DepthApp(
        sensor_id=args.sensor_id,
        input_size=args.input_size,
        frame_rate=args.frame_rate,
        engine_path=args.engine,
        grayscale=args.grayscale,
        stream=args.stream,
        lazy_trt=args.lazy_trt,
    )
    app.force_gui = args.force_gui
    app.cam_only = args.cam_only
    app.record = args.record
    app.save = args.save
    app.run()


if __name__ == "__main__":
    print(f"[ARGS] sys.argv={sys.argv}")
    print(f"[ENV] DISPLAY={os.environ.get('DISPLAY')}  XDG_SESSION_TYPE={os.environ.get('XDG_SESSION_TYPE')}")
    main()