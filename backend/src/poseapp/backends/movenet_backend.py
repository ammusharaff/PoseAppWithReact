import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Tuple

# Try to import config to get model path; fallback to the known path:
try:
    from .. import config as _config
    DEFAULT_MODEL_PATH = getattr(_config, "MODEL_PATH", None)
except Exception:
    DEFAULT_MODEL_PATH = None

if not DEFAULT_MODEL_PATH:
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "movenet_singlepose_lightning.tflite")
    DEFAULT_MODEL_PATH = os.path.normpath(DEFAULT_MODEL_PATH)


class MoveNetTFLiteBackend:
    """
    MoveNet TFLite backend wrapper.

    Usage:
        backend = MoveNetTFLiteBackend(model_path=..., variant="lightning")
        kps, meta = backend.infer(frame_bgr)   # frame from OpenCV BGR
    Returns:
        kps: list of 17 keypoints as dicts: {"x": float, "y": float, "score": float}
             x,y are normalized to the input frame (0..1)
        meta: dict with raw outputs and model information
    """

    def __init__(self, model_path: str = None, variant: str = None, **kwargs):
        # accept variant and ignore unknown kwargs for backwards compatibility
        self.variant = variant
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MoveNet model not found at {self.model_path}")
        # load interpreter
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        # cache input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # assume single input for MoveNet
        self.inp = self.input_details[0]
        self.inp_index = self.inp["index"]
        self.inp_shape = self.inp["shape"]  # typically [1, H, W, 3]
        self.inp_dtype = self.inp["dtype"]
        self.inp_quant = self.inp.get("quantization", (0.0, 0))
        # outputs info (useful for postprocessing)
        self.out_details = self.output_details

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess the OpenCV BGR frame according to interpreter input requirements.
        Returns a numpy array ready to be fed into interpreter.set_tensor.
        """
        # target shape
        _, target_h, target_w, target_c = [int(x) for x in self.inp_shape]
        # resize
        img = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # convert BGR->RGB (MoveNet expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Handle dtype expectations
        if np.issubdtype(self.inp_dtype, np.floating):
            # float model: normalize to 0..1
            arr = img.astype(np.float32) / 255.0
        elif np.issubdtype(self.inp_dtype, np.integer):
            # integer model: if quantized (scale, zero_point), quantize accordingly
            scale, zero_point = self.inp_quant if self.inp_quant is not None else (0.0, 0)
            if scale and scale != 0.0:
                f = img.astype(np.float32) / 255.0
                q = np.round(f / scale).astype(self.inp_dtype) + np.int32(zero_point)
                arr = q.astype(self.inp_dtype)
            else:
                # fallback: direct cast to integer range 0..255
                arr = img.astype(self.inp_dtype)
        else:
            arr = img.astype(np.float32) / 255.0

        # Add batch dim if required
        if arr.ndim == 3:
            arr = np.expand_dims(arr, axis=0)
        return arr

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
      """
      Run inference on a single BGR frame and return (kps, meta).
      - kps: list of 17 keypoints as dicts {'name','x','y','nx','ny','score'} where
             x,y are pixel coords on the provided frame, and nx,ny are normalized [0..1].
      - meta: raw outputs and diagnostic info
      """
      inp_array = self._preprocess(frame_bgr)
      expected_dtype = np.dtype(self.inp_dtype)
      if inp_array.dtype != expected_dtype:
          try:
              inp_array = inp_array.astype(expected_dtype)
          except Exception:
              raise RuntimeError(
                  f"Failed to cast input array to expected dtype {expected_dtype}. Got dtype {inp_array.dtype} and shape {inp_array.shape}"
              )

      # set tensor and run
      try:
          self.interpreter.set_tensor(self.inp_index, inp_array)
      except Exception as e:
          raise RuntimeError(f"interpreter.set_tensor failed: expected dtype {self.inp_dtype}, "
                             f"expected shape {self.inp_shape}, provided array shape {inp_array.shape}, error: {e}")
      self.interpreter.invoke()

      # collect raw outputs
      raw_outputs = [self.interpreter.get_tensor(out["index"]) for out in self.out_details]

      # Prepare kp_array similar to MoveNet output [1,1,17,3] -> (17,3)
      kp_array = None
      if len(raw_outputs) > 0:
          out0 = raw_outputs[0]
          if out0.ndim == 4:
              kp_array = out0[0, 0, :, :]   # (17,3)
          elif out0.ndim == 3:
              kp_array = out0[0, :, :]
          else:
              try:
                  kp_array = out0.reshape(-1, 3)
                  if kp_array.shape[0] != 17:
                      kp_array = None
              except Exception:
                  kp_array = None

      if kp_array is None:
          return [], {"raw": raw_outputs}

      # Map indices to MoveNet keypoint names (standard order)
      keypoint_names = [
          "nose","left_eye","right_eye","left_ear","right_ear",
          "left_shoulder","right_shoulder","left_elbow","right_elbow",
          "left_wrist","right_wrist","left_hip","right_hip",
          "left_knee","right_knee","left_ankle","right_ankle"
      ]

      # Frame size (pixel coords)
      h_frame, w_frame = frame_bgr.shape[0], frame_bgr.shape[1]

      kps_list: List[Dict[str, float]] = []
      # MoveNet outputs are typically (y, x, score) normalized in [0,1]
      for i in range(min(kp_array.shape[0], len(keypoint_names))):
          yp = float(kp_array[i, 0])
          xp = float(kp_array[i, 1])
          score = float(kp_array[i, 2])

          # clamp normalized coords
          nx = max(0.0, min(1.0, xp))
          ny = max(0.0, min(1.0, yp))

          # convert to pixel coordinates relative to the actual frame used for inference
          # Note: we resized the frame to model input in _preprocess. Use original frame dimensions
          # to map normalized coords to pixels for overlay/angle calculations.
          x_px = nx * float(w_frame)
          y_px = ny * float(h_frame)

          kdict = {
              "name": keypoint_names[i],
              "x": x_px,
              "y": y_px,
              "nx": nx,
              "ny": ny,
              "score": score
          }
          kps_list.append(kdict)

      meta = {
          "raw": raw_outputs,
          "model_path": self.model_path,
          "input_shape": tuple(self.inp_shape),
          "dtype": str(self.inp_dtype)
      }
      return kps_list, meta

    def get_model_io_info(self) -> Dict[str, Any]:
        return {
            "input": self.inp,
            "outputs": self.out_details
        }

    def close(self):
        # release resources if needed; tf.lite.Interpreter doesn't need explicit close,
        # but keep the method to follow expected backend API.
        try:
            del self.interpreter
        except Exception:
            pass


# Backwards compatibility alias:
# Older code imports `MoveNetBackend` — make that name point to our class.
try:
    MoveNetBackend
except NameError:
    if "MoveNetTFLiteBackend" in globals():
        MoveNetBackend = MoveNetTFLiteBackend


# Example quick test helper (not executed by default)
if __name__ == "__main__":
    mb = MoveNetTFLiteBackend()
    print("Loaded model:", mb.model_path)
    print("Input:", mb.inp)
    print("Output count:", len(mb.out_details))
