import base64
import json
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import AugmentationPipeline

app = FastAPI(title="Image Augmentation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUGMENTATION_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "gaussian_noise",
        "title": "Gaussian noise",
        "category": "noise",
        "description": "Adds Gaussian noise to every channel.",
        "params": [
            {"key": "mean", "label": "Mean", "type": "number", "default": 0, "step": 1},
            {"key": "sigma", "label": "Sigma", "type": "number", "default": 25, "min": 1, "max": 120},
        ],
    },
    {
        "name": "impulse_noise",
        "title": "Impulse noise",
        "category": "noise",
        "description": "Salt/pepper/gray noise with configurable distribution.",
        "params": [
            {"key": "noise_percentage", "label": "Noise portion", "type": "number", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
            {"key": "noise_values", "label": "Noise values", "type": "array", "default": [0, 255, 128]},
            {"key": "noise_probs", "label": "Noise probabilities", "type": "array", "default": [0.3, 0.3, 0.4]},
        ],
    },
    {
        "name": "rayleigh_noise",
        "title": "Rayleigh noise",
        "category": "noise",
        "description": "Adds Rayleigh distributed noise.",
        "params": [{"key": "scale", "label": "Scale", "type": "number", "default": 30, "min": 1, "max": 150}],
    },
    {
        "name": "exponential_noise",
        "title": "Exponential noise",
        "category": "noise",
        "description": "Adds exponential noise.",
        "params": [{"key": "scale", "label": "Scale", "type": "number", "default": 15, "min": 1, "max": 120}],
    },
    {
        "name": "average_blur",
        "title": "Average blur",
        "category": "denoise",
        "description": "Mean blur with odd kernel size.",
        "params": [{"key": "kernel_size", "label": "Kernel", "type": "number", "default": 3, "min": 1, "max": 21, "step": 2}],
    },
    {
        "name": "gaussian_blur",
        "title": "Gaussian blur",
        "category": "denoise",
        "description": "Gaussian blur with sigma.",
        "params": [
            {"key": "kernel_size", "label": "Kernel", "type": "number", "default": 3, "min": 1, "max": 31, "step": 2},
            {"key": "sigma", "label": "Sigma", "type": "number", "default": 0, "min": 0, "max": 50},
        ],
    },
    {
        "name": "median_blur",
        "title": "Median blur",
        "category": "denoise",
        "description": "Median blur with odd kernel size.",
        "params": [{"key": "kernel_size", "label": "Kernel", "type": "number", "default": 3, "min": 1, "max": 21, "step": 2}],
    },
    {
        "name": "histogram_equalization",
        "title": "Histogram equalization",
        "category": "tonal",
        "description": "Equalize histogram (V channel for color images).",
        "params": [{"key": "bins", "label": "Bins", "type": "number", "default": 256, "min": 16, "max": 512, "step": 16}],
    },
    {
        "name": "gamma_correction",
        "title": "Gamma correction",
        "category": "tonal",
        "description": "Gamma correction using lookup table.",
        "params": [{"key": "gamma", "label": "Gamma", "type": "number", "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}],
    },
    {
        "name": "rgb_to_gray",
        "title": "RGB to grayscale",
        "category": "color",
        "description": "Convert to grayscale keeping single channel.",
        "params": [],
    },
    {
        "name": "rgb_to_binary",
        "title": "RGB to binary",
        "category": "color",
        "description": "Threshold to binary (OTSU if threshold not set).",
        "params": [{"key": "threshold", "label": "Threshold", "type": "number", "default": None, "min": 0, "max": 255}],
    },
    {
        "name": "color_restoration",
        "title": "Color restoration",
        "category": "color",
        "description": "Recolor grayscale image using reference palette.",
        "params": [
            {"key": "reference_path", "label": "Reference path", "type": "text", "default": None, "placeholder": "data/reference_6.jpg"},
            {"key": "smooth_ksize", "label": "Median smooth", "type": "number", "default": 3, "min": 1, "max": 15, "step": 2},
        ],
    },
    {
        "name": "sobel",
        "title": "Sobel gradient",
        "category": "edges",
        "description": "Sharpen with Sobel gradients.",
        "params": [{"key": "alpha", "label": "Alpha", "type": "number", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}],
    },
    {
        "name": "previtt",
        "title": "Prewitt gradient",
        "category": "edges",
        "description": "Sharpen with Prewitt kernels.",
        "params": [{"key": "alpha", "label": "Alpha", "type": "number", "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}],
    },
    {
        "name": "image_mixing",
        "title": "Image mixing",
        "category": "composite",
        "description": "Blend with random or chessboard patches from dataset.",
        "params": [
            {"key": "mode", "label": "Mode", "type": "select", "options": ["random", "chess"], "default": "random"},
            {"key": "dataset_path", "label": "Dataset path", "type": "text", "default": None, "placeholder": "data/"},
            {"key": "alpha", "label": "Alpha", "type": "number", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
            {"key": "border_thickness", "label": "Feather", "type": "number", "default": 5, "min": 0, "max": 64},
            {"key": "min_size", "label": "Min patch (h,w)", "type": "array", "default": [32, 32]},
            {"key": "max_size", "label": "Max patch (h,w)", "type": "array", "default": [128, 128]},
            {"key": "patch_size", "label": "Chess patch (h,w)", "type": "array", "default": [64, 64]},
        ],
    },
    {
        "name": "auto_mixing",
        "title": "Auto mixing",
        "category": "composite",
        "description": "Pick textured patch from ref dataset and blend into source.",
        "params": [
            {"key": "dataset_path", "label": "Dataset path", "type": "text", "default": None, "placeholder": "data/"},
            {"key": "win_size", "label": "Window (h,w)", "type": "array", "default": [128, 128]},
            {"key": "alpha", "label": "Alpha", "type": "number", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
        ],
    },
    {
        "name": "scaling",
        "title": "Scaling",
        "category": "geometry",
        "description": "Resize with scale factors.",
        "params": [
            {"key": "scale_x", "label": "Scale X", "type": "number", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05},
            {"key": "scale_y", "label": "Scale Y", "type": "number", "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05},
        ],
    },
    {
        "name": "translation",
        "title": "Translation",
        "category": "geometry",
        "description": "Shift image by pixels.",
        "params": [
            {"key": "shift_x", "label": "Shift X", "type": "number", "default": 50, "min": -400, "max": 400},
            {"key": "shift_y", "label": "Shift Y", "type": "number", "default": 0, "min": -400, "max": 400},
        ],
    },
    {
        "name": "rotation",
        "title": "Rotation",
        "category": "geometry",
        "description": "Rotate around image center (deg).",
        "params": [{"key": "angle", "label": "Angle", "type": "number", "default": 45, "min": -180, "max": 180, "step": 1}],
    },
    {
        "name": "glass_effect",
        "title": "Glass effect",
        "category": "geometry",
        "description": "Random pixel displacement.",
        "params": [{"key": "max_dist", "label": "Max distance", "type": "number", "default": 10, "min": 1, "max": 40}],
    },
    {
        "name": "wave1",
        "title": "Wave X(y)",
        "category": "geometry",
        "description": "Horizontal sinusoidal distortion.",
        "params": [
            {"key": "amplitude", "label": "Amplitude", "type": "number", "default": 20, "min": 1, "max": 150},
            {"key": "period", "label": "Period", "type": "number", "default": 60, "min": 4, "max": 400},
        ],
    },
    {
        "name": "wave2",
        "title": "Wave X(x)",
        "category": "geometry",
        "description": "Vertical sinusoidal distortion.",
        "params": [
            {"key": "amplitude", "label": "Amplitude", "type": "number", "default": 20, "min": 1, "max": 150},
            {"key": "period_x", "label": "Period X", "type": "number", "default": 30, "min": 4, "max": 400},
        ],
    },
    {
        "name": "motion_blur",
        "title": "Motion blur",
        "category": "geometry",
        "description": "Blur along diagonal line.",
        "params": [{"key": "kernel_size", "label": "Kernel", "type": "number", "default": 15, "min": 3, "max": 64, "step": 2}],
    },
    {
        "name": "yiq_wrapper",
        "title": "YIQ wrapper",
        "category": "color",
        "description": "Apply another augmentation in YIQ space.",
        "params": [
            {"key": "apply_to", "label": "Channels", "type": "text", "default": "Y", "placeholder": "Y, I, Q or combination"},
            {"key": "inner_name", "label": "Inner augmentation", "type": "select", "options": ["gaussian_noise", "impulse_noise", "rayleigh_noise", "exponential_noise", "histogram_equalization", "gamma_correction"], "default": "gaussian_noise"},
            {"key": "inner_params", "label": "Inner params (JSON)", "type": "text", "default": {}, "placeholder": "{\"sigma\": 30}"},
        ],
    },
]


def _normalize_config(config_str: str) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(config_str) if config_str else []
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}")

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="Config must be a list of steps.")

    normalized = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Step #{idx + 1} must be an object.")
        name = item.get("name")
        if not name:
            raise HTTPException(status_code=400, detail=f"Step #{idx + 1} is missing 'name'.")
        params = item.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail=f"Step #{idx + 1} params must be an object.")
        normalized.append({"name": name, "params": _coerce_params(params)})
    return normalized


def _coerce_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Try to parse string values that look like JSON, leave others unchanged."""
    coerced: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            trimmed = value.strip()
            if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
                try:
                    coerced[key] = json.loads(trimmed)
                    continue
                except Exception:
                    coerced[key] = value
                    continue
        coerced[key] = value
    return coerced


def _load_image(upload: UploadFile) -> np.ndarray:
    upload.file.seek(0)
    content = upload.file.read()
    np_arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")
    return image


def _encode_image(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/augmentations")
async def list_augmentations() -> Dict[str, Any]:
    return {"items": AUGMENTATION_CATALOG}


@app.post("/process")
async def process_image(
    file: UploadFile = File(..., description="Image file to process"),
    config: str = Form("[]", description="JSON list of pipeline steps"),
) -> Dict[str, Any]:
    pipeline_config = _normalize_config(config)
    image = _load_image(file)

    pipeline = AugmentationPipeline()
    pipeline.build_from_json(pipeline_config)

    try:
        result = pipeline.run(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Pipeline failed: {exc}")

    encoded = _encode_image(result)
    channels = 1 if result.ndim == 2 else result.shape[2]

    return {
        "mime_type": "image/png",
        "shape": {"width": int(result.shape[1]), "height": int(result.shape[0]), "channels": int(channels)},
        "image_base64": encoded,
        "pipeline": pipeline_config,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
