import numpy as np
import tensorflow as tf
from ..config import MODEL_PATH_STAGE_A, MODEL_PATH_STAGE_B
from .gradcam import auto_select_and_make_outputs
from typing import Dict, Any


# Classes for stage B (tampering types)
TAMPER_CLASSES = ["Copy_move", "Enhancement", "Removal_inpainting", "Splicing"]
# Overall output classes
CLASSES = ["Authentic", "Copy_move", "Enhancement", "Removal_inpainting", "Splicing"]

_model_stage_a = None
_model_stage_b = None


def get_model_stage_a():
    global _model_stage_a
    if _model_stage_a is None:
        _model_stage_a = tf.keras.models.load_model(MODEL_PATH_STAGE_A, compile=False)
    return _model_stage_a


def get_model_stage_b():
    global _model_stage_b
    if _model_stage_b is None:
        _model_stage_b = tf.keras.models.load_model(MODEL_PATH_STAGE_B, compile=False)
    return _model_stage_b


def predict_pair_with_visuals(ela_batch: np.ndarray, noise_batch: np.ndarray, pil_input_224) -> Dict[str, Any]:
    """Run hierarchical prediction and produce visuals (ELA, noise, gradcam overlay, heatmap, mask).

    Returns a dict with keys: predicted_class, confidence, probs, classes,
    ela_base64, noise_base64, gradcam_overlay_base64, heatmap_base64, mask_base64
    """
    model_a = get_model_stage_a()
    probs_a = model_a.predict([ela_batch, noise_batch], verbose=0)[0]
    is_tampered = int(np.argmax(probs_a))  # 0=authentic, 1=tampered

    if is_tampered == 0:
        idx = 0
        confidence = float(probs_a[0])
        probs = [float(probs_a[0])] + [0.0] * 4
        predicted_class = CLASSES[idx]

        # no gradcam for authentic images
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probs": probs,
            "classes": CLASSES,
            "ela_base64": _array_to_base64(ela_batch[0]),
            "noise_base64": _array_to_base64(noise_batch[0]),
            "gradcam_overlay_base64": None,
            "heatmap_base64": None,
            "mask_base64": None,
        }
        return result

    # tampered -> run stage B
    model_b = get_model_stage_b()
    probs_b = model_b.predict([ela_batch, noise_batch], verbose=0)[0]
    tamper_idx = int(np.argmax(probs_b))
    idx = tamper_idx + 1
    confidence = float(probs_b[tamper_idx])
    probs = [0.0] + probs_b.tolist()
    predicted_class = CLASSES[idx]

    # make gradcam outputs (auto-select)
    gc_outputs = auto_select_and_make_outputs(model_b, ela_batch, noise_batch, pil_input_224, class_index=tamper_idx)

    result = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probs": probs,
        "classes": CLASSES,
        "ela_base64": _array_to_base64(ela_batch[0]),
        "noise_base64": _array_to_base64(noise_batch[0]),
        "gradcam_overlay_base64": gc_outputs.get("overlay_b64"),
        "heatmap_base64": gc_outputs.get("heatmap_b64"),
        "mask_base64": gc_outputs.get("mask_b64"),
    }
    return result


def _array_to_base64(arr: np.ndarray) -> str:
    # arr expected in float [0,1] or uint8 [0,255], shape (H,W,3)
    from PIL import Image
    import io, base64

    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(a)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

