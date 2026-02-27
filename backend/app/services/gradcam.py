import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

from ..config import IMG_SIZE

# Style
CANNY1 = 60
CANNY2 = 140
EDGE_THICKNESS = 1

# Visualization toggles
USE_BBOX = True
DRAW_CONTOUR = True
OVERLAY_ALPHA = 0.55



# Config for auto-layer search
MIN_HW = 14
TOP_K_LAYERS = 12
KEEP_PERCENT = 10
MIN_COMPONENT_AREA = 80



def _to_base64_pil(img_arr):
    pil = Image.fromarray(img_arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def list_candidate_conv_layers(model: tf.keras.Model, min_hw=14):
    """Return conv/depthwise conv layer names whose feature map resolution >= min_hw."""
    candidates = []

    def walk(m):
        for layer in m.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                try:
                    sh = layer.output_shape
                    if isinstance(sh, (tuple, list)) and len(sh) >= 4:
                        h, w = sh[1], sh[2]
                        if h is not None and w is not None and h >= min_hw and w >= min_hw:
                            candidates.append(layer.name)
                except Exception:
                    pass
            if isinstance(layer, tf.keras.Model):
                walk(layer)

    walk(model)

    # fallback: collect all conv layers if none matched min_hw
    if not candidates:
        def walk_all(m):
            for layer in m.layers:
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    candidates.append(layer.name)
                if isinstance(layer, tf.keras.Model):
                    walk_all(layer)
        walk_all(model)

    # dedupe, keep order
    seen = set()
    uniq = []
    for n in candidates:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


def predict_probs(model, ela, noise):
    """Run prediction and return probabilities for the batch."""
    return model({"ela": ela, "noise": noise}, training=False).numpy()[0]


# ======================================================
# FINAL OVERLAY (heat only where suspected)
# ======================================================
def build_overlay(
    original_rgb,
    cam01,
    mask255=None,
    alpha=OVERLAY_ALPHA,
    contour=None,
    draw_edges=True,
    draw_contour_flag=True,
    draw_bbox_flag=True,
):
    H, W = IMG_SIZE
    orig = cv2.resize(original_rgb, (W, H))

    heat_u8 = np.uint8(255 * cam01)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    if mask255 is not None:
        mask3 = cv2.cvtColor(mask255, cv2.COLOR_GRAY2BGR)
        heat_color = cv2.bitwise_and(heat_color, mask3)

    orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(orig_bgr, 1 - alpha, heat_color, alpha, 0)

    if draw_edges:
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, CANNY1, CANNY2)
        if EDGE_THICKNESS > 1:
            k = np.ones((EDGE_THICKNESS, EDGE_THICKNESS), np.uint8)
            edges = cv2.dilate(edges, k, iterations=1)
        blended[edges > 0] = (0, 0, 0)

    if contour is not None:
        if draw_contour_flag:
            cv2.drawContours(blended, [contour], -1, (0, 255, 0), 2)
        if draw_bbox_flag:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def make_gradcam_heatmap_for_layer(model, ela_input, noise_input, class_index, conv_layer_name):
    """Compute Grad-CAM heatmap for a specific layer."""
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model({"ela": ela_input, "noise": noise_input}, training=False)
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_out)
    if grads is None:
        return None

    # Compute mean gradient weights over spatial dims
    grad_rank = len(grads.shape)
    pool_axes = tuple(range(grad_rank - 1))
    weights = tf.reduce_mean(grads, axis=pool_axes)

    conv_out = conv_out[0]
    cam = tf.reduce_sum(tf.multiply(weights, conv_out), axis=-1)

    cam = tf.maximum(cam, 0)
    cam_max = tf.reduce_max(cam)
    if cam_max > 0:
        cam = cam / cam_max
    return cam.numpy()


def cam_to_binary_mask(cam01, keep_percent=10, out_hw=IMG_SIZE):
    """Convert CAM to binary mask using top percentile threshold."""
    cam_r = cv2.resize(cam01, out_hw)
    cam_r = np.clip(cam_r, 0, 1)

    # keep top keep_percent% of pixels
    thr = np.percentile(cam_r, 100 - keep_percent)
    mask = (cam_r >= thr).astype(np.uint8) * 255

    # morphological cleanup
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return cam_r, mask


def keep_largest_component(mask255, min_area=80):
    """Extract only the largest connected component from the mask."""
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask255), None

    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return np.zeros_like(mask255), None

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask255)
    cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
    return clean, largest


def apply_mask_on_inputs(ela, noise, mask255, mode="zero"):
    """Apply mask to both ELA and noise inputs for faithfulness scoring."""
    mask_norm = (mask255.astype(np.float32) / 255.0)[None, ..., None]
    if mode == "zero":
        ela_m = ela * (1.0 - mask_norm)
        noise_m = noise * (1.0 - mask_norm)
        return ela_m.astype(np.float32), noise_m.astype(np.float32)

    # Blur mode (alternative)
    ela_np = ela[0].copy()
    noise_np = noise[0].copy()
    ela_blur = cv2.GaussianBlur(ela_np, (11, 11), 0)
    noise_blur = cv2.GaussianBlur(noise_np, (11, 11), 0)
    m2 = mask_norm[0]
    ela_np = ela_np * (1 - m2) + ela_blur * m2
    noise_np = noise_np * (1 - m2) + noise_blur * m2
    return ela_np[None, ...].astype(np.float32), noise_np[None, ...].astype(np.float32)


def auto_select_best_layer(model, ela, noise, target_class, min_hw=MIN_HW, keep_percent=KEEP_PERCENT, top_k=TOP_K_LAYERS):
    """Auto-select best conv layer using faithfulness scoring (drop in confidence when region masked)."""
    base_probs = predict_probs(model, ela, noise)
    base = float(base_probs[target_class])

    layers = list_candidate_conv_layers(model, min_hw=min_hw)
    layers_to_test = layers[-top_k:] if len(layers) > top_k else layers

    best = None
    results = []

    for lname in layers_to_test:
        cam = make_gradcam_heatmap_for_layer(model, ela, noise, target_class, lname)
        if cam is None:
            continue

        cam01, mask = cam_to_binary_mask(cam, keep_percent=keep_percent, out_hw=IMG_SIZE)

        # Mask both inputs and measure drop in confidence
        ela_m, noise_m = apply_mask_on_inputs(ela, noise, mask, mode="zero")
        masked_probs = predict_probs(model, ela_m, noise_m)
        masked_score = float(masked_probs[target_class])

        drop = base - masked_score
        area = float(mask.mean() / 255.0)
        score = drop - 0.10 * area  # prioritize drop, lightly prefer tight masks

        results.append((lname, score, drop, area, cam01, mask))

        if best is None or score > best[1]:
            best = (lname, score, drop, area, cam01, mask)

    if best is None:
        # Fallback: use last conv layer
        layer = layers[-1] if layers else None
        if layer:
            cam = make_gradcam_heatmap_for_layer(model, ela, noise, target_class, layer)
            cam01, mask = cam_to_binary_mask(cam, keep_percent=keep_percent, out_hw=IMG_SIZE)
            best = (layer, 0.0, 0.0, 0.0, cam01, mask)
        else:
            return base, base_probs, None, []

    results.sort(key=lambda x: x[1], reverse=True)
    return base, base_probs, best, results


def auto_select_and_make_outputs(model, ela_batch, noise_batch, pil_input_224: Image.Image, class_index: int, keep_percent=KEEP_PERCENT, top_k=TOP_K_LAYERS, min_hw=MIN_HW):
    """Run auto-layer selection and produce gradcam outputs with mask morphology and component extraction."""
    base_conf, base_probs, best, results = auto_select_best_layer(
        model=model,
        ela=ela_batch,
        noise=noise_batch,
        target_class=class_index,
        min_hw=min_hw,
        keep_percent=keep_percent,
        top_k=top_k
    )

    best_layer, best_score, best_drop, best_area, cam01, raw_mask = best

    # Extract largest suspected component
    mask, contour = keep_largest_component(raw_mask, min_area=MIN_COMPONENT_AREA)

    # For suspected Area overlay, use the raw cam01 and mask from the best layer. This focuses on the most suspicious region.
    original = np.array(pil_input_224)

    overlay_img = build_overlay(
        original_rgb=original,
        cam01=cam01,
        mask255=mask,
        contour=contour,
        draw_edges=True,
        draw_contour_flag=DRAW_CONTOUR,
        draw_bbox_flag=USE_BBOX,
    )

    overlay_b64 = _to_base64_pil(overlay_img)
    mask_b64 = _to_base64_pil(mask)

    heat_only = build_overlay(
        original_rgb=original,
        cam01=cam01,
        mask255=mask,
        contour=None,
        draw_edges=False,
        draw_contour_flag=False,
        draw_bbox_flag=False,
    )
    heatmap_b64 = _to_base64_pil(heat_only)

    return {
        "overlay_b64": overlay_b64,
        "heatmap_b64": heatmap_b64,
        "mask_b64": mask_b64,
        "layer": best_layer,
        "best_score": best_score,
        "best_drop": best_drop,
        "best_area": best_area,
    }
