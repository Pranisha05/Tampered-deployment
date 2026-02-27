from pydantic import BaseModel
from typing import List, Optional


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probs: List[float]
    classes: List[str]
    gradcam_overlay_base64: Optional[str] = None   # PNG base64 (data URI without header)
    ela_base64: Optional[str] = None
    noise_base64: Optional[str] = None
    heatmap_base64: Optional[str] = None
    mask_base64: Optional[str] = None
