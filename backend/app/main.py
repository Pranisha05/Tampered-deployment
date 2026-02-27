from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictResponse
from .services.preprocess import read_image_bytes, make_model_inputs
from .services.predict import predict_pair_with_visuals

app = FastAPI(title="Tampered Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    original = read_image_bytes(file_bytes)

    ela_batch, noise_batch, img_224 = make_model_inputs(original)

    result = predict_pair_with_visuals(ela_batch, noise_batch, img_224)

    return PredictResponse(
        predicted_class=result.get("predicted_class"),
        confidence=result.get("confidence", 0.0),
        probs=result.get("probs", []),
        classes=result.get("classes", []),
        gradcam_overlay_base64=result.get("gradcam_overlay_base64"),
        ela_base64=result.get("ela_base64"),
        noise_base64=result.get("noise_base64"),
        heatmap_base64=result.get("heatmap_base64"),
        mask_base64=result.get("mask_base64"),
    )
