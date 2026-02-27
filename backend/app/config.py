import os

# Two-stage model paths (absolute paths based on this file's location)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_STAGE_A = os.path.join(BASE_DIR, "models", "best_stageA_binary.keras")
MODEL_PATH_STAGE_B = os.path.join(BASE_DIR, "models", "final_stageB_tamper4.keras")

IMG_SIZE = (224, 224)          # must match training
JPEG_QUALITY = 90              # for ELA


