from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
import io
import logging
from typing import Optional
import sys

# Ensure project root in path
APP_ROOT = Path(__file__).parent.parent
sys.path.append(str(APP_ROOT))

# ✅ Fix import
from SIC.api.inference import SceneAnalysisModel

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent Scene Analysis API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Setup ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
model_path = PROJECT_ROOT / "SIC/models/weights/transfer_learning_resnet50.h5"
model: Optional[SceneAnalysisModel] = None

@app.on_event("startup")
def load_model():
    global model
    logger.info(f"Attempting to load model from: {model_path}")
    try:
        model = SceneAnalysisModel(model_path=model_path)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model = None

# --- Endpoints ---
@app.get("/health", summary="Check if the API is running")
def health_check():
    return {"status": "ok"}

@app.get("/model_info", summary="Get information about the model")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    return {
        "model_type": "Transfer Learning with ResNet50",
        "image_captioning_base": "Salesforce/blip-image-captioning-base",
        "classes": model.class_names
    }

@app.post("/predict", summary="Analyze a natural scene image")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Check server logs.")
    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")

    try:
        category, confidence, uncertainty = model.predict_with_uncertainty(pil_image)
        description = model.generate_description(pil_image)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return {
        "predicted_class": category,
        "confidence_score": confidence,
        "uncertainty_estimate": uncertainty,
        "description": description
    }
