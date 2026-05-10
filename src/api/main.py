"""
FastAPI Backend
API endpoint cho VQA prediction
"""
import io
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Vi-VQA API Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
_model = None
_processor = None

def init_model(model_path):
    """Initialize model khi app start"""
    global _model, _processor
    from src.inference.predict import load_model_and_processor
    _model, _processor = load_model_and_processor(model_path)

@app.post("/predict")
async def predict_api(image: UploadFile = File(...), question: str = Form(...)):
    """Predict VQA answer từ image và question"""
    start_time = time.time()

    image_data = await image.read()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    try:
        from src.inference.predict import predict
        answer = predict(pil_image, question, _model, _processor)
        latency = (time.time() - start_time) * 1000
        return {"answer": answer, "latency_ms": latency}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": _model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)