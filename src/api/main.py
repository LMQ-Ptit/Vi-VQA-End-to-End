"""
FastAPI Backend
API endpoint cho VQA prediction
"""
import io
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
_predict_fn = None

def init_model(model_path):
    """Initialize model khi app start"""
    global _model, _processor, _predict_fn
    from src.inference.predict import load_model_and_processor, predict
    _model, _processor = load_model_and_processor(model_path)
    _predict_fn = predict

@app.post("/predict")
async def predict_api(image: UploadFile = File(...), question: str = Form(...)):
    """Predict VQA answer từ image và question"""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start_time = time.time()

    image_data = await image.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Image data is empty")

    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        pil_image.save(tmp.path)
        tmp_path = tmp.name

    try:
        answer = _predict_fn(tmp_path, question, _model, _processor)
        latency = (time.time() - start_time) * 1000
        return {"answer": answer, "latency_ms": latency}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": str(next(_model.parameters()).device) if _model else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)