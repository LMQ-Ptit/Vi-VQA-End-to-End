"""
FastAPI Schemas
Pydantic models cho API request/response
"""
from pydantic import BaseModel

class PredictRequest(BaseModel):
    image: str  # Base64 encoded image hoặc URL
    question: str

class PredictResponse(BaseModel):
    answer: str
    model: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool