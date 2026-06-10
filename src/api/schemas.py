"""
Pydantic schemas cho API
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum


class PredictionLabel(str, Enum):
    CLEAN = "clean"
    TOXIC = "toxic"
    SPAM  = "spam"
    #ADULT = "adult"


class SinglePredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text cần phân tích")
    include_scores: bool = Field(default=True, description="Trả về scores chi tiết")

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("text không được rỗng")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    include_scores: bool = Field(default=True)

    @validator('texts', each_item=True)
    def text_not_empty(cls, v):
        return v.strip()


class ScoreDetail(BaseModel):
    clean: float
    toxic: float
    spam:  float
    #adult: float


class PredictionResult(BaseModel):
    text:        str
    prediction:  PredictionLabel
    label:       int
    confidence:  float
    is_flagged:  bool
    scores:      Optional[ScoreDetail] = None
    inference_time_ms: Optional[float] = None


class SinglePredictResponse(BaseModel):
    success:    bool = True
    data:       PredictionResult
    request_id: Optional[str] = None


class BatchPredictResponse(BaseModel):
    success:       bool = True
    total:         int
    flagged_count: int
    data:          List[PredictionResult]


class HealthResponse(BaseModel):
    status:      str
    model:       str
    device:      str
    version:     str = "1.0.0"


class ChatModerationRequest(BaseModel):
    """Schema nhận từ NestJS qua Redis/HTTP"""
    message_id:  str = Field(..., description="ID tin nhắn từ NestJS")
    user_id:     str = Field(..., description="ID người dùng")
    room_id:     str = Field(..., description="ID phòng chat")
    text:        str = Field(..., min_length=1, max_length=2000)
    timestamp:   Optional[str] = None


class ChatModerationResponse(BaseModel):
    message_id:  str
    user_id:     str
    room_id:     str
    prediction:  PredictionLabel
    label:       int
    confidence:  float
    is_flagged:  bool
    scores:      ScoreDetail
    action:      str  # "allow", "warn", "block"
    reason:      Optional[str] = None
