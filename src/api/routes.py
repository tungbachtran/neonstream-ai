"""
API Routes
"""
import uuid
import torch
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from loguru import logger
from typing import Optional

from src.api.schemas import (
    SinglePredictRequest, SinglePredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ChatModerationRequest, ChatModerationResponse,
    HealthResponse, PredictionResult, ScoreDetail
)

router = APIRouter()


def get_predictor(request: Request):
    """Dependency injection lấy predictor từ app state"""
    return request.app.state.predictor


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Health check endpoint"""
    predictor = request.app.state.predictor
    return HealthResponse(
        status="healthy",
        model=predictor.model_name,
        device=str(predictor.device),
        version="1.0.0"
    )


@router.post("/predict", response_model=SinglePredictResponse, tags=["Prediction"])
async def predict_single(
    body: SinglePredictRequest,
    predictor=Depends(get_predictor)
):
    """Predict 1 text"""
    try:
        result = predictor.predict_single(body.text)
        pred_result = PredictionResult(
            text=result['text'],
            prediction=result['prediction'],
            label=result['label'],
            confidence=result['confidence'],
            is_flagged=result['is_flagged'],
            scores=ScoreDetail(**result['scores']) if body.include_scores else None,
            inference_time_ms=result.get('inference_time_ms')
        )
        return SinglePredictResponse(
            data=pred_result,
            request_id=str(uuid.uuid4())
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(
    body: BatchPredictRequest,
    predictor=Depends(get_predictor)
):
    """Predict batch texts"""
    try:
        results = predictor.predict_batch(body.texts)
        pred_results = []
        flagged = 0
        for r in results:
            if r.get('is_flagged'):
                flagged += 1
            pred_results.append(PredictionResult(
                text=r['text'],
                prediction=r['prediction'],
                label=r['label'],
                confidence=r['confidence'],
                is_flagged=r['is_flagged'],
                scores=ScoreDetail(**r['scores']) if body.include_scores else None
            ))
        return BatchPredictResponse(
            total=len(pred_results),
            flagged_count=flagged,
            data=pred_results
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/moderate/chat", response_model=ChatModerationResponse, tags=["Moderation"])
async def moderate_chat(
    body: ChatModerationRequest,
    predictor=Depends(get_predictor)
):
    """
    Endpoint chính nhận tin nhắn từ NestJS.
    Trả về kết quả moderation với action: allow/warn/block
    """
    try:
        result = predictor.predict_single(body.text)

        # Xác định action dựa trên prediction
        action, reason = _determine_action(result)

        return ChatModerationResponse(
            message_id=body.message_id,
            user_id=body.user_id,
            room_id=body.room_id,
            prediction=result['prediction'],
            label=result['label'],
            confidence=result['confidence'],
            is_flagged=result['is_flagged'],
            scores=ScoreDetail(**result['scores']),
            action=action,
            reason=reason
        )
    except Exception as e:
        logger.error(f"Chat moderation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _determine_action(result: dict) -> tuple:
    prediction = result['prediction']
    scores     = result['scores']

    if prediction == 'clean':
        return "allow", None

    if prediction == 'toxic':
        if scores['toxic'] >= 0.8:
            return "block", "Nội dung vi phạm nghiêm trọng - ngôn ngữ thù địch"
        elif scores['toxic'] >= 0.5:
            return "warn",  "Nội dung có thể vi phạm - ngôn ngữ không phù hợp"
        return "allow", None

    if prediction == 'spam':
        if scores['spam'] >= 0.8:
            return "block", "Nội dung spam - quảng cáo không được phép"
        elif scores['spam'] >= 0.5:
            return "warn",  "Nội dung có thể là spam"
        return "allow", None

    if prediction == 'adult':                              # ← THÊM BLOCK MỚI
        if scores['adult'] >= 0.6:
            return "block", "Nội dung khiêu dâm - vi phạm nghiêm trọng"
        elif scores['adult'] >= 0.4:
            return "warn",  "Nội dung có thể không phù hợp với trẻ em"
        return "allow", None

    return "allow", None

