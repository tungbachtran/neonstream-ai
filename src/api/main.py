"""
FastAPI Application Entry Point
"""
import yaml
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger

from src.api.routes import router
from src.api.middleware import LoggingMiddleware, RateLimitMiddleware
from src.inference.predictor import ToxicSpamPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup và shutdown events"""
    # ── STARTUP ──────────────────────────────────────────
    logger.info("🚀 Starting AI Moderation Server...")

    with open("configs/config.yaml", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    checkpoint_path = Path(config['paths']['checkpoint_dir']) / "checkpoint_best.pt"

    if not checkpoint_path.exists():
        logger.warning(f"⚠️  Checkpoint not found at {checkpoint_path}")
        logger.warning("Server starting without model - train first!")
        app.state.predictor = None
    else:
        app.state.predictor = ToxicSpamPredictor(
            checkpoint_path=str(checkpoint_path),
            model_name=config['model']['name'],
            confidence_threshold=config['inference']['confidence_threshold'],
            toxic_threshold=config['inference']['toxic_threshold'],
            spam_threshold=config['inference']['spam_threshold']
        )
        logger.info("✅ Model loaded successfully!")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────
    logger.info("Shutting down server...")
    if hasattr(app.state, 'predictor') and app.state.predictor:
        del app.state.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title="Vietnamese Toxic & Spam Detection API",
    description="PhoBERT-based AI server nhận diện từ tục và spam tiếng Việt",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── Middleware ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)

# ── Routes ────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,        # 1 worker vì model dùng GPU memory
        log_level="info"
    )
