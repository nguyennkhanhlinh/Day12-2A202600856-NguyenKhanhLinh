"""
Production AI Agent — Kết hợp tất cả Day 12 concepts

Checklist:
  ✅ Config từ environment (12-factor)
  ✅ Structured JSON logging
  ✅ API Key authentication
  ✅ Rate limiting (Redis — stateless, 10 req/min/user)
  ✅ Cost guard (Redis — $10/month/user)
  ✅ Conversation history (Redis, theo user_id)
  ✅ Input validation (Pydantic)
  ✅ Health check + Readiness probe (check Redis)
  ✅ Graceful shutdown
  ✅ Security headers
  ✅ CORS
  ✅ Error handling
  ✅ Stateless design — toàn bộ state nằm trong Redis
"""
import time
import signal
import logging
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Security, Depends, Request, Response
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.config import settings

# Mock LLM (thay bằng OpenAI/Anthropic khi có API key)
from utils.mock_llm import ask as llm_ask

# ─────────────────────────────────────────────────────────
# Logging — JSON structured
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

START_TIME = time.time()
_request_count = 0
_error_count = 0

# Redis client — khởi tạo trong lifespan. Toàn bộ state (history, rate-limit,
# cost) lưu ở đây ⇒ agent stateless, scale nhiều instance vẫn nhất quán.
redis_client: aioredis.Redis | None = None

# Giới hạn lịch sử hội thoại giữ cho mỗi user (số lượt q/a gần nhất)
HISTORY_MAX_TURNS = 10
# Chi phí ước tính mỗi 1k token (giống bảng giá demo)
COST_PER_1K_INPUT = 0.00015
COST_PER_1K_OUTPUT = 0.0006


# ─────────────────────────────────────────────────────────
# Rate Limiter — Redis sliding window (sorted set)
# ─────────────────────────────────────────────────────────
async def check_rate_limit(user_id: str):
    """Sliding window 60s bằng Redis sorted set. Vượt limit → 429."""
    key = f"ratelimit:{user_id}"
    now = time.time()
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(key, 0, now - 60)   # bỏ request cũ hơn 60s
        pipe.zadd(key, {f"{now}": now})           # ghi nhận request hiện tại
        pipe.zcard(key)                           # đếm số request trong cửa sổ
        pipe.expire(key, 60)
        _, _, count, _ = await pipe.execute()
    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.rate_limit_per_minute} req/min",
            headers={"Retry-After": "60"},
        )


# ─────────────────────────────────────────────────────────
# Cost Guard — Redis budget theo tháng/user
# ─────────────────────────────────────────────────────────
def _cost_key(user_id: str) -> str:
    month = time.strftime("%Y-%m")
    return f"cost:{user_id}:{month}"


async def check_budget(user_id: str):
    """Chặn TRƯỚC khi gọi LLM nếu user đã vượt ngân sách tháng → 402."""
    spent = float(await redis_client.get(_cost_key(user_id)) or 0.0)
    if spent >= settings.monthly_budget_usd:
        raise HTTPException(
            status_code=402,
            detail=f"Monthly budget exhausted (${settings.monthly_budget_usd}/month). Try next month.",
        )


async def record_cost(user_id: str, input_tokens: int, output_tokens: int) -> float:
    """Cộng dồn chi phí thực tế vào Redis, set TTL ~33 ngày."""
    cost = (input_tokens / 1000) * COST_PER_1K_INPUT + (output_tokens / 1000) * COST_PER_1K_OUTPUT
    key = _cost_key(user_id)
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.incrbyfloat(key, cost)
        pipe.expire(key, 60 * 60 * 24 * 33)
        new_total, _ = await pipe.execute()
    return float(new_total)


# ─────────────────────────────────────────────────────────
# Conversation history — Redis list theo user_id
# ─────────────────────────────────────────────────────────
async def get_history(user_id: str) -> list[dict]:
    raw = await redis_client.lrange(f"history:{user_id}", -HISTORY_MAX_TURNS, -1)
    return [json.loads(item) for item in raw]


async def save_turn(user_id: str, question: str, answer: str):
    key = f"history:{user_id}"
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.rpush(key, json.dumps({"q": question, "a": answer}))
        pipe.ltrim(key, -HISTORY_MAX_TURNS * 2, -1)   # giữ tối đa N lượt
        pipe.expire(key, 60 * 60 * 24 * 7)            # lịch sử sống 7 ngày
        await pipe.execute()


# ─────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key != settings.agent_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include header: X-API-Key: <key>",
        )
    return api_key


# ─────────────────────────────────────────────────────────
# Lifespan — mở/đóng kết nối Redis
# ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    logger.info(json.dumps({
        "event": "startup",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "redis_url": settings.redis_url,
    }))
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await redis_client.ping()
        logger.info(json.dumps({"event": "redis_connected"}))
    except Exception as e:
        logger.error(json.dumps({"event": "redis_error", "detail": str(e)}))
    logger.info(json.dumps({"event": "ready"}))

    yield

    if redis_client is not None:
        await redis_client.aclose()
    logger.info(json.dumps({"event": "shutdown"}))


# ─────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    global _request_count, _error_count
    start = time.time()
    _request_count += 1
    try:
        response: Response = await call_next(request)
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if "server" in response.headers:
            del response.headers["server"]
        duration = round((time.time() - start) * 1000, 1)
        logger.info(json.dumps({
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": duration,
        }))
        return response
    except Exception:
        _error_count += 1
        raise


# ─────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Your question for the agent")
    user_id: str = Field("anonymous", min_length=1, max_length=128,
                         description="User identifier — dùng cho history, rate-limit, budget")


class AskResponse(BaseModel):
    question: str
    answer: str
    user_id: str
    history_len: int
    model: str
    timestamp: str


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "endpoints": {
            "ask": "POST /ask (requires X-API-Key)",
            "health": "GET /health",
            "ready": "GET /ready",
        },
    }


@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask_agent(
    body: AskRequest,
    request: Request,
    _key: str = Depends(verify_api_key),
):
    """
    Send a question to the AI agent.

    **Authentication:** Include header `X-API-Key: <your-key>`
    Conversation history được nhớ theo `user_id` (lưu trong Redis).
    """
    user_id = body.user_id

    # Bảo vệ: rate limit + budget (theo user, state ở Redis)
    await check_rate_limit(user_id)
    await check_budget(user_id)

    # Lấy lịch sử hội thoại trước đó → dựng ngữ cảnh cho LLM
    history = await get_history(user_id)
    context = "\n".join(f"User: {t['q']}\nAgent: {t['a']}" for t in history)
    prompt = f"{context}\nUser: {body.question}" if context else body.question

    logger.info(json.dumps({
        "event": "agent_call",
        "user": user_id,
        "q_len": len(body.question),
        "history_turns": len(history),
        "client": str(request.client.host) if request.client else "unknown",
    }))

    answer = llm_ask(prompt)

    # Ghi nhận chi phí + lưu lịch sử (đều vào Redis)
    input_tokens = len(prompt.split()) * 2
    output_tokens = len(answer.split()) * 2
    await record_cost(user_id, input_tokens, output_tokens)
    await save_turn(user_id, body.question, answer)

    return AskResponse(
        question=body.question,
        answer=answer,
        user_id=user_id,
        history_len=len(history) + 1,
        model=settings.llm_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/health", tags=["Operations"])
def health():
    """Liveness probe. Platform restarts container if this fails."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "total_requests": _request_count,
        "checks": {"llm": "mock" if not settings.openai_api_key else "openai"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["Operations"])
async def ready():
    """Readiness probe. Chỉ ready khi Redis kết nối được (state store sẵn sàng)."""
    try:
        if redis_client is None:
            raise RuntimeError("redis not initialized")
        await redis_client.ping()
    except Exception as e:
        raise HTTPException(503, f"Not ready: Redis unavailable ({e})")
    return {"ready": True, "redis": "ok"}


@app.get("/metrics", tags=["Operations"])
async def metrics(_key: str = Depends(verify_api_key)):
    """Basic metrics (protected)."""
    return {
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "total_requests": _request_count,
        "error_count": _error_count,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "monthly_budget_usd": settings.monthly_budget_usd,
    }


# ─────────────────────────────────────────────────────────
# Graceful Shutdown
# ─────────────────────────────────────────────────────────
def _handle_signal(signum, _frame):
    logger.info(json.dumps({"event": "signal", "signum": signum}))

signal.signal(signal.SIGTERM, _handle_signal)


if __name__ == "__main__":
    logger.info(f"Starting {settings.app_name} on {settings.host}:{settings.port}")
    logger.info(f"API Key: {settings.agent_api_key[:4]}****")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        timeout_graceful_shutdown=30,
    )
