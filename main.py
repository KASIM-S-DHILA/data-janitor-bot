import os
import json
import logging
import asyncio
from functools import partial

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import httpx

from agent import run_agent
from logger import JSONLLogger

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
LOG_URL = os.environ.get("LOG_URL") or (
    RENDER_URL.rstrip("/") + "/run.jsonl" if RENDER_URL else "http://localhost:8080/run.jsonl"
)
PORT = int(os.environ.get("PORT", 8080))

jsonl_logger = JSONLLogger("run.jsonl")

app = FastAPI(title="Data-Analyst Telegram Bot")


@app.on_event("startup")
async def startup():
    from agent import _init_client
    try:
        _init_client()
    except ValueError as e:
        log.warning("No COHERE_API_KEY set: %s", e)

    webhook_url = RENDER_URL or os.environ.get("WEBHOOK_URL")
    if webhook_url:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
        full_url = webhook_url.rstrip("/") + "/webhook"
        async with httpx.AsyncClient(timeout=15) as c:
            resp = await c.post(url, json={"url": full_url})
            log.info("Webhook set to %s: %s", full_url, resp.json())
    else:
        log.warning("No RENDER_EXTERNAL_URL — skipping webhook setup")


@app.post("/webhook")
async def telegram_webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"ok": False}

    message = body.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    if not chat_id or not text:
        return {"ok": True}

    log.info("Received from %s: %.200s", chat_id, text)

    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None, partial(run_agent, text, LOG_URL, jsonl_logger)
        )
    except Exception as e:
        log.exception("Agent error")
        answer = json.dumps({"error": "agent_error", "detail": str(e)})

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as c:
        await c.post(url, json={"chat_id": chat_id, "text": answer})

    return {"ok": True}


@app.get("/run.jsonl")
async def serve_log():
    return PlainTextResponse(jsonl_logger.get_all_raw(), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
