from fastapi import FastAPI, Request
import requests, os

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
API_KEY  = os.getenv("API_KEY", "dev-key")
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()
    r = requests.post(f"{VLLM_URL}/chat/completions",
                      headers={"Authorization": f"Bearer {API_KEY}"},
                      json=payload, timeout=60)
    return r.json()
