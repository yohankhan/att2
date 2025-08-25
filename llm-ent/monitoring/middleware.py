from fastapi import Request
from . import pii_redactor

async def log_request(request: Request, call_next):
    body = await request.body()
    redacted = pii_redactor.redact(body.decode("utf-8", errors="ignore"))
    # TODO: send 'redacted' to your log sink / OTEL exporter
    response = await call_next(request)
    return response
