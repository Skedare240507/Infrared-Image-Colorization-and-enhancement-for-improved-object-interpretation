"""Request logging middleware."""

from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, request

logger = logging.getLogger(__name__)


def register_request_logging(app: Flask) -> None:
    @app.before_request
    def start_request_timer() -> None:
        g.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        g.start_time = time.perf_counter()
        logger.info(
            "Request started",
            extra={
                "request_id": g.request_id,
                "method": request.method,
                "path": request.path,
            },
        )

    @app.after_request
    def log_request(response):
        duration_ms = (time.perf_counter() - g.get("start_time", time.perf_counter())) * 1000
        logger.info(
            "Request completed",
            extra={
                "request_id": g.get("request_id"),
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        response.headers["X-Request-ID"] = g.get("request_id", "")
        return response
