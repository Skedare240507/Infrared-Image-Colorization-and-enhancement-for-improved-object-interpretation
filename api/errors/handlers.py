"""Global Flask error handlers."""

from __future__ import annotations

import logging

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from api.errors.exceptions import APIError

logger = logging.getLogger(__name__)


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        logger.warning(
            "API error [%s]: %s",
            error.error_code,
            error.message,
            extra={"details": error.details},
        )
        payload = {
            "error": error.error_code,
            "message": error.message,
            "details": error.details,
        }
        return jsonify(payload), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException):
        logger.warning("HTTP error [%s]: %s", error.code, error.description)
        return (
            jsonify(
                {
                    "error": "http_error",
                    "message": error.description,
                    "details": {},
                }
            ),
            error.code,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        logger.exception("Unhandled exception: %s", error)
        return (
            jsonify(
                {
                    "error": "internal_error",
                    "message": "An unexpected error occurred.",
                    "details": {},
                }
            ),
            500,
        )
