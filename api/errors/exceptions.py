"""Custom API exceptions."""

from __future__ import annotations


class APIError(Exception):
    status_code = 500
    error_code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(APIError):
    status_code = 400
    error_code = "validation_error"


class JobNotFoundError(APIError):
    status_code = 404
    error_code = "job_not_found"


class StorageError(APIError):
    status_code = 500
    error_code = "storage_error"


class ProcessingError(APIError):
    status_code = 422
    error_code = "processing_error"
