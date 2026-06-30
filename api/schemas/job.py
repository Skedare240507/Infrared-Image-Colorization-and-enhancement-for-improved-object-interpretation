"""Request and response schema helpers."""

from __future__ import annotations

from api.services.storage_service import JobRecord


def job_response(record: JobRecord) -> dict:
    return {
        "job_id": record.job_id,
        "status": record.status.value,
        "original_filename": record.original_filename,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "artifacts": list(record.artifacts.keys()),
        "metadata": record.metadata,
        "error": record.error,
    }
