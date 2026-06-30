"""Job-scoped file storage for uploads and pipeline outputs."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from api.errors.exceptions import JobNotFoundError, StorageError, ValidationError

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactName(str, Enum):
    ORIGINAL = "original"
    SUPER_RESOLUTION = "super_resolution"
    COLORIZED = "colorized"
    PREVIEW = "preview"


ARTIFACT_FILES = {
    ArtifactName.ORIGINAL: "original.tif",
    ArtifactName.SUPER_RESOLUTION: "super_resolution.tif",
    ArtifactName.COLORIZED: "colorized.png",
    ArtifactName.PREVIEW: "preview.png",
}


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    original_filename: str
    created_at: str
    updated_at: str
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class StorageService:
    MANIFEST_FILE = "manifest.json"

    def __init__(self, upload_dir: Path, output_dir: Path):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, uploaded_file: FileStorage, allowed_extensions: frozenset[str]) -> JobRecord:
        if uploaded_file is None or not uploaded_file.filename:
            raise ValidationError("No file provided in the upload request.")

        filename = secure_filename(uploaded_file.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed_extensions:
            raise ValidationError(
                "Invalid file type. Only thermal TIFF and PNG uploads are supported.",
                details={"allowed_extensions": sorted(allowed_extensions)},
            )

        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)

        destination = job_dir / f"original{suffix}"
        try:
            uploaded_file.save(destination)
        except OSError as exc:
            logger.exception("Failed to save upload for job %s", job_id)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise StorageError("Failed to store uploaded image.", details={"reason": str(exc)}) from exc

        now = self._now()
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.UPLOADED,
            original_filename=filename,
            created_at=now,
            updated_at=now,
            artifacts={ArtifactName.ORIGINAL.value: str(destination)},
        )
        self._write_manifest(job_id, record)
        logger.info("Created job %s for file %s", job_id, filename)
        return record

    def get_job(self, job_id: str) -> JobRecord:
        manifest_path = self._manifest_path(job_id)
        if not manifest_path.exists():
            raise JobNotFoundError(
                "Job not found.",
                details={"job_id": job_id},
            )
        return self._read_manifest(job_id)

    def update_job(self, record: JobRecord) -> JobRecord:
        record.updated_at = self._now()
        self._write_manifest(record.job_id, record)
        return record

    def artifact_path(self, job_id: str, artifact: ArtifactName) -> Path:
        record = self.get_job(job_id)
        key = artifact.value
        if key not in record.artifacts:
            raise ValidationError(
                "Requested artifact is not available for this job.",
                details={"job_id": job_id, "artifact": key, "status": record.status.value},
            )
        path = Path(record.artifacts[key])
        if not path.exists():
            raise StorageError(
                "Artifact file is missing on disk.",
                details={"job_id": job_id, "artifact": key},
            )
        return path

    def register_artifact(self, job_id: str, artifact: ArtifactName, path: Path) -> None:
        record = self.get_job(job_id)
        record.artifacts[artifact.value] = str(path)
        self.update_job(record)

    def job_directory(self, job_id: str) -> Path:
        job_dir = self._job_dir(job_id)
        if not job_dir.exists():
            raise JobNotFoundError("Job not found.", details={"job_id": job_id})
        return job_dir

    def delete_job(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        if not job_dir.exists():
            raise JobNotFoundError("Job not found.", details={"job_id": job_id})
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info("Deleted job %s", job_id)

    def _job_dir(self, job_id: str) -> Path:
        return self.output_dir / job_id

    def _manifest_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / self.MANIFEST_FILE

    def _write_manifest(self, job_id: str, record: JobRecord) -> None:
        manifest_path = self._manifest_path(job_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

    def _read_manifest(self, job_id: str) -> JobRecord:
        data = json.loads(self._manifest_path(job_id).read_text(encoding="utf-8"))
        return JobRecord(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            original_filename=data["original_filename"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            artifacts=data.get("artifacts", {}),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
