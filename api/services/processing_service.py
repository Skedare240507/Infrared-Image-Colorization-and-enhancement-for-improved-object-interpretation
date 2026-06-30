"""Orchestrates upload, processing, and artifact retrieval."""

from __future__ import annotations

import logging

from werkzeug.datastructures import FileStorage

from api.config import AppConfig
from api.errors.exceptions import ProcessingError
from api.services.storage_service import ArtifactName, JobRecord, JobStatus, StorageService
from api.services.tiff_service import TiffService
from inference.colorization import ColorizationService
from inference.pipeline import ProcessingPipeline
from inference.super_resolution import SuperResolutionService

logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.storage = StorageService(config.upload_dir, config.output_dir)
        self.tiff_service = TiffService()
        self.pipeline = ProcessingPipeline(
            tiff_service=self.tiff_service,
            sr_service=SuperResolutionService(
                scale=config.sr_scale,
                checkpoint=config.sr_checkpoint,
                device=config.device,
            ),
            colorization_service=ColorizationService(
                checkpoint=config.colorization_checkpoint,
                device=config.device,
            ),
        )

    def upload_thermal(self, uploaded_file: FileStorage) -> JobRecord:
        record = self.storage.create_job(uploaded_file, self.config.allowed_extensions)
        original_path = self.storage.artifact_path(record.job_id, ArtifactName.ORIGINAL)
        self.tiff_service.validate_upload(original_path, self.config.allowed_extensions)
        return record

    def process_job(self, job_id: str) -> JobRecord:
        record = self.storage.get_job(job_id)

        if record.status == JobStatus.COMPLETED:
            logger.info("Job %s already completed; returning existing artifacts", job_id)
            return record

        if record.status == JobStatus.PROCESSING:
            raise ProcessingError(
                "Job is already being processed.",
                details={"job_id": job_id},
            )

        record.status = JobStatus.PROCESSING
        record.error = None
        self.storage.update_job(record)

        try:
            original_path = self.storage.artifact_path(job_id, ArtifactName.ORIGINAL)
            job_dir = self.storage.job_directory(job_id)
            result = self.pipeline.run(original_path, job_dir)

            self.storage.register_artifact(
                job_id, ArtifactName.SUPER_RESOLUTION, result.super_resolution_path
            )
            self.storage.register_artifact(job_id, ArtifactName.COLORIZED, result.colorized_path)
            self.storage.register_artifact(job_id, ArtifactName.PREVIEW, result.preview_path)

            record = self.storage.get_job(job_id)
            record.status = JobStatus.COMPLETED
            record.metadata.update(result.metadata)
            self.storage.update_job(record)
            logger.info("Job %s completed successfully", job_id)
            return record
        except Exception as exc:
            logger.exception("Processing failed for job %s", job_id)
            record = self.storage.get_job(job_id)
            record.status = JobStatus.FAILED
            record.error = str(exc)
            self.storage.update_job(record)
            raise ProcessingError(
                "Thermal processing failed.",
                details={"job_id": job_id, "reason": str(exc)},
            ) from exc

    def get_job(self, job_id: str) -> JobRecord:
        return self.storage.get_job(job_id)

    def get_artifact_path(self, job_id: str, artifact: ArtifactName):
        return self.storage.artifact_path(job_id, artifact)

    def delete_job(self, job_id: str) -> None:
        self.storage.delete_job(job_id)
