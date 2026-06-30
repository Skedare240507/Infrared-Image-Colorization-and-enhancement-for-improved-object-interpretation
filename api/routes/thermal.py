"""Thermal upload, processing, and download routes."""

from __future__ import annotations

import io
import logging
import shutil
import uuid

import cv2
import numpy as np
import rasterio
from flask import Blueprint, current_app, jsonify, request, send_file
from rasterio.transform import from_origin

from api.schemas.job import job_response
from api.services.storage_service import ARTIFACT_FILES, ArtifactName, JobRecord, JobStatus

logger = logging.getLogger(__name__)

bp = Blueprint("thermal", __name__)

# BUG-FIX: MIME type for ORIGINAL must be dynamic — when a PNG is uploaded
# the stored file is 'original.png', NOT 'original.tif'. Returning image/tiff
# for a PNG file causes browsers to reject / mis-render the download.
# We derive the mime type from the actual artifact path at download time instead.
DOWNLOAD_MIMETYPES_BY_EXT = {
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".png": "image/png",
}


def _processing_service():
    return current_app.extensions["processing_service"]


@bp.post("/thermal/upload")
def upload_thermal():
    """Upload a thermal TIFF and create a processing job."""
    uploaded_file = request.files.get("file")
    record = _processing_service().upload_thermal(uploaded_file)
    logger.info("Thermal upload accepted for job %s", record.job_id)
    return jsonify(job_response(record)), 201


@bp.post("/thermal/<job_id>/process")
def process_thermal(job_id: str):
    """Run super-resolution and colorization for an uploaded job."""
    record = _processing_service().process_job(job_id)
    return jsonify(job_response(record)), 200


@bp.get("/thermal/<job_id>")
def get_job(job_id: str):
    """Return job status, metadata, and available artifacts."""
    record = _processing_service().get_job(job_id)
    return jsonify(job_response(record)), 200


@bp.get("/thermal/<job_id>/download/<artifact>")
def download_artifact(job_id: str, artifact: str):
    """Download a saved pipeline artifact."""
    try:
        artifact_name = ArtifactName(artifact)
    except ValueError as exc:
        return (
            jsonify(
                {
                    "error": "validation_error",
                    "message": "Unknown artifact type.",
                    "details": {
                        "artifact": artifact,
                        "allowed": [item.value for item in ArtifactName],
                    },
                }
            ),
            400,
        )

    path = _processing_service().get_artifact_path(job_id, artifact_name)
    # BUG-FIX: derive MIME type dynamically from actual file extension
    mimetype = DOWNLOAD_MIMETYPES_BY_EXT.get(path.suffix.lower(), "application/octet-stream")
    logger.info("Serving artifact %s for job %s", artifact_name.value, job_id)
    return send_file(
        path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=path.name,
    )


@bp.get("/thermal/<job_id>/preview/<artifact>")
def preview_artifact(job_id: str, artifact: str):
    """Serve a browser-compatible PNG preview of any artifact (converts TIFFs on-the-fly)."""
    try:
        artifact_name = ArtifactName(artifact)
    except ValueError:
        return (
            jsonify(
                {
                    "error": "validation_error",
                    "message": "Unknown artifact type.",
                    "details": {
                        "artifact": artifact,
                        "allowed": [item.value for item in ArtifactName],
                    },
                }
            ),
            400,
        )

    path = _processing_service().get_artifact_path(job_id, artifact_name)

    # If already a PNG, send it directly
    if path.suffix.lower() == ".png":
        return send_file(path, mimetype="image/png")

    # If it's a TIFF, read and convert to PNG on the fly
    try:
        import io
        import cv2
        import numpy as np

        # Read using TiffService
        thermal_image = _processing_service().tiff_service.read_thermal(path)
        array = thermal_image.array

        # Normalize to 0-255 range
        array_min = float(array.min())
        array_max = float(array.max())
        if array_max - array_min > 1e-6:
            normalized = (array - array_min) / (array_max - array_min)
        else:
            normalized = np.zeros_like(array)

        gray = (normalized * 255.0).astype(np.uint8)

        # Encode as PNG
        success, encoded_img = cv2.imencode(".png", gray)
        if not success:
            raise ValueError("Failed to encode image to PNG format.")

        img_bytes = io.BytesIO(encoded_img.tobytes())
        img_bytes.seek(0)  # BUG-FIX: BytesIO must be rewound to pos 0 before send_file reads it
        return send_file(img_bytes, mimetype="image/png")
    except Exception as exc:
        logger.exception("Failed to generate preview for job %s, artifact %s", job_id, artifact)
        return (
            jsonify(
                {
                    "error": "processing_error",
                    "message": f"Failed to generate preview for {artifact}: {exc}",
                }
            ),
            500,
        )


@bp.post("/thermal/sample")
def create_sample():
    """Create a sample job with simulated thermal data and process it."""
    # BUG-FIX: All imports moved to module-level above; using them directly here
    service = _processing_service()
    job_id = uuid.uuid4().hex
    job_dir = service.storage._job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Generate an interesting simulated thermal pattern (128x128 grid)
    # A base temperature gradient, some hot spots (factories/pipelines) and a cold stream.
    size = 128
    y, x = np.mgrid[0:size, 0:size]
    
    # Base temperature with gradient
    data = 25.0 + 5.0 * np.sin(x / 20.0) * np.cos(y / 20.0)
    
    # A cold winding linear feature (river or pipeline)
    river = 5.0 * np.sin(x / 15.0 + y / 30.0)
    data += river
    
    # Some hot circular spots (factories, buildings)
    centers = [(35, 45, 12), (90, 80, 18), (60, 95, 8)]
    for cy, cx, radius in centers:
        dist = np.sqrt((y - cy)**2 + (x - cx)**2)
        data += np.where(dist < radius, 18.0 * (1.0 - dist / radius), 0.0)

    # Add minor high-frequency noise for realism
    np.random.seed(42)
    data += np.random.normal(0, 0.4, size=(size, size))

    destination = job_dir / ARTIFACT_FILES[ArtifactName.ORIGINAL]
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(77.58, 12.97, 0.0001, 0.0001), # Georeferenced (Bangalore region)
    }

    try:
        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)
    except OSError as exc:
        logger.exception("Failed to save sample TIFF for job %s", job_id)
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        return (
            jsonify(
                {
                    "error": "storage_error",
                    "message": f"Failed to store sample TIFF: {exc}",
                }
            ),
            500,
        )

    now = service.storage._now()
    record = JobRecord(
        job_id=job_id,
        status=JobStatus.UPLOADED,
        original_filename="sample_geospatial_thermal.tif",
        created_at=now,
        updated_at=now,
        artifacts={ArtifactName.ORIGINAL.value: str(destination)},
    )
    service.storage._write_manifest(job_id, record)
    logger.info("Created sample job %s", job_id)

    # Process the job immediately
    try:
        processed_record = service.process_job(job_id)
        return jsonify(job_response(processed_record)), 201
    except Exception as exc:
        logger.exception("Failed to process sample job %s", job_id)
        return (
            jsonify(
                {
                    "error": "processing_error",
                    "message": f"Failed to process sample job: {exc}",
                }
            ),
            500,
        )


@bp.delete("/thermal/<job_id>")
def delete_job(job_id: str):
    """Delete a job and all saved outputs."""
    _processing_service().delete_job(job_id)
    return jsonify({"message": "Job deleted.", "job_id": job_id}), 200

