from io import BytesIO

from api.services.storage_service import ArtifactName


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["service"] == "ir-colorization-api"


def test_upload_process_and_download(client, sample_tiff):
    with sample_tiff.open("rb") as handle:
        upload_response = client.post(
            "/api/v1/thermal/upload",
            data={"file": (handle, "thermal.tif")},
            content_type="multipart/form-data",
        )

    assert upload_response.status_code == 201
    upload_payload = upload_response.get_json()
    job_id = upload_payload["job_id"]
    assert upload_payload["status"] == "uploaded"

    process_response = client.post(f"/api/v1/thermal/{job_id}/process")
    assert process_response.status_code == 200
    process_payload = process_response.get_json()
    assert process_payload["status"] == "completed"
    assert ArtifactName.COLORIZED.value in process_payload["artifacts"]
    assert process_payload["metadata"]["sr_backend"] == "opencv"

    for artifact in [
        ArtifactName.SUPER_RESOLUTION.value,
        ArtifactName.COLORIZED.value,
        ArtifactName.PREVIEW.value,
    ]:
        download_response = client.get(f"/api/v1/thermal/{job_id}/download/{artifact}")
        assert download_response.status_code == 200
        assert download_response.data


def test_upload_rejects_invalid_extension(client):
    response = client.post(
        "/api/v1/thermal/upload",
        data={"file": (BytesIO(b"not-a-tiff"), "image.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "validation_error"


def test_job_not_found(client):
    response = client.get("/api/v1/thermal/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] == "job_not_found"


def test_sample_endpoint(client):
    response = client.post("/api/v1/thermal/sample")
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["status"] == "completed"
    assert "job_id" in payload
    assert "original" in payload["artifacts"]
    assert "super_resolution" in payload["artifacts"]
    assert "colorized" in payload["artifacts"]


def test_previews(client, sample_tiff):
    # First upload
    with sample_tiff.open("rb") as handle:
        upload_response = client.post(
            "/api/v1/thermal/upload",
            data={"file": (handle, "thermal.tif")},
            content_type="multipart/form-data",
        )
    job_id = upload_response.get_json()["job_id"]

    # Process it
    client.post(f"/api/v1/thermal/{job_id}/process")

    # Fetch previews
    for artifact in ["original", "super_resolution", "colorized", "preview"]:
        preview_response = client.get(f"/api/v1/thermal/{job_id}/preview/{artifact}")
        assert preview_response.status_code == 200
        assert preview_response.content_type == "image/png"
        assert preview_response.data


def test_png_upload_and_processing(client, tmp_path):
    import cv2
    import numpy as np

    # Create a 64x64 grayscale PNG
    png_path = tmp_path / "thermal_test.png"
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    cv2.imwrite(str(png_path), img)

    with open(png_path, "rb") as handle:
        upload_response = client.post(
            "/api/v1/thermal/upload",
            data={"file": (handle, "thermal.png")},
            content_type="multipart/form-data",
        )

    assert upload_response.status_code == 201
    job_id = upload_response.get_json()["job_id"]
    assert upload_response.get_json()["status"] == "uploaded"

    process_response = client.post(f"/api/v1/thermal/{job_id}/process")
    assert process_response.status_code == 200
    assert process_response.get_json()["status"] == "completed"

    # Fetch previews
    for artifact in ["original", "super_resolution", "colorized"]:
        preview_response = client.get(f"/api/v1/thermal/{job_id}/preview/{artifact}")
        assert preview_response.status_code == 200
        assert preview_response.content_type == "image/png"
        assert preview_response.data


