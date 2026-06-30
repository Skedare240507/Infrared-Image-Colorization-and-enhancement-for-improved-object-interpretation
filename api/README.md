# Backend API

Production Flask service for thermal TIFF upload, super-resolution, colorization, and artifact download.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
flask --app api.app run --debug
```

Production (Gunicorn):

```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 wsgi:app
```

## Folder structure

```
api/
├── app.py                 # Application factory
├── config.py              # Env + YAML configuration
├── errors/                # Custom exceptions + handlers
├── middleware/            # Request logging
├── routes/
│   ├── health.py          # GET /health
│   └── thermal.py         # Upload / process / download
├── schemas/               # Response serializers
└── services/
    ├── processing_service.py
    ├── storage_service.py
    └── tiff_service.py

inference/
├── super_resolution.py    # PyTorch SR + OpenCV fallback
├── colorization.py        # PyTorch colorization + colormap fallback
└── pipeline.py            # SR → colorize → save artifacts

outputs/                   # Job artifacts (gitignored)
uploads/                   # Reserved upload staging (gitignored)
logs/                      # Rotating application logs
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/api/v1/thermal/upload` | Upload thermal TIFF (`file` field) |
| POST | `/api/v1/thermal/{job_id}/process` | Run super-resolution + colorization |
| GET | `/api/v1/thermal/{job_id}` | Job status and artifact list |
| GET | `/api/v1/thermal/{job_id}/download/{artifact}` | Download result |
| DELETE | `/api/v1/thermal/{job_id}` | Delete job and outputs |

### Artifacts

- `original` — uploaded thermal TIFF
- `super_resolution` — upscaled thermal GeoTIFF
- `colorized` — RGB PNG output
- `preview` — side-by-side comparison PNG

## Example workflow

```bash
# 1. Upload
curl -F "file=@thermal.tif" http://localhost:5000/api/v1/thermal/upload

# 2. Process
curl -X POST http://localhost:5000/api/v1/thermal/{job_id}/process

# 3. Download colorized result
curl -OJ http://localhost:5000/api/v1/thermal/{job_id}/download/colorized
```

## Configuration

Environment variables override `configs/default.yaml`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FLASK_SECRET_KEY` | `change-me-in-production` | Flask session secret |
| `FLASK_DEBUG` | `false` | Debug mode |
| `MAX_UPLOAD_MB` | `50` | Upload size limit |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `SR_SCALE` | `2` | Super-resolution scale factor |
| `SR_CHECKPOINT` | — | Path to SR `.pt` weights |
| `COLORIZATION_CHECKPOINT` | `models/weights/colorization.pt` | Colorization weights |
| `INFERENCE_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |

## Error responses

All errors return JSON:

```json
{
  "error": "validation_error",
  "message": "Human-readable message",
  "details": {}
}
```

| Code | Error | When |
|------|-------|------|
| 400 | `validation_error` | Invalid upload or artifact |
| 404 | `job_not_found` | Unknown job ID |
| 422 | `processing_error` | Pipeline failure |
| 500 | `internal_error` | Unexpected server error |
