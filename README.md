# Infrared Image Colorization and Enhancement

> Developed by Sahil Kedare

This project turns hard-to-read infrared images into clearer, colorized pictures. It is built to help people and machines understand satellite thermal imagery better.

## Problem Statement

Infrared satellite images are useful at night and in bad weather, but they come in a single grayscale channel.

- IR images have low contrast and poor detail.
- Objects such as buildings, roads, and vehicles are hard to identify.
- This makes analysis slow and unreliable for remote sensing tasks.

The image attached with the project shows this exact problem: satellite thermal data looks noisy and difficult to interpret.

## Solution Provided by This Project

This codebase solves the problem by:

- Enhancing infrared image quality with deep learning
- Translating IR input into realistic RGB output
- Improving contrast and structural detail
- Providing a full workflow from training to deployment

In short, it makes thermal images easier to read and use for object detection and monitoring.

## How It Works

1. `dataset/` prepares and loads infrared training data.
2. `training/` contains the model training code for learning colorization.
3. `inference/` runs predictions on new infrared images.
4. `api/` serves the model through a Flask endpoint.
5. `frontend/` lets users upload infrared images and see colorized results.

## Tech Stack

- Python 3.11+ for backend and model code
- PyTorch for training and inference
- Flask for REST API
- React, Vite, Tailwind CSS for the frontend UI
- YAML config files for experiment settings
- pytest for automated tests

## Folder Structure

| Path | Purpose |
| ------ | --------- |
| `configs/` | Experiment and runtime settings |
| `data/` | Raw and processed dataset files |
| `dataset/` | Data loading, augmentation, preprocessing |
| `models/` | Network architectures and saved model files |
| `training/` | Training loops, loss functions, evaluation |
| `inference/` | Prediction scripts and post-processing |
| `api/` | Backend routes, error handling, application factory |
| `frontend/` | Web UI for image upload and result preview |
| `scripts/` | Helper scripts for dataset setup and deployment |
| `tests/` | Tests for key project modules |
| `utils/` | Common configuration and logging utilities |
| `wsgi.py` | Production entrypoint for WSGI servers |

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
flask --app api.app run
```

```bash
cd frontend
npm install
npm run dev
```

## Notes

- The README text has been rewritten to match the hackathon problem more clearly.
- The project remains the same under the hood; only documentation has changed.
- Use the frontend to quickly test the IR colorization pipeline.
