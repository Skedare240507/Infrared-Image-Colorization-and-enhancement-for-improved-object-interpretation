"""Flask application factory and entrypoint."""

from __future__ import annotations

import logging

from flask import Flask
from flask_cors import CORS

from api.config import load_app_config
from api.errors import register_error_handlers
from api.middleware import register_request_logging
from api.routes import health, thermal
from api.services.processing_service import ProcessingService
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app(config_override=None) -> Flask:
    config = config_override or load_app_config()
    setup_logging(log_dir=config.log_dir, level=config.log_level)

    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.secret_key
    app.config["DEBUG"] = config.debug
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length

    CORS(app)
    register_error_handlers(app)
    register_request_logging(app)

    app.extensions["processing_service"] = ProcessingService(config)

    app.register_blueprint(health.bp)
    app.register_blueprint(thermal.bp, url_prefix="/api/v1")

    logger.info("Flask application initialized")
    return app


app = create_app()
