"""WSGI entrypoint for production servers."""

from api.app import app

__all__ = ["app"]
