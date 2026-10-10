"""Backward-compatible import path; the implementation lives in :mod:`pygeovision.models.vlm.clip`.

This module used to hold a second, divergent ``CLIPGeo`` that mapped RemoteCLIP to Hugging Face
repositories that do not exist and loaded the checkpoints with the wrong library.
"""
from pygeovision.models.vlm.clip import CLIPGeo

__all__ = ["CLIPGeo"]
