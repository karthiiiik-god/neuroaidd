# src/__init__.py
# Re-export the helpers from the correct subpackages

from src.audio.feature_extractor import extract_audio_features  # noqa: F401
from src.video.landmarks import extract_frame_features          # noqa: F401

__all__ = ["extract_audio_features", "extract_frame_features"]
