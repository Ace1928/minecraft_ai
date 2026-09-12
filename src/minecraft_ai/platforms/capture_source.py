from enum import StrEnum


class BedrockCaptureSource(StrEnum):
    """Supported capture preferences shared by CLI, child and capture factory."""

    PIPEWIRE = "pipewire"
    X11 = "x11"
