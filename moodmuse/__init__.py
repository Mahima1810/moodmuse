"""MoodMuse: find the mood and the themes in a pile of written feedback."""

from .pipeline import Analysis, analyze, to_markdown

__all__ = ["Analysis", "analyze", "to_markdown"]
__version__ = "0.1.0"
