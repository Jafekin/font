"""
Pre-processing module for ancient book image processing.
Provides functionality for structural element detection, distortion handling, and line information extraction.
"""

from .book_processor import AncientBookProcessor
from .structure_detector import StructureDetector
from .distortion_handler import DistortionHandler
from .line_info_extractor import LineInfoExtractor

__all__ = [
    'AncientBookProcessor',
    'StructureDetector',
    'DistortionHandler',
    'LineInfoExtractor',
]

