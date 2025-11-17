"""
Main ancient book processor - orchestrates all preprocessing steps.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Optional, Union
from PIL import Image
import logging
import io

from .structure_detector import StructureDetector
from .distortion_handler import DistortionHandler
from .line_info_extractor import LineInfoExtractor

logger = logging.getLogger(__name__)


class AncientBookProcessor:
    """
    Main processor for ancient book pages.
    Orchestrates structure detection, distortion handling, and line extraction.
    """

    def __init__(self, device: str = 'cpu', use_cuda: bool = False):
        """
        Initialize the ancient book processor.

        Args:
            device: Device to use ('cpu' or 'cuda')
            use_cuda: Whether to use CUDA (will auto-detect if available)
        """
        # Auto-detect CUDA if requested
        if use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = device if device != 'cuda' or torch.cuda.is_available() else 'cpu'

        self.logger = logger

        # Initialize sub-processors
        self.structure_detector = StructureDetector(device=self.device)
        self.distortion_handler = DistortionHandler(device=self.device)
        self.line_extractor = LineInfoExtractor(device=self.device)

        self.logger.info(f"AncientBookProcessor initialized with device: {self.device}")

    def process(self, image: Union[str, np.ndarray, Image.Image, io.BytesIO],
               full_analysis: bool = True) -> Dict:
        """
        Process an ancient book page image.

        Args:
            image: Input image (file path, numpy array, PIL Image, or BytesIO)
            full_analysis: Whether to perform all analyses or just basic ones

        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            # Load and prepare image
            img_array = self._load_image(image)

            self.logger.info(f"Processing image of size: {img_array.shape}")

            result = {
                'success': True,
                'image_shape': img_array.shape,
                'device': self.device,
            }

            # Step 1: Detect structures
            self.logger.info("Detecting structural elements...")
            structures = self.structure_detector.detect_structures(img_array)
            result['structures'] = structures

            # Step 2: Handle distortions
            self.logger.info("Processing distortions...")
            distortions = self.distortion_handler.process_distortions(img_array)
            result['distortions'] = distortions

            # Use cleaned image for further processing
            cleaned_image = distortions['cleaned_image']

            # Step 3: Extract line information
            self.logger.info("Extracting line information...")
            text_regions = structures['text_regions']
            line_info = self.line_extractor.extract_line_info(cleaned_image, text_regions)
            result['line_info'] = line_info

            # Step 4: Generate summary and recommendations
            summary = self._generate_summary(structures, distortions, line_info)
            result['summary'] = summary

            # Optional: Full detailed analysis
            if full_analysis:
                detailed_analysis = self._generate_detailed_analysis(
                    structures, distortions, line_info, img_array
                )
                result['detailed_analysis'] = detailed_analysis

            result['timestamp'] = self._get_timestamp()

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            result = {
                'success': False,
                'error': str(e),
                'timestamp': self._get_timestamp()
            }

        return result

    def _load_image(self, image: Union[str, np.ndarray, Image.Image, io.BytesIO]) -> np.ndarray:
        """
        Load image from various formats.

        Args:
            image: Input image in various formats

        Returns:
            Image as numpy array in RGB format
        """
        if isinstance(image, np.ndarray):
            # Already numpy array
            if len(image.shape) == 2:
                # Grayscale, convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                return image
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                return image

        elif isinstance(image, Image.Image):
            # PIL Image
            return np.array(image.convert('RGB'))

        elif isinstance(image, io.BytesIO):
            # BytesIO stream
            image.seek(0)
            pil_image = Image.open(image)
            return np.array(pil_image.convert('RGB'))

        elif isinstance(image, str):
            # File path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _generate_summary(self, structures: Dict, distortions: Dict,
                         line_info: Dict) -> Dict:
        """
        Generate a summary of the analysis results.

        Args:
            structures: Structure detection results
            distortions: Distortion handling results
            line_info: Line extraction results

        Returns:
            Summary dictionary
        """
        summary = {
            'page_type': self._determine_page_type(structures),
            'image_quality': {
                'overall_score': distortions['quality_score'],
                'stain_count': distortions['stains_info'].__len__(),
                'blur_regions': len(distortions['blur_regions']),
                'fold_marks': len(distortions['fold_marks']),
            },
            'layout_info': {
                'layout_type': line_info['layout_type'],
                'num_lines': line_info['num_lines'],
                'chars_per_line': line_info['chars_per_line'],
                'estimated_char_size': line_info['char_size_estimate'],
            },
            'structural_elements': {
                'has_version_heart': structures['version_heart']['found'],
                'column_line_count': len(structures['column_lines']),
                'page_number_locations': len(structures['page_numbers']),
                'title_labels': len(structures['title_labels']),
                'border_type': structures['borders'].get('type', 'unknown'),
            },
            'recommendations': self._generate_recommendations(structures, distortions, line_info)
        }

        return summary

    def _determine_page_type(self, structures: Dict) -> str:
        """Determine page type based on structural elements."""
        has_columns = len(structures['column_lines']) > 0
        has_borders = any(structures['borders'].values())
        has_title = len(structures['title_labels']) > 0

        if has_title and has_borders:
            return 'title_page'
        elif has_columns:
            return 'multi_column_text'
        elif structures['version_heart']['found']:
            return 'single_column_text'
        else:
            return 'unknown'

    def _generate_recommendations(self, structures: Dict, distortions: Dict,
                                 line_info: Dict) -> list:
        """Generate recommendations for image improvement."""
        recommendations = []

        # Quality-based recommendations
        if distortions['quality_score'] < 0.6:
            recommendations.append("低质量图片：建议重新扫描或优化")

        # Stain-based recommendations
        if len(distortions['stains_info']) > 5:
            recommendations.append(f"检测到{len(distortions['stains_info'])}个污渍，建议使用去污工具处理")

        # Blur-based recommendations
        if len(distortions['blur_regions']) > 3:
            recommendations.append("检测到多个模糊区域，建议检查原始扫描质量")

        # Fold-based recommendations
        if len(distortions['fold_marks']) > 0:
            recommendations.append(f"检测到{len(distortions['fold_marks'])}条折痕，建议使用展平工具")

        # Layout recommendations
        if line_info['num_lines'] == 0:
            recommendations.append("未能检测到文字行，请检查图片内容")

        # Border recommendations
        if not any(structures['borders'].values()):
            recommendations.append("未检测到边框，可能是装裱页或特殊页面")

        if not recommendations:
            recommendations.append("图片质量良好，可以进行进一步的文字识别")

        return recommendations

    def _generate_detailed_analysis(self, structures: Dict, distortions: Dict,
                                   line_info: Dict, image: np.ndarray) -> Dict:
        """Generate detailed analysis for each component."""
        return {
            'structure_analysis': {
                'version_heart_confidence': structures['version_heart']['confidence'],
                'column_lines_details': structures['column_lines'],
                'page_numbers_details': structures['page_numbers'],
            },
            'distortion_analysis': {
                'stains_distribution': self._analyze_stain_distribution(distortions),
                'blur_map_stats': self._analyze_blur_stats(distortions),
                'fold_patterns': self._analyze_fold_patterns(distortions),
            },
            'line_analysis': {
                'line_spacing_uniformity': self._analyze_line_spacing(line_info),
                'character_size_distribution': line_info['char_distribution'],
                'border_configuration': line_info['borders'],
            }
        }

    def _analyze_stain_distribution(self, distortions: Dict) -> Dict:
        """Analyze stain distribution patterns."""
        stains = distortions['stains_info']
        if not stains:
            return {'pattern': 'none', 'density': 0.0}

        intensities = [s['intensity'] for s in stains]
        return {
            'pattern': 'scattered' if len(stains) > 3 else 'clustered',
            'density': len(stains) / 100,
            'avg_intensity': float(np.mean(intensities)),
            'intensity_range': [float(min(intensities)), float(max(intensities))]
        }

    def _analyze_blur_stats(self, distortions: Dict) -> Dict:
        """Analyze blur statistics."""
        regions = distortions['blur_regions']
        if not regions:
            return {'status': 'no_blur', 'coverage': 0.0}

        variances = [r['variance'] for r in regions]
        return {
            'status': 'significant_blur' if len(regions) > 5 else 'minor_blur',
            'coverage': len(regions) * 0.01,  # Estimate coverage
            'avg_variance': float(np.mean(variances)),
            'region_count': len(regions)
        }

    def _analyze_fold_patterns(self, distortions: Dict) -> Dict:
        """Analyze fold patterns."""
        marks = distortions['fold_marks']
        regions = distortions['fold_regions']

        if not marks:
            return {'pattern': 'no_folds', 'count': 0}

        vertical_folds = sum(1 for m in marks if m['type'] == 'vertical')
        horizontal_folds = sum(1 for m in marks if m['type'] == 'horizontal')

        return {
            'pattern': 'complex' if len(marks) > 3 else 'simple',
            'vertical_folds': vertical_folds,
            'horizontal_folds': horizontal_folds,
            'total_marks': len(marks),
            'affected_regions': len(regions)
        }

    def _analyze_line_spacing(self, line_info: Dict) -> Dict:
        """Analyze line spacing uniformity."""
        details = line_info['line_details']
        if len(details) < 2:
            return {'uniformity': 1.0, 'consistency': 'high'}

        spacings = []
        for i in range(len(details) - 1):
            spacing = details[i + 1]['y_position'] - details[i]['y_position']
            spacings.append(spacing)

        variance = np.var(spacings) if spacings else 0
        uniformity = 1.0 / (1.0 + variance / 100)

        return {
            'uniformity': float(uniformity),
            'consistency': 'high' if uniformity > 0.8 else 'medium' if uniformity > 0.6 else 'low',
            'avg_spacing': float(np.mean(spacings)) if spacings else 0,
            'spacing_variance': float(variance)
        }

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_device_info(self) -> Dict:
        """Get device and capability information."""
        return {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'pytorch_version': torch.__version__,
            'opencv_version': cv2.__version__,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

