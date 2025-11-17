"""
Distortion handling for ancient book images.
Handles: stains (污渍), blurriness (模糊), and fold marks (折页) removal/mitigation.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DistortionHandler:
    """
    Handles various distortions in ancient book images using classical computer vision
    and optional deep learning techniques.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize the distortion handler.

        Args:
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.logger = logger

    def process_distortions(self, image: np.ndarray) -> Dict:
        """
        Process all types of distortions in the image.

        Args:
            image: Input image (numpy array, RGB)

        Returns:
            Dictionary containing:
            {
                'cleaned_image': cleaned image,
                'stain_mask': stain locations,
                'blur_map': blurriness regions,
                'fold_marks': fold locations,
                'quality_score': overall quality score,
            }
        """
        if len(image.shape) != 3:
            raise ValueError("Input image must be RGB")

        result = {}

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect stains
        stains = self._detect_stains(gray, image)
        result['stain_mask'] = stains['mask']
        result['stains_info'] = stains['info']

        # Detect blur regions
        blur_info = self._detect_blur(gray)
        result['blur_map'] = blur_info['map']
        result['blur_regions'] = blur_info['regions']

        # Detect fold marks
        folds = self._detect_fold_marks(gray)
        result['fold_marks'] = folds['marks']
        result['fold_regions'] = folds['regions']

        # Create cleaned image
        cleaned = self._clean_image(image, stains['mask'], folds['marks'])
        result['cleaned_image'] = cleaned

        # Calculate quality score
        quality_score = self._calculate_quality_score(stains, blur_info, folds)
        result['quality_score'] = quality_score

        return result

    def _detect_stains(self, gray: np.ndarray, image: np.ndarray) -> Dict:
        """
        Detect stains (污渍) in the image.
        Uses texture and color analysis.

        Args:
            gray: Grayscale image
            image: Original RGB image

        Returns:
            Dictionary with stain mask and information
        """
        h, w = gray.shape
        stain_mask = np.zeros((h, w), dtype=np.uint8)
        stains_info = []

        # Apply morphological opening to find dark stains
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Calculate difference (stains appear as dark areas)
        stain_diff = cv2.absdiff(gray, opened)

        # Threshold to get stain regions
        _, stain_binary = cv2.threshold(stain_diff, 30, 255, cv2.THRESH_BINARY)

        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        stain_binary = cv2.morphologyEx(stain_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours of stains
        contours, _ = cv2.findContours(stain_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter stains by size (too small = noise, too large = not a stain)
            if 20 < area < (h * w // 20):
                x, y, w_c, h_c = cv2.boundingRect(contour)
                cv2.drawContours(stain_mask, [contour], 0, 255, -1)

                stains_info.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w_c),
                    'height': int(h_c),
                    'area': int(area),
                    'intensity': float(np.mean(gray[y:y+h_c, x:x+w_c]))
                })

        return {
            'mask': stain_mask,
            'info': stains_info,
            'stain_count': len(stains_info)
        }

    def _detect_blur(self, gray: np.ndarray) -> Dict:
        """
        Detect blurry regions (模糊) using Laplacian variance.

        Args:
            gray: Grayscale image

        Returns:
            Dictionary with blur map and regions
        """
        h, w = gray.shape
        blur_map = np.zeros((h, w), dtype=np.float32)
        blur_regions = []

        # Compute Laplacian variance for blur detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Define blur threshold (lower variance = more blur)
        blur_threshold = 10  # Adjust based on requirements

        # Divide image into regions and check blur level
        region_size = 32

        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                y_end = min(y + region_size, h)
                x_end = min(x + region_size, w)

                region = gray[y:y_end, x:x_end]

                # Calculate Laplacian variance for this region
                lap = cv2.Laplacian(region, cv2.CV_64F)
                var = lap.var()

                blur_map[y:y_end, x:x_end] = var

                if var < blur_threshold:
                    blur_regions.append({
                        'x': x,
                        'y': y,
                        'width': x_end - x,
                        'height': y_end - y,
                        'variance': float(var),
                        'blur_level': 'high' if var < 5 else 'medium'
                    })

        # Normalize blur map
        blur_map = cv2.normalize(blur_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return {
            'map': blur_map,
            'regions': blur_regions,
            'blur_count': len(blur_regions),
            'overall_variance': float(variance)
        }

    def _detect_fold_marks(self, gray: np.ndarray) -> Dict:
        """
        Detect fold marks (折页) - crease lines in the page.

        Args:
            gray: Grayscale image

        Returns:
            Dictionary with fold marks and regions
        """
        marks = []
        regions = []

        # Edge detection to find strong lines (fold marks)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=gray.shape[0] // 4,
            maxLineGap=5
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line properties
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Check if this is a strong vertical line (likely a fold)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Fold marks typically run across the page
                if length > gray.shape[0] * 0.3 and dy > dx:
                    marks.append({
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'length': float(length),
                        'type': 'vertical'
                    })
                elif length > gray.shape[1] * 0.3 and dx > dy:
                    marks.append({
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'length': float(length),
                        'type': 'horizontal'
                    })

        # Detect fold regions (areas with high intensity variation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        grad_x = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        _, fold_binary = cv2.threshold(grad_x, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fold_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(cv2.contourArea(contour))
                })

        return {
            'marks': marks,
            'regions': regions,
            'mark_count': len(marks)
        }

    def _clean_image(self, image: np.ndarray, stain_mask: np.ndarray,
                     fold_marks: list) -> np.ndarray:
        """
        Clean image by reducing stains and fold mark effects.

        Args:
            image: Original RGB image
            stain_mask: Mask of detected stains
            fold_marks: List of detected fold marks

        Returns:
            Cleaned image
        """
        cleaned = image.copy()

        # Inpaint stains using OpenCV's inpainting
        if np.any(stain_mask > 0):
            for channel in range(3):
                cleaned[:, :, channel] = cv2.inpaint(
                    image[:, :, channel],
                    stain_mask,
                    3,
                    cv2.INPAINT_TELEA
                )

        # Enhance contrast
        lab = cv2.cvtColor(cleaned, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        enhanced = cv2.merge([l, a, b])
        cleaned = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return cleaned

    def _calculate_quality_score(self, stains: Dict, blur_info: Dict,
                                 folds: Dict) -> float:
        """
        Calculate overall image quality score (0-1).

        Args:
            stains: Stain detection results
            blur_info: Blur detection results
            folds: Fold detection results

        Returns:
            Quality score between 0 and 1
        """
        h, w = 1000, 1000  # Assume standard size for calculation

        # Score based on stains (fewer stains = higher score)
        stain_score = max(0, 1 - (stains['stain_count'] * 0.01))

        # Score based on blur (lower variance = lower score)
        blur_variance = blur_info['overall_variance']
        blur_score = min(1, blur_variance / 100)

        # Score based on folds
        fold_score = max(0, 1 - (folds['mark_count'] * 0.05))

        # Combine scores with weights
        quality_score = (stain_score * 0.4 + blur_score * 0.3 + fold_score * 0.3)

        return float(quality_score)

