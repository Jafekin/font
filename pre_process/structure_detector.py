"""
Structure detection for ancient book pages.
Detects: version heart (版心), column lines (栏线), page numbers, title labels (题签), etc.
"""

import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StructureDetector:
    """
    Detects and segments structural elements in ancient book pages.
    Uses edge detection and morphological operations for structure analysis.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize the structure detector.

        Args:
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.logger = logger

    def detect_structures(self, image: np.ndarray) -> Dict:
        """
        Detect all structural elements in the image.

        Args:
            image: Input image (numpy array, RGB or grayscale)

        Returns:
            Dictionary containing detected structures:
            {
                'version_heart': {coords, mask, confidence},
                'column_lines': [line1, line2, ...],
                'page_numbers': [pos1, pos2, ...],
                'title_labels': [label1, label2, ...],
                'borders': {type, positions},
                'text_regions': [region1, region2, ...],
            }
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        structures = {}

        # Detect version heart (主要的文字区域)
        structures['version_heart'] = self._detect_version_heart(gray)

        # Detect column lines (栏线)
        structures['column_lines'] = self._detect_column_lines(gray)

        # Detect page numbers (页码)
        structures['page_numbers'] = self._detect_page_numbers(gray, image)

        # Detect title labels (题签)
        structures['title_labels'] = self._detect_title_labels(gray)

        # Detect borders (边框)
        structures['borders'] = self._detect_borders(gray)

        # Detect text regions (文字区域)
        structures['text_regions'] = self._detect_text_regions(gray)

        return structures

    def _detect_version_heart(self, gray: np.ndarray) -> Dict:
        """
        Detect the version heart (版心) - main text area.

        Args:
            gray: Grayscale image

        Returns:
            Dictionary with version heart information
        """
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                'found': False,
                'coords': None,
                'mask': None,
                'confidence': 0.0
            }

        # Find the largest contour (likely the version heart)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)

        # Calculate confidence based on contour area ratio
        contour_area = cv2.contourArea(largest_contour)
        total_area = gray.shape[0] * gray.shape[1]
        confidence = min(contour_area / total_area * 2, 1.0)  # Normalize

        return {
            'found': True,
            'coords': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'mask': mask,
            'confidence': float(confidence)
        }

    def _detect_column_lines(self, gray: np.ndarray) -> List[Dict]:
        """
        Detect column lines (栏线) in the page.

        Args:
            gray: Grayscale image

        Returns:
            List of detected column lines with positions
        """
        lines = []

        # Use Hough line detection for straight lines
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines using HoughLinesP
        detected_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=gray.shape[0] // 4,
            maxLineGap=10
        )

        if detected_lines is not None:
            for line in detected_lines:
                x1, y1, x2, y2 = line[0]

                # Filter lines to get mostly vertical column lines
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dy > dx:  # Vertical lines
                    lines.append({
                        'type': 'vertical',
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'confidence': 0.8
                    })
                elif dx > dy:  # Horizontal lines
                    lines.append({
                        'type': 'horizontal',
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'confidence': 0.8
                    })

        return lines

    def _detect_page_numbers(self, gray: np.ndarray, image: np.ndarray) -> List[Dict]:
        """
        Detect page numbers (页码) locations.

        Args:
            gray: Grayscale image
            image: Original image

        Returns:
            List of detected page number positions
        """
        page_numbers = []

        # Page numbers typically appear in corners or margins
        h, w = gray.shape

        # Define search regions (corners and margins)
        regions = [
            {'name': 'top_left', 'x': 0, 'y': 0, 'w': w // 4, 'h': h // 6},
            {'name': 'top_right', 'x': 3 * w // 4, 'y': 0, 'w': w // 4, 'h': h // 6},
            {'name': 'bottom_left', 'x': 0, 'y': 5 * h // 6, 'w': w // 4, 'h': h // 6},
            {'name': 'bottom_right', 'x': 3 * w // 4, 'y': 5 * h // 6, 'w': w // 4, 'h': h // 6},
        ]

        for region in regions:
            roi = gray[region['y']:region['y']+region['h'],
                       region['x']:region['x']+region['w']]

            # Apply thresholding to find dark regions (text)
            _, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)

            # Find contours in this region
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Check if there are meaningful contours (likely text)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 10 < area < 1000:  # Typical size for page numbers
                        x, y, w_c, h_c = cv2.boundingRect(contour)
                        page_numbers.append({
                            'region': region['name'],
                            'x': region['x'] + x,
                            'y': region['y'] + y,
                            'width': w_c,
                            'height': h_c,
                            'confidence': 0.6
                        })

        return page_numbers

    def _detect_title_labels(self, gray: np.ndarray) -> List[Dict]:
        """
        Detect title labels (题签) - usually in upper margins.

        Args:
            gray: Grayscale image

        Returns:
            List of detected title label positions
        """
        title_labels = []

        # Title labels typically appear in the top margin
        h, w = gray.shape
        top_region = gray[0:h // 8, :]

        # Find dark horizontal stripes (title areas)
        _, thresh = cv2.threshold(top_region, 150, 255, cv2.THRESH_BINARY_INV)

        # Dilate to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum size for title label
                x, y, w_c, h_c = cv2.boundingRect(contour)
                title_labels.append({
                    'x': x,
                    'y': y,
                    'width': w_c,
                    'height': h_c,
                    'confidence': 0.7
                })

        return title_labels

    def _detect_borders(self, gray: np.ndarray) -> Dict:
        """
        Detect page borders and determine if they are single or double.

        Args:
            gray: Grayscale image

        Returns:
            Dictionary with border information
        """
        h, w = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find edges at the borders
        top_edge = np.sum(edges[0:h//10, :])
        bottom_edge = np.sum(edges[9*h//10:h, :])
        left_edge = np.sum(edges[:, 0:w//10])
        right_edge = np.sum(edges[:, 9*w//10:w])

        # Detect if borders are present
        edge_threshold = (h + w) * 5

        borders = {
            'top': top_edge > edge_threshold,
            'bottom': bottom_edge > edge_threshold,
            'left': left_edge > edge_threshold,
            'right': right_edge > edge_threshold,
        }

        # Determine border type (single vs double)
        # Analyze horizontal and vertical lines in border regions
        border_type = self._determine_border_type(gray, edges)
        borders['type'] = border_type

        return borders

    def _determine_border_type(self, gray: np.ndarray, edges: np.ndarray) -> str:
        """
        Determine if borders are single or double.

        Args:
            gray: Grayscale image
            edges: Edge-detected image

        Returns:
            'single' or 'double' or 'unknown'
        """
        h, w = gray.shape

        # Analyze top border
        top_region = edges[0:h//6, :]

        # Count vertical line groups
        vertical_lines = np.sum(top_region, axis=0)
        threshold = np.mean(vertical_lines) + np.std(vertical_lines)

        line_pixels = np.where(vertical_lines > threshold)[0]

        if len(line_pixels) == 0:
            return 'unknown'

        # Find gaps between line groups
        gaps = np.diff(line_pixels)
        large_gaps = np.where(gaps > 5)[0]

        if len(large_gaps) > 1:
            return 'double'
        else:
            return 'single'

    def _detect_text_regions(self, gray: np.ndarray) -> List[Dict]:
        """
        Detect text regions for better understanding of layout.

        Args:
            gray: Grayscale image

        Returns:
            List of detected text regions
        """
        text_regions = []

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by reasonable text region sizes
            if 100 < area < (gray.shape[0] * gray.shape[1] // 2):
                x, y, w, h = cv2.boundingRect(contour)
                text_regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area)
                })

        # Sort by y coordinate (top to bottom)
        text_regions.sort(key=lambda r: r['y'])

        return text_regions

