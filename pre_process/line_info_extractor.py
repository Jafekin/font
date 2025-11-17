"""
Line information extraction for ancient book pages.
Extracts: number of lines (行数), characters per line (每行字数), border type (边框).
"""

import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LineInfoExtractor:
    """
    Extracts line information and layout details from ancient book pages.
    Determines number of lines, characters per line, and border configuration.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize the line info extractor.

        Args:
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.logger = logger

    def extract_line_info(self, image: np.ndarray,
                         text_regions: Optional[List[Dict]] = None) -> Dict:
        """
        Extract comprehensive line information from the page.

        Args:
            image: Input image (numpy array, RGB or grayscale)
            text_regions: Optional pre-detected text regions

        Returns:
            Dictionary containing:
            {
                'num_lines': number of lines,
                'chars_per_line': estimated characters per line,
                'line_positions': [y1, y2, ...],
                'line_details': [line1_info, line2_info, ...],
                'borders': border configuration,
                'layout_type': layout type (single/multi-column),
                'char_size_estimate': estimated character size,
                'line_spacing': average line spacing,
            }
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        result = {}

        # Get text regions if not provided
        if text_regions is None:
            text_regions = self._extract_text_regions(gray)

        # Extract line positions
        line_info = self._extract_lines(gray, text_regions)
        result['num_lines'] = line_info['num_lines']
        result['line_positions'] = line_info['positions']
        result['line_details'] = line_info['details']
        result['line_spacing'] = line_info['avg_spacing']

        # Extract characters per line
        char_info = self._extract_chars_per_line(gray, text_regions, line_info)
        result['chars_per_line'] = char_info['chars_per_line']
        result['char_size_estimate'] = char_info['char_size']
        result['char_distribution'] = char_info['distribution']

        # Detect borders
        border_info = self._detect_border_type(gray)
        result['borders'] = border_info

        # Determine layout type
        layout_type = self._determine_layout_type(text_regions, line_info)
        result['layout_type'] = layout_type

        # Calculate confidence scores
        result['confidence'] = self._calculate_confidence(line_info, char_info)

        return result

    def _extract_text_regions(self, gray: np.ndarray) -> List[Dict]:
        """
        Extract text regions from the image.

        Args:
            gray: Grayscale image

        Returns:
            List of text region information
        """
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by reasonable text region sizes
            if 20 < area < (gray.shape[0] * gray.shape[1] // 5):
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area),
                    'aspect_ratio': float(w / h) if h > 0 else 1.0
                })

        # Sort by y coordinate
        regions.sort(key=lambda r: r['y'])

        return regions

    def _extract_lines(self, gray: np.ndarray,
                      text_regions: List[Dict]) -> Dict:
        """
        Extract line positions and information.

        Args:
            gray: Grayscale image
            text_regions: List of text regions

        Returns:
            Dictionary with line information
        """
        if not text_regions:
            return {
                'num_lines': 0,
                'positions': [],
                'details': [],
                'avg_spacing': 0
            }

        # Extract y-coordinates of text regions to group into lines
        y_coords = [region['y'] for region in text_regions]

        # Group regions into lines based on y-coordinate clustering
        lines = self._cluster_into_lines(text_regions)

        # Calculate line spacing
        if len(lines) > 1:
            spacings = []
            for i in range(len(lines) - 1):
                spacing = lines[i + 1]['y_center'] - lines[i]['y_center']
                spacings.append(spacing)
            avg_spacing = np.mean(spacings) if spacings else 0
        else:
            avg_spacing = 0

        # Prepare line details
        line_details = []
        line_positions = []

        for i, line in enumerate(lines):
            line_pos_y = int(line['y_center'])
            line_positions.append(line_pos_y)

            line_details.append({
                'line_id': i,
                'y_position': line_pos_y,
                'y_min': int(line['y_min']),
                'y_max': int(line['y_max']),
                'height': int(line['height']),
                'char_count': len(line['regions']),
                'confidence': line['confidence']
            })

        return {
            'num_lines': len(lines),
            'positions': line_positions,
            'details': line_details,
            'avg_spacing': float(avg_spacing)
        }

    def _cluster_into_lines(self, text_regions: List[Dict],
                           threshold: float = 0.2) -> List[Dict]:
        """
        Cluster text regions into lines based on y-coordinate proximity.

        Args:
            text_regions: List of text regions
            threshold: Y-distance threshold for clustering (as fraction of region height)

        Returns:
            List of clustered lines
        """
        if not text_regions:
            return []

        lines = []
        current_line_regions = [text_regions[0]]

        for i in range(1, len(text_regions)):
            region = text_regions[i]
            last_region = current_line_regions[-1]

            # Calculate vertical distance
            y_distance = abs(region['y'] - last_region['y'])

            # If close in y-coordinate, add to current line
            avg_height = (region['height'] + last_region['height']) / 2
            if y_distance < avg_height * threshold:
                current_line_regions.append(region)
            else:
                # Start a new line
                lines.append(self._create_line_info(current_line_regions))
                current_line_regions = [region]

        # Don't forget the last line
        if current_line_regions:
            lines.append(self._create_line_info(current_line_regions))

        return lines

    def _create_line_info(self, regions: List[Dict]) -> Dict:
        """
        Create line information from grouped regions.

        Args:
            regions: List of regions in the line

        Returns:
            Line information dictionary
        """
        y_coords = [r['y'] for r in regions]
        y_max_coords = [r['y'] + r['height'] for r in regions]

        y_min = min(y_coords)
        y_max = max(y_max_coords)
        y_center = (y_min + y_max) / 2
        height = y_max - y_min

        # Calculate confidence (based on alignment)
        y_variance = np.var(y_coords)
        confidence = 1.0 if y_variance < 5 else max(0.5, 1.0 - y_variance / 100)

        return {
            'y_min': y_min,
            'y_max': y_max,
            'y_center': y_center,
            'height': height,
            'regions': regions,
            'confidence': float(confidence)
        }

    def _extract_chars_per_line(self, gray: np.ndarray,
                               text_regions: List[Dict],
                               line_info: Dict) -> Dict:
        """
        Extract character count and size information.

        Args:
            gray: Grayscale image
            text_regions: List of text regions
            line_info: Line information from _extract_lines

        Returns:
            Dictionary with character information
        """
        if not text_regions or line_info['num_lines'] == 0:
            return {
                'chars_per_line': 0,
                'char_size': 0,
                'distribution': []
            }

        # Sort regions by x-coordinate to get left-to-right order
        sorted_regions = sorted(text_regions, key=lambda r: r['x'])

        # Estimate character size
        char_widths = [r['width'] for r in sorted_regions]
        char_heights = [r['height'] for r in sorted_regions]

        if char_widths and char_heights:
            avg_char_width = np.mean(char_widths)
            avg_char_height = np.mean(char_heights)
            char_size = int((avg_char_width + avg_char_height) / 2)
        else:
            char_size = 0

        # Count characters per line
        chars_per_line_list = []
        for line_detail in line_info['details']:
            chars_per_line_list.append(line_detail['char_count'])

        # Calculate average and mode
        if chars_per_line_list:
            avg_chars = np.mean(chars_per_line_list)
            mode_chars = max(set(chars_per_line_list), key=chars_per_line_list.count)
        else:
            avg_chars = 0
            mode_chars = 0

        return {
            'chars_per_line': int(mode_chars),  # Use mode as typical value
            'chars_per_line_avg': float(avg_chars),
            'char_size': char_size,
            'distribution': chars_per_line_list
        }

    def _detect_border_type(self, gray: np.ndarray) -> Dict:
        """
        Detect and classify border type (single or double).

        Args:
            gray: Grayscale image

        Returns:
            Dictionary with border information
        """
        h, w = gray.shape

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Analyze border regions
        border_thickness = min(h, w) // 10

        borders = {}

        # Top border
        top_region = edges[0:border_thickness, :]
        top_lines = self._count_parallel_lines(top_region, 'horizontal')
        borders['top'] = {
            'detected': top_lines > 0,
            'type': 'double' if top_lines > 1 else 'single',
            'line_count': top_lines
        }

        # Bottom border
        bottom_region = edges[h-border_thickness:h, :]
        bottom_lines = self._count_parallel_lines(bottom_region, 'horizontal')
        borders['bottom'] = {
            'detected': bottom_lines > 0,
            'type': 'double' if bottom_lines > 1 else 'single',
            'line_count': bottom_lines
        }

        # Left border
        left_region = edges[:, 0:border_thickness]
        left_lines = self._count_parallel_lines(left_region, 'vertical')
        borders['left'] = {
            'detected': left_lines > 0,
            'type': 'double' if left_lines > 1 else 'single',
            'line_count': left_lines
        }

        # Right border
        right_region = edges[:, w-border_thickness:w]
        right_lines = self._count_parallel_lines(right_region, 'vertical')
        borders['right'] = {
            'detected': right_lines > 0,
            'type': 'double' if right_lines > 1 else 'single',
            'line_count': right_lines
        }

        # Determine overall border configuration
        border_config = self._determine_border_config(borders)
        borders['configuration'] = border_config

        return borders

    def _count_parallel_lines(self, edge_region: np.ndarray,
                             direction: str) -> int:
        """
        Count parallel lines in a region.

        Args:
            edge_region: Edge-detected region
            direction: 'horizontal' or 'vertical'

        Returns:
            Number of parallel lines detected
        """
        if direction == 'horizontal':
            # Sum along columns to find horizontal lines
            line_strength = np.sum(edge_region, axis=1)
        else:
            # Sum along rows to find vertical lines
            line_strength = np.sum(edge_region, axis=0)

        # Find peaks (strong lines)
        threshold = np.mean(line_strength) + np.std(line_strength)
        peaks = np.where(line_strength > threshold)[0]

        if len(peaks) == 0:
            return 0

        # Group consecutive peaks
        line_count = 0
        in_peak = False

        for peak in peaks:
            if not in_peak:
                line_count += 1
                in_peak = True
            # Continue in peak if next peak is close
            elif len(peaks) > 1:
                in_peak = True

        return min(line_count, 2)  # Cap at 2 for single/double detection

    def _determine_border_config(self, borders: Dict) -> str:
        """
        Determine overall border configuration.

        Args:
            borders: Border information for all sides

        Returns:
            Border configuration string
        """
        sides = ['top', 'bottom', 'left', 'right']
        detected_count = sum(1 for side in sides if borders[side]['detected'])

        # Check if borders are mostly single or double
        border_types = [borders[side]['type'] for side in sides if borders[side]['detected']]

        if not border_types:
            return 'no_border'

        double_count = border_types.count('double')

        if double_count >= len(border_types) / 2:
            if detected_count == 4:
                return 'complete_double'
            else:
                return 'partial_double'
        else:
            if detected_count == 4:
                return 'complete_single'
            elif detected_count == 2:
                return 'partial_single'
            else:
                return 'minimal'

    def _determine_layout_type(self, text_regions: List[Dict],
                              line_info: Dict) -> str:
        """
        Determine page layout type (single-column, multi-column, etc.).

        Args:
            text_regions: List of text regions
            line_info: Line information

        Returns:
            Layout type string
        """
        if not text_regions:
            return 'unknown'

        # Analyze horizontal distribution of regions
        x_coords = [r['x'] for r in text_regions]

        # Use k-means like approach to detect columns
        x_sorted = sorted(x_coords)

        # Look for gaps in x distribution
        if len(x_sorted) > 1:
            gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
            large_gaps = sum(1 for gap in gaps if gap > np.mean(gaps) * 2)

            if large_gaps >= 1:
                return 'multi_column'

        return 'single_column'

    def _calculate_confidence(self, line_info: Dict, char_info: Dict) -> float:
        """
        Calculate overall confidence score for extraction.

        Args:
            line_info: Line extraction information
            char_info: Character extraction information

        Returns:
            Confidence score between 0 and 1
        """
        scores = []

        # Confidence based on line detection
        if line_info['details']:
            avg_line_confidence = np.mean([d['confidence'] for d in line_info['details']])
            scores.append(avg_line_confidence)

        # Confidence based on character consistency
        if char_info['distribution']:
            char_variance = np.var(char_info['distribution'])
            char_consistency = 1.0 / (1.0 + char_variance / 100)
            scores.append(char_consistency)

        # Overall confidence
        if scores:
            return float(np.mean(scores))
        else:
            return 0.0

