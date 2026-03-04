"""
Text Detection Module for Vietnamese OCR
Automatically detects text regions in images for better processing
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image

class TextDetector:
    """Text region detection for improved OCR accuracy"""

    def __init__(self):
        self.min_area = 100  # Minimum area for text regions
        self.max_regions = 20  # Maximum number of regions to detect

    def detect_text_regions(self, image: Image.Image) -> List[Dict]:
        """
        Detect text regions in an image

        Args:
            image: PIL Image

        Returns:
            List of detected regions with bbox coordinates
        """
        # Convert to OpenCV format
        img_array = np.array(image.convert('L'))

        # Apply preprocessing for better detection
        processed = self._preprocess_for_detection(img_array)

        # Find contours
        contours = self._find_text_contours(processed)

        # Filter and convert to bounding boxes
        regions = []
        for contour in contours[:self.max_regions]:
            bbox = self._contour_to_bbox(contour)
            if self._is_valid_text_region(bbox, img_array.shape):
                regions.append({
                    'bbox': bbox,
                    'area': cv2.contourArea(contour),
                    'confidence': self._calculate_region_confidence(contour, processed)
                })

        # Sort by area (largest first)
        regions.sort(key=lambda x: x['area'], reverse=True)

        return regions

    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for text detection"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        return eroded

    def _find_text_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """Find contours that likely contain text"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area and aspect ratio
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            if 0.1 < aspect_ratio < 20:  # Reasonable text aspect ratio
                filtered_contours.append(contour)

        return filtered_contours

    def _contour_to_bbox(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert contour to bounding box (x, y, w, h)"""
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, x + w, y + h)  # Convert to (x1, y1, x2, y2)

    def _is_valid_text_region(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> bool:
        """Check if a bounding box represents a valid text region"""
        x1, y1, x2, y2 = bbox
        img_h, img_w = image_shape[:2]

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            return False

        # Check minimum size
        w, h = x2 - x1, y2 - y1
        if w < 10 or h < 8:
            return False

        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if not (0.1 <= aspect_ratio <= 50):
            return False

        return True

    def _calculate_region_confidence(self, contour: np.ndarray, binary_image: np.ndarray) -> float:
        """Calculate confidence score for a text region"""
        # Create mask for the contour
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Calculate solidity (area / convex hull area)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        solidity = area / hull_area if hull_area > 0 else 0

        # Higher solidity indicates more regular shapes (likely text)
        confidence = min(solidity * 2, 1.0)

        return confidence

    def merge_overlapping_regions(self, regions: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping text regions"""
        if len(regions) <= 1:
            return regions

        merged = []
        used = set()

        for i, region1 in enumerate(regions):
            if i in used:
                continue

            current_bbox = region1['bbox']
            current_area = region1['area']
            current_conf = region1['confidence']

            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue

                if self._bboxes_overlap(current_bbox, region2['bbox'], overlap_threshold):
                    # Merge bounding boxes
                    current_bbox = self._merge_bboxes(current_bbox, region2['bbox'])
                    current_area = max(current_area, region2['area'])
                    current_conf = max(current_conf, region2['confidence'])
                    used.add(j)

            merged.append({
                'bbox': current_bbox,
                'area': current_area,
                'confidence': current_conf
            })

        return merged

    def _bboxes_overlap(self, bbox1: Tuple, bbox2: Tuple, threshold: float) -> bool:
        """Check if two bounding boxes overlap significantly"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
            return False

        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

        # Calculate areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate overlap ratio
        union_area = area1 + area2 - inter_area
        overlap_ratio = inter_area / union_area if union_area > 0 else 0

        return overlap_ratio >= threshold

    def _merge_bboxes(self, bbox1: Tuple, bbox2: Tuple) -> Tuple:
        """Merge two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        return (
            min(x1_min, x2_min),
            min(y1_min, y2_min),
            max(x1_max, x2_max),
            max(y1_max, y2_max)
        )






