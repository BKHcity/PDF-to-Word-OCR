"""
Table Structure Extractor - Extract table structure using image analysis and OCR positions
Trích xuất cấu trúc bảng sử dụng phân tích hình ảnh và vị trí OCR
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import re


class TableStructureExtractor:
    """Extract and organize table structure using cell positions"""
    
    def __init__(self):
        self.cell_threshold = 10  # Pixels to cluster cells in same row/column
    
    def extract_structure_from_image(self, table_image: Image.Image, 
                                    ocr_system=None, 
                                    engines: List[str] = None) -> Tuple[List[List[str]], str]:
        """
        Extract table structure by analyzing image and OCR text positions
        
        Args:
            table_image: PIL Image of table
            ocr_system: OCR system with position detection
            engines: List of OCR engines to use
            
        Returns:
            Tuple of (structured data 2D list, format type)
        """
        if ocr_system is None:
            # Fallback to basic structure extraction
            return self._extract_from_text_fallback(table_image)
        
        # OCR with detailed results (should include bounding boxes)
        try:
            ocr_result = ocr_system.recognize(
                table_image,
                engines=engines or ['paddleocr'],
                return_details=True  # Request detailed results with positions
            )
            
            # Try to extract positions from OCR result
            if hasattr(ocr_result, 'details') and ocr_result.details:
                return self._organize_by_positions(ocr_result.details, table_image)
            elif hasattr(ocr_result, 'positions') and ocr_result.positions:
                return self._organize_by_positions(ocr_result.positions, table_image)
        except Exception as e:
            print(f"Error in OCR with positions: {e}")
        
        # Fallback
        return self._extract_from_text_fallback(table_image)
    
    def _organize_by_positions(self, ocr_results: List[Dict], 
                               table_image: Image.Image) -> Tuple[List[List[str]], str]:
        """
        Organize OCR results by their positions in the image
        
        Args:
            ocr_results: List of OCR results with position info
            table_image: Original table image
            
        Returns:
            Organized table structure
        """
        if not ocr_results:
            return [], 'empty'
        
        # Extract text with positions
        items = []
        for result in ocr_results:
            if isinstance(result, dict):
                text = result.get('text', '')
                bbox = result.get('bbox', result.get('box', None))
            else:
                # Try to unpack result
                try:
                    text, bbox = result[0], result[1]
                except (TypeError, IndexError):
                    continue
            
            if text and text.strip():
                items.append({
                    'text': text.strip(),
                    'bbox': bbox
                })
        
        if not items:
            return [], 'empty'
        
        # Extract coordinates
        positions = []
        for item in items:
            try:
                bbox = item['bbox']
                # Bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or (x1,y1,x2,y2)
                if isinstance(bbox, list) and len(bbox) == 4:
                    if isinstance(bbox[0], (list, tuple)):
                        # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] format
                        x_coords = [p[0] for p in bbox]
                        y_coords = [p[1] for p in bbox]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                    else:
                        # (x1,y1,x2,y2) format
                        x_min, y_min, x_max, y_max = bbox
                else:
                    continue
                
                positions.append({
                    'text': item['text'],
                    'x': (x_min + x_max) / 2,
                    'y': (y_min + y_max) / 2,
                    'y_top': y_min,
                    'y_bottom': y_max,
                    'x_left': x_min,
                    'x_right': x_max
                })
            except (TypeError, ValueError, IndexError):
                continue
        
        if not positions:
            return [], 'empty'
        
        # Sort by Y position (top to bottom) then X position (left to right)
        positions.sort(key=lambda p: (p['y_top'], p['x']))
        
        # Group by rows (similar Y position)
        rows = []
        current_row = []
        current_row_y = None
        
        for item in positions:
            y = item['y_top']
            
            # If Y position differs significantly, start new row
            if current_row_y is None:
                current_row_y = y
            elif abs(y - current_row_y) > self.cell_threshold:
                # Save current row and start new one
                if current_row:
                    # Sort by X position within row
                    current_row.sort(key=lambda x: x['x'])
                    rows.append([item['text'] for item in current_row])
                current_row = []
                current_row_y = y
            
            current_row.append(item)
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda x: x['x'])
            rows.append([item['text'] for item in current_row])
        
        if not rows:
            return [], 'empty'
        
        return rows, 'structured'
    
    def _extract_from_text_fallback(self, table_image: Image.Image) -> Tuple[List[List[str]], str]:
        """
        Fallback method: Extract table structure from image by detecting lines
        
        Args:
            table_image: PIL Image of table
            
        Returns:
            Organized table structure
        """
        # Convert to OpenCV
        cv_image = cv2.cvtColor(np.array(table_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_horizontal_lines(gray)
        vertical_lines = self._detect_vertical_lines(gray)
        
        if not horizontal_lines or not vertical_lines:
            return [], 'no_structure'
        
        # Use lines to define cell boundaries
        h_lines = sorted(horizontal_lines)
        v_lines = sorted(vertical_lines)
        
        # Try simple grid-based OCR
        # This is a fallback - ideally would have proper OCR here
        return [], 'fallback'
    
    def _detect_horizontal_lines(self, gray: np.ndarray) -> List[int]:
        """Detect horizontal lines in image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        _, binary = cv2.threshold(morph, 150, 255, cv2.THRESH_BINARY)
        
        # Find horizontal line positions
        h_sum = cv2.reduce(binary, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        lines = []
        for i in range(len(h_sum)):
            if h_sum[i][0] > 100:  # Threshold for line presence
                lines.append(i)
        
        # Cluster nearby lines
        return self._cluster_lines(lines)
    
    def _detect_vertical_lines(self, gray: np.ndarray) -> List[int]:
        """Detect vertical lines in image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        _, binary = cv2.threshold(morph, 150, 255, cv2.THRESH_BINARY)
        
        # Find vertical line positions
        v_sum = cv2.reduce(binary, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        lines = []
        for i in range(len(v_sum[0])):
            if v_sum[0][i] > 100:  # Threshold for line presence
                lines.append(i)
        
        # Cluster nearby lines
        return self._cluster_lines(lines)
    
    def _cluster_lines(self, lines: List[int], threshold: int = 5) -> List[int]:
        """Cluster nearby line positions"""
        if not lines:
            return []
        
        clustered = []
        current_cluster = [lines[0]]
        
        for line in lines[1:]:
            if line - current_cluster[-1] <= threshold:
                current_cluster.append(line)
            else:
                # Save average of cluster
                clustered.append(int(np.mean(current_cluster)))
                current_cluster = [line]
        
        if current_cluster:
            clustered.append(int(np.mean(current_cluster)))
        
        return clustered


def extract_table_from_cells(table_image: Image.Image, 
                            cells_data: List[Dict]) -> List[List[str]]:
    """
    Build table from cell data with positions
    
    Args:
        table_image: PIL Image
        cells_data: List of dicts with 'text' and 'bbox' keys
        
    Returns:
        2D list table structure
    """
    extractor = TableStructureExtractor()
    structured, _ = extractor._organize_by_positions(cells_data, table_image)
    return structured
