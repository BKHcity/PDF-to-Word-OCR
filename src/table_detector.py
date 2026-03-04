"""
Table Detection Module - Phát hiện bảng biểu trong ảnh và PDF
Sử dụng OpenCV + PaddleOCR Layout Analysis để phát hiện và trích xuất bảng
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw
import io


@dataclass
class Table:
    """Table detection result"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    cells: Optional[List[Dict]] = None
    table_image: Optional[Image.Image] = None
    text_content: Optional[str] = None


@dataclass
class TableDetectionResult:
    """Table detection results for an image"""
    image: Image.Image
    tables: List[Table]
    has_tables: bool
    table_count: int


class TableDetector:
    """Detect tables in images using OpenCV"""
    
    def __init__(self, min_table_area: int = 5000, line_threshold: int = 100):
        """
        Initialize table detector
        
        Args:
            min_table_area: Minimum area to be considered a table (pixels²)
            line_threshold: Threshold for detecting lines in table
        """
        self.min_table_area = min_table_area
        self.line_threshold = line_threshold
    
    def detect_tables(self, image: Image.Image) -> TableDetectionResult:
        """
        Detect tables in image using line detection
        
        Args:
            image: PIL Image
            
        Returns:
            TableDetectionResult with detected tables
        """
        # Convert PIL to OpenCV
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        tables = []
        
        # Method 1: Line-based detection (for structured tables)
        horizontal_tables = self._detect_by_lines(gray, cv_image, orientation='horizontal')
        vertical_tables = self._detect_by_lines(gray, cv_image, orientation='vertical')
        
        tables.extend(horizontal_tables)
        tables.extend(vertical_tables)
        
        # Method 2: Contour-based detection (for bordered tables)
        contour_tables = self._detect_by_contours(gray, cv_image)
        tables.extend(contour_tables)
        
        # Remove duplicates (overlapping detections)
        tables = self._remove_duplicates(tables)
        
        return TableDetectionResult(
            image=image,
            tables=tables,
            has_tables=len(tables) > 0,
            table_count=len(tables)
        )
    
    def _detect_by_lines(self, gray: np.ndarray, cv_image: np.ndarray, 
                         orientation: str) -> List[Table]:
        """Detect tables by finding horizontal/vertical lines"""
        tables = []
        
        # Create kernel for line detection
        if orientation == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
        else:  # vertical
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
        
        # Detect lines
        line_img = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Threshold
        _, binary = cv2.threshold(line_img, self.line_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > self.min_table_area:
                table = Table(
                    bbox=(x, y, x + w, y + h),
                    confidence=0.7,  # Line-based detection confidence
                    table_image=self._crop_table_image(cv_image, (x, y, x + w, y + h))
                )
                tables.append(table)
        
        return tables
    
    def _detect_by_contours(self, gray: np.ndarray, cv_image: np.ndarray) -> List[Table]:
        """Detect tables by finding rectangular contours"""
        tables = []
        
        # Threshold
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 points)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by area and aspect ratio (tables are usually wider than tall)
                if area > self.min_table_area and 0.5 < aspect_ratio < 5:
                    table = Table(
                        bbox=(x, y, x + w, y + h),
                        confidence=0.6,  # Contour-based detection confidence
                        table_image=self._crop_table_image(cv_image, (x, y, x + w, y + h))
                    )
                    tables.append(table)
        
        return tables
    
    def _crop_table_image(self, cv_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Crop table image from bounding box"""
        x1, y1, x2, y2 = bbox
        cropped = cv_image[max(0, y1):min(cv_image.shape[0], y2), 
                          max(0, x1):min(cv_image.shape[1], x2)]
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
    def _remove_duplicates(self, tables: List[Table], iou_threshold: float = 0.3) -> List[Table]:
        """Remove duplicate/overlapping table detections"""
        if len(tables) < 2:
            return tables
        
        # Sort by confidence (descending)
        tables = sorted(tables, key=lambda t: t.confidence, reverse=True)
        
        filtered = []
        for table in tables:
            # Check if overlaps with already filtered tables
            is_duplicate = False
            for filtered_table in filtered:
                iou = self._compute_iou(table.bbox, filtered_table.bbox)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(table)
        
        return filtered
    
    @staticmethod
    def _compute_iou(bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union (IoU) between two bboxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Compute intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Compute union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def visualize_detections(self, result: TableDetectionResult, 
                            color: Tuple[int, int, int] = (0, 255, 0),
                            thickness: int = 2) -> Image.Image:
        """
        Draw table detections on image
        
        Args:
            result: TableDetectionResult
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Annotated PIL Image
        """
        cv_image = cv2.cvtColor(np.array(result.image), cv2.COLOR_RGB2BGR)
        
        for i, table in enumerate(result.tables):
            x1, y1, x2, y2 = table.bbox
            
            # Draw rectangle
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence and index
            label = f"Table {i+1} ({table.confidence:.1%})"
            cv2.putText(cv_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


class TableOCRExtractor:
    """Extract text and structure from detected tables"""
    
    def __init__(self, ocr_system):
        """
        Initialize table OCR extractor
        
        Args:
            ocr_system: OCR system instance (for recognizing text in tables)
        """
        self.ocr_system = ocr_system
    
    def extract_text_from_tables(self, result: TableDetectionResult, 
                                engines: List[str] = None) -> List[Dict]:
        """
        Extract text from detected tables
        
        Args:
            result: TableDetectionResult
            engines: OCR engines to use
            
        Returns:
            List of table data with extracted text
        """
        extracted_tables = []
        
        for i, table in enumerate(result.tables):
            if table.table_image is None:
                continue
            
            # OCR on table image
            ocr_result = self.ocr_system.recognize(
                table.table_image,
                engines=engines or ['vietocr', 'paddleocr'],
                voting_method='best'
            )
            
            table.text_content = ocr_result.text
            
            extracted_tables.append({
                'index': i,
                'bbox': table.bbox,
                'confidence': table.confidence,
                'text': ocr_result.text,
                'image': table.table_image
            })
        
        return extracted_tables


def detect_tables_in_pdf_page(page_image: Image.Image, 
                              detector: TableDetector) -> TableDetectionResult:
    """
    Detect tables in a single PDF page
    
    Args:
        page_image: PIL Image of PDF page
        detector: TableDetector instance
        
    Returns:
        TableDetectionResult
    """
    return detector.detect_tables(page_image)


def extract_all_tables(pdf_pages: List[Image.Image]) -> Dict[int, List[Table]]:
    """
    Extract tables from all PDF pages
    
    Args:
        pdf_pages: List of PIL Images (one per page)
        
    Returns:
        Dictionary mapping page number to list of detected tables
    """
    detector = TableDetector()
    all_tables = {}
    
    for page_idx, page_image in enumerate(pdf_pages):
        result = detector.detect_tables(page_image)
        if result.has_tables:
            all_tables[page_idx] = result.tables
    
    return all_tables
