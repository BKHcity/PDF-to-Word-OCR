"""
Table Formatter - Format detected table data as structured text/HTML
Định dạng bảng được phát hiện thành text hoặc HTML có cấu trúc
"""

from typing import List, Dict, Tuple, Union, Optional
import re
import base64
import numpy as np
from io import BytesIO
from PIL import Image


class TableFormatter:
    """Format table detection results as readable text with preserved structure"""
    
    @staticmethod
    def format_table_text(table_text: str) -> str:
        """
        Format raw OCR text from table into structured format
        
        Args:
            table_text: Raw text from table OCR
            
        Returns:
            Formatted table text
        """
        if not table_text.strip():
            return "[Empty table]"
        
        # Try to parse as CSV-like data
        lines = table_text.strip().split('\n')
        if not lines:
            return table_text
        
        # Filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        # Try to detect columns by multiple spaces
        formatted_lines = []
        for line in lines:
            # Replace multiple spaces with tab for better alignment
            formatted_line = re.sub(r'\s{2,}', '\t', line)
            formatted_lines.append(formatted_line)
        
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def format_table_as_grid(data: List[List[str]]) -> str:
        """
        Format table data as ASCII grid
        
        Args:
            data: 2D list of table cells
            
        Returns:
            ASCII formatted table
        """
        if not data or not data[0]:
            return "[Empty table]"
        
        # Calculate column widths
        col_widths = []
        for col_idx in range(len(data[0])):
            max_width = max(len(str(row[col_idx] if col_idx < len(row) else '')) for row in data)
            col_widths.append(max_width + 2)
        
        # Build header line
        lines = []
        lines.append('┌' + '┬'.join('─' * width for width in col_widths) + '┐')
        
        # Add rows
        for row_idx, row in enumerate(data):
            # Add row data
            cells = []
            for col_idx, width in enumerate(col_widths):
                cell_text = str(row[col_idx] if col_idx < len(row) else '')
                cells.append(cell_text.center(width))
            
            lines.append('│' + '│'.join(cells) + '│')
            
            # Add separator after each row (including header)
            if row_idx == 0 or row_idx == len(data) - 1:
                lines.append('├' + '┼'.join('─' * width for width in col_widths) + '┤')
            else:
                lines.append('├' + '┼'.join('─' * width for width in col_widths) + '┤')
        
        # Bottom border
        lines[-1] = '└' + '┴'.join('─' * width for width in col_widths) + '┘'
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_table_as_markdown(data: List[List[str]]) -> str:
        """
        Format table data as Markdown
        
        Args:
            data: 2D list of table cells
            
        Returns:
            Markdown formatted table
        """
        if not data or not data[0]:
            return "[Empty table]"
        
        lines = []
        
        # Header row
        header = data[0]
        lines.append('| ' + ' | '.join(str(h) for h in header) + ' |')
        
        # Separator
        lines.append('|' + '|'.join(['---' for _ in header]) + '|')
        
        # Data rows
        for row in data[1:]:
            lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_table_as_box_drawing(data: List[List[str]]) -> str:
        """
        Format table using Unicode box drawing characters (like in the image)
        Creates professional looking tables with clear borders
        
        Args:
            data: 2D list of table cells
            
        Returns:
            Table formatted with box drawing characters
        """
        if not data or not data[0]:
            return "[Empty table]"
        
        # Calculate column widths (add padding)
        col_widths = []
        for col_idx in range(len(data[0])):
            max_width = max(len(str(row[col_idx] if col_idx < len(row) else '')) for row in data)
            col_widths.append(max_width + 2)  # Add padding
        
        lines = []
        
        # Top border
        top_line = '┌' + '┬'.join('─' * width for width in col_widths) + '┐'
        lines.append(top_line)
        
        # Add header row
        header_cells = []
        for col_idx, width in enumerate(col_widths):
            cell_text = str(data[0][col_idx])
            # Center align header
            centered = cell_text.center(width)
            header_cells.append(centered)
        lines.append('│' + '│'.join(header_cells) + '│')
        
        # Separator after header
        separator = '├' + '┼'.join('─' * width for width in col_widths) + '┤'
        lines.append(separator)
        
        # Add data rows
        for row_idx, row in enumerate(data[1:]):
            cells = []
            for col_idx, width in enumerate(col_widths):
                cell_text = str(row[col_idx] if col_idx < len(row) else '')
                # Left align data cells with padding
                padded = ' ' + cell_text.ljust(width - 1)
                cells.append(padded)
            lines.append('│' + '│'.join(cells) + '│')
        
        # Bottom border
        bottom_line = '└' + '┴'.join('─' * width for width in col_widths) + '┘'
        lines.append(bottom_line)
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_table_as_simple(data: List[List[str]]) -> str:
        """
        Format table data as simple aligned columns
        
        Args:
            data: 2D list of table cells
            
        Returns:
            Simple aligned table
        """
        if not data or not data[0]:
            return "[Empty table]"
        
        # Calculate column widths
        col_widths = []
        for col_idx in range(len(data[0])):
            max_width = max(len(str(row[col_idx] if col_idx < len(row) else '')) for row in data)
            col_widths.append(max_width)
        
        lines = []
        for row in data:
            cells = []
            for col_idx, width in enumerate(col_widths):
                cell_text = str(row[col_idx] if col_idx < len(row) else '')
                cells.append(cell_text.ljust(width))
            
            lines.append('  '.join(cells))
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_table_as_html(data: List[List[str]], table_title: str = "Table", 
                            with_borders: bool = True, zebra_striping: bool = True) -> str:
        """
        Format table data as HTML with styling
        
        Args:
            data: 2D list of table cells
            table_title: Title for the table
            with_borders: Add borders around cells
            zebra_striping: Alternate row colors
            
        Returns:
            HTML formatted table
        """
        if not data or not data[0]:
            return "<p>[Empty table]</p>"
        
        # Build HTML
        html = '<table style="'
        if with_borders:
            html += 'border-collapse: collapse; border: 2px solid #333; width: 100%; margin: 10px 0;'
        else:
            html += 'width: 100%; margin: 10px 0;'
        html += '">\n'
        
        # Header row
        header_row = data[0]
        html += '  <thead>\n    <tr style="background-color: #4CAF50; color: white;">\n'
        for cell in header_row:
            cell_html = str(cell).replace('<', '&lt;').replace('>', '&gt;')
            border_style = 'border: 1px solid #ddd; ' if with_borders else ''
            html += f'      <th style="{border_style}padding: 12px; text-align: left; font-weight: bold;">{cell_html}</th>\n'
        html += '    </tr>\n  </thead>\n'
        
        # Data rows
        html += '  <tbody>\n'
        for row_idx, row in enumerate(data[1:], 1):
            bg_color = '#f9f9f9' if (zebra_striping and row_idx % 2 == 0) else 'white'
            html += f'    <tr style="background-color: {bg_color};">\n'
            
            for cell in row:
                cell_html = str(cell).replace('<', '&lt;').replace('>', '&gt;')
                border_style = 'border: 1px solid #ddd; ' if with_borders else ''
                html += f'      <td style="{border_style}padding: 10px;">{cell_html}</td>\n'
            
            html += '    </tr>\n'
        
        html += '  </tbody>\n</table>'
        
        return html
    
    @staticmethod
    def format_table_image_with_grid(table_image: Image.Image, data: List[List[str]]) -> Image.Image:
        """
        Draw table grid on image with detected structure
        
        Args:
            table_image: PIL Image of detected table
            data: Structured table data
            
        Returns:
            Image with grid overlay
        """
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            return table_image
        
        img_copy = table_image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Draw borders around table
        width, height = img_copy.size
        line_width = 3
        draw.rectangle(
            [(line_width//2, line_width//2), (width - line_width//2, height - line_width//2)],
            outline='red',
            width=line_width
        )
        
        return img_copy
    
    @staticmethod
    def extract_table_structure(table_text: str) -> Tuple[List[List[str]], str]:
        """
        Try to extract table structure from OCR text
        
        Args:
            table_text: Raw OCR text from table
            
        Returns:
            Tuple of (structured data, format type)
        """
        if not table_text.strip():
            return [], 'empty'
        
        lines = table_text.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return [], 'empty'
        
        # Try Method 1: One cell per line (vertical table format) FIRST
        # This is more likely for OCR output from PDF tables
        result = TableFormatter._extract_vertical_table(lines)
        if result is not None and len(result) > 1:
            return result, 'structured'
        
        # Try Method 2: Split by multiple spaces/tabs (inline table format)
        data = []
        for line in lines:
            # Split by multiple spaces or tabs
            parts = re.split(r'\s{2,}|\t', line)
            parts = [p.strip() for p in parts if p.strip()]
            if parts:
                data.append(parts)
        
        if data and len(data) > 1:
            col_counts = [len(row) for row in data]
            # If most rows have same column count, it's likely a table
            if len(set(col_counts)) <= 2 and col_counts[0] > 1:  # Allow some variance, but need multiple columns
                return data, 'structured'
        
        if data:
            return data, 'list'
        
        return [[table_text]], 'text'
    
    @staticmethod
    def _extract_vertical_table(lines: List[str]) -> Optional[List[List[str]]]:
        """
        Extract table when each cell is on a separate line
        
        This method tries to detect the number of columns by analyzing patterns.
        
        Args:
            lines: List of text lines
            
        Returns:
            2D table structure or None if can't detect
        """
        if len(lines) < 2:
            return None
        
        # Try to find the number of columns by looking for repeating patterns
        # Count how many "header-like" items are at the start
        num_cols = TableFormatter._detect_num_columns(lines)
        
        if num_cols < 2 or num_cols > 10:
            return None
        
        # Try extraction with detected columns
        result = TableFormatter._try_vertical_extraction(lines, num_cols)
        return result
    
    @staticmethod
    def _detect_num_columns(lines: List[str]) -> int:
        """
        Detect the number of columns in a vertical table
        
        Looks for the first continuous "header" section and counts it.
        
        Args:
            lines: List of text lines
            
        Returns:
            Estimated number of columns (2-10)
        """
        if not lines:
            return 4
        
        # Strategy: Find the longest continuous sequence of headers at the start
        # Headers are usually grouped together, then data follows
        
        header_count = 0
        for i, line in enumerate(lines):
            if TableFormatter._is_proper_header(line):
                header_count += 1
            else:
                # Check if next few items break the header pattern
                # If we've found at least 2 headers and now hit non-header, that's likely the column count
                if header_count >= 2:
                    # Verify this could be data (not another type of content)
                    if i + 1 < len(lines):
                        next_item = lines[i + 1]
                        # If next item is also non-header, this confirms end of header section
                        if not TableFormatter._is_proper_header(next_item):
                            return header_count
                    # Even if next is header-like, if we have 2+ headers, return
                    if header_count >= 2:
                        return header_count
                else:
                    # Haven't found enough headers yet, keep looking
                    # But if we hit non-header early, reset
                    if i > 5:  # Allow up to 10 items to be "headers"
                        break
        
        if header_count >= 2:
            return header_count
        
        # If no clear header section found, try common patterns
        for num_cols in [4, 3, 5, 6, 2, 7]:
            if len(lines) % num_cols == 0 and num_cols > 1:
                first_group = lines[:num_cols]
                # Check if all first group look like headers
                header_like_count = sum(1 for item in first_group if TableFormatter._is_proper_header(item))
                if header_like_count >= num_cols * 0.8:  # At least 80% look like headers
                    return num_cols
        
        # Default to 4
        return 4
    
    @staticmethod
    def _is_proper_header(text: str) -> bool:
        """
        Check if text is a proper table header (strict version)
        
        Args:
            text: Text to check
            
        Returns:
            True if looks like a header
        """
        text = text.strip()
        
        # Headers should be 2-50 chars
        if len(text) < 2 or len(text) > 50:
            return False
        
        # Must have mostly letters
        letter_ratio = sum(c.isalpha() or c in 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ' for c in text) / max(1, len(text))
        if letter_ratio < 0.5:
            return False
        
        # Digit ratio should be low
        digit_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
        if digit_ratio > 0.3:
            return False
        
        word_count = len(text.split())
        
        if word_count >= 2:
            # Multi-word items are usually headers
            return True
        elif word_count == 1:
            # Single word: use heuristic
            single_word_lower = text.lower()
            
            # Common headers: Vietnamese column names
            common_headers = {'tên', 'tuổi', 'ngày', 'số', 'địa chỉ', 'địa điểm', 'thành phố', 'quốc gia', 'loại', 'ngành', 'xếp hạng', 'giá', 'tổng', 'mục đích', 'lương', 'chuyên môn', 'ngoài'}
            if single_word_lower in common_headers:
                return True
            
            # Single word under 4 chars that's NOT a known header = probably data (name)
            # Exception: check if it's in the list above
            if len(text) <= 3:
                return False
            
            # Longer single words could still be headers or data
            # Use another heuristic: Vietnamese column headers often end with specific patterns
            # or contain specific morphemes
            # For now: long single words (>5 chars) are probably headers
            if len(text) > 5:
                return True
            
            # 4-5 char single words: ambiguous, assume NOT header to be conservative
            return False
        
        return False
    
    @staticmethod
    def _try_vertical_extraction(lines: List[str], num_cols: int) -> Optional[List[List[str]]]:
        """
        Try to extract table with specific number of columns
        
        Args:
            lines: List of text lines
            num_cols: Expected number of columns
            
        Returns:
            2D table structure or None if doesn't fit
        """
        if len(lines) % num_cols != 0:
            # Number of lines doesn't divide evenly
            return None
        
        rows = []
        for i in range(0, len(lines), num_cols):
            row = lines[i:i + num_cols]
            if len(row) == num_cols:
                rows.append(row)
        
        if len(rows) < 2:  # Need at least 2 rows (header + 1 data row)
            return None
        
        # Validate: check if first row looks like a real header row
        header = rows[0]
        data_rows = rows[1:]
        
        # Good header characteristics:
        # 1. All cells are short (< 50 chars typically)
        # 2. Cells don't look like data (e.g., not dates, not email addresses)
        # 3. Cells don't have lots of numbers
        # 4. Cells look like labels/names
        
        # Check if this could be a header
        is_likely_header = TableFormatter._is_likely_header_row(header)
        
        if not is_likely_header:
            return None
        
        # Additional validation: check data consistency
        if len(rows) >= 3:  # If we have at least 3 rows, check data consistency
            # Most data rows should have similar characteristics
            valid_data_rows = 0
            for row in data_rows[:5]:  # Check first 5 data rows
                if TableFormatter._is_likely_data_row(row, len(header)):
                    valid_data_rows += 1
            
            # If less than half of data rows look valid, reject
            if valid_data_rows < len(data_rows[:5]) / 2:
                return None
        
        return rows
    
    @staticmethod
    def _is_likely_header_row(row: List[str]) -> bool:
        """
        Check if a row looks like a table header
        
        Args:
            row: List of cell values
            
        Returns:
            True if row looks like a header
        """
        # All cells should be reasonably short
        if any(len(cell) > 50 for cell in row):
            return False
        
        # Headers usually have descriptive names (not pure numbers or dates)
        non_data_cells = 0
        for cell in row:
            # Check if cell looks like a label/name (not data)
            is_label = True
            
            # If it's mostly digits, probably not a label
            digit_ratio = sum(c.isdigit() for c in cell) / max(1, len(cell))
            if digit_ratio > 0.5:
                is_label = False
            
            # Vietnamese labels usually contain letters
            has_letters = any(c.isalpha() or c in 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ' for c in cell.lower())
            if not has_letters:
                is_label = False
            
            if is_label:
                non_data_cells += 1
        
        # At least 70% of header cells should look like labels
        if non_data_cells >= len(row) * 0.7:
            return True
        
        return False
    
    @staticmethod
    def _is_likely_data_row(row: List[str], num_cols: int) -> bool:
        """
        Check if a row looks like data (not a header)
        
        Args:
            row: List of cell values
            num_cols: Number of columns
            
        Returns:
            True if row looks like data
        """
        if len(row) != num_cols:
            return False
        
        # Data rows can have mixed content (numbers, text, dates)
        # Just check that it's not all empty or unreasonably long
        if all(len(cell) == 0 for cell in row):
            return False
        
        if any(len(cell) > 200 for cell in row):
            return False
        
        return True
    
    @staticmethod
    def format_table_with_borders(table_text: str, format_type: str = 'simple') -> str:
        """
        Format table with borders/structure
        
        Args:
            table_text: Raw table text from OCR
            format_type: 'simple', 'markdown', 'grid'
            
        Returns:
            Formatted table text
        """
        data, detected_type = TableFormatter.extract_table_structure(table_text)
        
        if detected_type == 'empty':
            return "[Empty table]"
        elif detected_type == 'text':
            return table_text
        
        # Use detected format or specified format
        if format_type == 'markdown':
            return TableFormatter.format_table_as_markdown(data)
        elif format_type == 'grid':
            return TableFormatter.format_table_as_grid(data)
        else:  # simple
            return TableFormatter.format_table_as_simple(data)


def preserve_table_structure(text: str, tables: List[Dict]) -> str:
    """
    Preserve table structure in main text
    
    Args:
        text: Main extracted text
        tables: List of detected tables with their data
        
    Returns:
        Text with preserved table structure
    """
    if not tables:
        return text
    
    formatter = TableFormatter()
    
    # Add table section
    result = text
    result += "\n\n" + "="*60 + "\n"
    result += "📊 EXTRACTED TABLES\n"
    result += "="*60 + "\n"
    
    for page_idx, page_tables in enumerate(tables):
        result += f"\n--- PAGE {page_idx + 1} ---\n"
        for table_data in page_tables:
            result += f"\n📋 Table {table_data['index'] + 1}:\n"
            result += "-" * 40 + "\n"
            
            # Format table with structure
            formatted = formatter.format_table_with_borders(
                table_data['text'],
                format_type='simple'
            )
            result += formatted
            result += "\n"
    
    return result


if __name__ == "__main__":
    # Test
    test_table = """Tên         Tuổi    Thành phố      Nghề nghiệp
An          25      Hà Nội         Kỹ sư
Bình        30      Hồ Chí Minh    Giáo viên
Chi         28      Đà Nẵng        Bác sĩ"""
    
    formatter = TableFormatter()
    print("Simple format:")
    print(formatter.format_table_with_borders(test_table, 'simple'))
    print("\nMarkdown format:")
    print(formatter.format_table_with_borders(test_table, 'markdown'))
