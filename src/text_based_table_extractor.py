"""
Extract tables from OCR text output (without image-based detection)
Trích xuất bảng từ output OCR text (không cần detect từ ảnh)
"""

import re
from typing import List, Dict, Tuple, Optional


class TextBasedTableExtractor:
    """
    Extract table structures directly from OCR text output
    Useful when image-based detection is not available
    """
    
    @staticmethod
    def extract_tables_from_text(text: str, min_rows: int = 3) -> List[Dict]:
        """
        Try to find table-like structures in OCR text
        
        Args:
            text: Full OCR text output
            min_rows: Minimum rows to be considered a table
            
        Returns:
            List of detected table blocks with their text
        """
        tables = []
        
        # First, try to split by paragraph boundaries and find tables
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # Split each paragraph into lines
            lines = [line.strip() for line in para.split('\n') if line.strip()]
            
            if len(lines) >= min_rows:
                # Try to extract tables from this block
                table_blocks = TextBasedTableExtractor._extract_table_blocks_from_lines(lines)
                tables.extend(table_blocks)
        
        return tables
    
    @staticmethod
    def _extract_table_blocks_from_lines(lines: List[str]) -> List[Dict]:
        """
        Extract table blocks from a list of lines
        Handles case where table is mixed with regular text
        
        Args:
            lines: List of text lines
            
        Returns:
            List of detected table blocks
        """
        tables = []
        
        if not lines:
            return tables
        
        # Find continuous blocks that look like tables
        current_block = []
        
        for line in lines:
            # Check if line looks like table data or header
            if TextBasedTableExtractor._looks_like_table_line(line):
                current_block.append(line)
            else:
                # Non-table line encountered
                if len(current_block) >= 3:  # At least 3 lines to be a table
                    # Check if accumulated block is actually a table
                    if TextBasedTableExtractor._looks_like_table(current_block):
                        # Clean up the table block - remove duplicate/garbage rows
                        cleaned_block = TextBasedTableExtractor._clean_table_block(current_block)
                        if len(cleaned_block) >= 3:  # Still valid after cleaning
                            table_text = '\n'.join(cleaned_block)
                            tables.append({
                                'text': table_text,
                                'index': len(tables),
                                'bbox': None,
                                'image': None
                            })
                current_block = []
        
        # Don't forget last block
        if len(current_block) >= 3:
            if TextBasedTableExtractor._looks_like_table(current_block):
                # Clean up the table block - remove duplicate/garbage rows
                cleaned_block = TextBasedTableExtractor._clean_table_block(current_block)
                if len(cleaned_block) >= 3:  # Still valid after cleaning
                    table_text = '\n'.join(cleaned_block)
                    tables.append({
                        'text': table_text,
                        'index': len(tables),
                        'bbox': None,
                        'image': None
                    })
        
        return tables
    
    @staticmethod
    def _clean_table_block(block: List[str]) -> List[str]:
        """
        Remove duplicate or garbage rows from table block
        
        Detects rows that are repetitions of previous rows or obviously garbage
        
        Args:
            block: List of table lines
            
        Returns:
            Cleaned list of table lines
        """
        if len(block) < 2:
            return block
        
        cleaned = [block[0]]  # Always keep header
        
        for i in range(1, len(block)):
            current_line = block[i].strip()
            current_tokens = current_line.split()
            
            # Check if this line is a duplicate/subset of previous lines
            is_duplicate = False
            
            # Check if this line appears to be a partial repetition of a previous row
            for j in range(max(0, i-3), i):  # Check last 3 rows
                prev_line = block[j].strip()
                prev_tokens = prev_line.split()
                
                # Case 1: Exact duplicate
                if current_line == prev_line:
                    is_duplicate = True
                    break
                
                # Case 2: Current line is a subset of previous (e.g., last part of previous)
                # This handles cases like "Em 35 Hải Phòng Nhà báo" becoming "35 Hải Phòng Nhà báo"
                if current_tokens and len(current_tokens) <= len(prev_tokens):
                    # Check if all tokens from current appear in order in previous
                    # (allowing some distance between them)
                    prev_str = ' '.join(prev_tokens)
                    current_str = ' '.join(current_tokens)
                    
                    if current_str in prev_str:
                        is_duplicate = True
                        break
                    
                    # Also check if tokens appear with similar order/proximity
                    token_matches = 0
                    for token in current_tokens:
                        if token in prev_tokens:
                            token_matches += 1
                    
                    # If most tokens match previous row, likely a duplicate
                    if token_matches >= max(2, len(current_tokens) - 1):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                cleaned.append(block[i])
        
        return cleaned
    
    @staticmethod
    def _looks_like_table_line(line: str) -> bool:
        """
        Check if a single line looks like it's part of a table
        
        Table lines usually:
        - Contain 1-4 columns separated by whitespace/tabs
        - May contain numbers or names
        - Usually not starting with bullet points or dashes
        
        Args:
            line: Single text line
            
        Returns:
            True if looks like table line
        """
        line = line.strip()
        
        # Reject bullet points and list-like structures
        if line.startswith('-') or line.startswith('•') or line.startswith('*'):
            return False
        
        # Too short but non-empty is suspicious
        if len(line) < 2:
            return False
        
        # Check word count (split by ANY whitespace including tabs)
        words = line.split()
        
        # Single very long word = probably regular text
        if len(words) == 1 and len(line) > 20:
            return False
        
        # Tables with tabs can have more words when split by whitespace
        # But don't go too crazy - if we have tons of words, it's probably narrative
        # Allow up to 25 words for Vietnamese tables with longer text entries
        if len(words) > 25:
            return False
        
        # Tables with multiple words (after filtering out lists)
        if len(words) >= 1:
            # Check if it looks like table content
            # Table lines usually contain:
            # - Names (short words)
            # - Numbers
            # - Place names
            # - Occupation names
            # - Column headers
            # - Longer text entries in columns
            
            # Common table keywords (Vietnamese)
            table_keywords = {
                'tên', 'tuổi', 'địa chỉ', 'thành phố', 'quốc gia', 'ngày', 'số',
                'loại', 'giá', 'nghề', 'trạng thái', 'ghi chú', 'mã', 'lý do',
                'mô tả', 'công việc', 'dự án', 'trạng thái'
            }
            
            line_lower = line.lower()
            
            # If contains table keywords, likely a header or table row
            if any(kw in line_lower for kw in table_keywords):
                return True
            
            # Check for Vietnamese place names or common names
            place_keywords = {'hà nội', 'hồ chí minh', 'đà nẵng', 'hải phòng', 'cần thơ'}
            if any(place in line_lower for place in place_keywords):
                return True
            
            # Check for occupation keywords
            job_keywords = {'kỹ sư', 'giáo viên', 'bác sĩ', 'sinh viên', 'nhà báo', 'công nhân'}
            if any(job in line_lower for job in job_keywords):
                return True
            
            # Check for numbers (age, ID, quantity)
            has_number = any(c.isdigit() for c in line)
            if has_number:
                # If line has numbers, likely table data
                return True
            
            # Short, named-like (2-3 words, starts with capital)
            if len(words) <= 2 and line and line[0].isupper():
                return True
            
            # If we have 2-10 words and reasonable length, might be table
            if 2 <= len(words) <= 10:
                return True
        
        return False
    
    @staticmethod
    def _looks_like_table(lines: List[str]) -> bool:
        """
        Check if a block of lines looks like a table
        
        Characteristics:
        - Multiple lines with similar structure
        - Lines have repeated patterns
        - Could be column-separated or row-based
        
        Args:
            lines: List of text lines
            
        Returns:
            True if looks like a table
        """
        if len(lines) < 2:
            return False
        
        text_block = '\n'.join(lines)
        
        # Check for Vietnamese table header keywords (strong indicator)
        header_keywords = {
            'tên', 'tuổi', 'địa chỉ', 'thành phố', 'quốc gia', 'ngày', 'số', 
            'loại', 'giá', 'nghề', 'nghề nghiệp', 'mã', 'tổng', 'chi tiết',
            'nội dung', 'ghi chú', 'trạng thái', 'lý do'
        }
        
        header_matches = sum(1 for kw in header_keywords if kw in text_block.lower())
        
        # If multiple header keywords found, very likely a table
        if header_matches >= 2:
            return True
        
        # Heuristic 2: Check for repeating patterns of similar-length lines
        # Tables often have consistent row structure
        line_lengths = [len(line) for line in lines]
        
        if len(line_lengths) > 1:
            avg_length = sum(line_lengths) / len(line_lengths)
            
            # If lines have similar lengths (within 60%), might be table
            similar_length_lines = sum(1 for l in line_lengths if 0.4 * avg_length <= l <= 1.6 * avg_length)
            if similar_length_lines >= len(lines) * 0.5:  # At least 50% similar
                # Additional check: lines should be relatively short (< 200 chars)
                if avg_length < 200:
                    return True
        
        # Heuristic 3: Check if lines might be separable into columns
        # Look for multiple items per line
        column_like_items = 0
        for line in lines:
            items = line.split()
            if 2 <= len(items) <= 10:  # Reasonable number of columns
                column_like_items += 1
        
        if column_like_items >= len(lines) * 0.6:  # At least 60% have multiple items
            return True
        
        # Heuristic 4: Check for numerical data (suggest table with data rows)
        numbers = re.findall(r'\d+', text_block)
        if numbers:
            number_ratio = len(numbers) / max(1, len(text_block.split()))
            
            # Good amount of numbers (15-80%) suggests tabular data
            if 0.15 <= number_ratio <= 0.8 and len(lines) >= 3:
                return True
        
        return False

