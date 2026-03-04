"""
Table positioning utility - Insert tables at correct positions in extracted text
Hỗ trợ chèn bảng vào đúng vị trí trong nội dung trích xuất
"""

import re
from typing import List, Dict, Tuple


class TablePositioner:
    """Position and insert tables at their correct locations in extracted text"""
    
    @staticmethod
    def find_table_positions(text: str, tables: List[Dict]) -> List[Tuple[int, int, Dict]]:
        """
        Find the start position of each table in the original text
        
        Args:
            text: Original extracted text
            tables: List of table data from TextBasedTableExtractor
            
        Returns:
            List of (start_pos, end_pos, table_dict)
        """
        positions = []
        search_start = 0
        
        for table in tables:
            table_text = table['text']
            table_lines = table_text.split('\n')
            
            if not table_lines:
                continue
            
            # Find first line of table (usually header)
            first_line = table_lines[0].strip()
            
            # Search for this line in remaining text
            pos = text.find(first_line, search_start)
            
            if pos != -1:
                # Try to find the actual end of the table by looking for all lines
                # Start from the first line position and find where the table actually ends
                current_pos = pos
                
                # For each line in the table, find it in the text
                last_line_pos = pos + len(first_line)
                
                for i in range(1, len(table_lines)):
                    line = table_lines[i].strip()
                    if line:  # Skip empty lines
                        # Search for this line starting from after the previous line
                        line_pos = text.find(line, last_line_pos)
                        if line_pos != -1:
                            last_line_pos = line_pos + len(line)
                        else:
                            # If we can't find this line, the table might have ended
                            break
                
                # The end position is after the last line we found
                end_pos = last_line_pos
                
                # Make sure we skip past whitespace/newlines to not duplicate content
                while end_pos < len(text) and text[end_pos] in '\n\r\t ':
                    end_pos += 1
                
                positions.append((pos, end_pos, table))
                search_start = end_pos
        
        return sorted(positions, key=lambda x: x[0])
    
    @staticmethod
    def insert_styled_tables(text: str, tables: List[Dict], formatter) -> str:
        """
        Insert HTML-styled tables at their positions in text
        
        Args:
            text: Original text
            tables: List of table data
            formatter: TableFormatter instance
            
        Returns:
            Text with inline styled tables
        """
        if not tables:
            return text
        
        # Find positions
        positions = TablePositioner.find_table_positions(text, tables)
        
        if not positions:
            return text
        
        # Build new text by inserting styled tables
        result = []
        last_pos = 0
        
        for start_pos, end_pos, table in positions:
            # Add text before table
            result.append(text[last_pos:start_pos])
            
            # Add styled table
            styled_table = TablePositioner._create_styled_table_html(table, formatter)
            result.append(styled_table)
            
            last_pos = end_pos
        
        # Add remaining text
        result.append(text[last_pos:])
        
        return ''.join(result)
    
    @staticmethod
    def _create_styled_table_html(table: Dict, formatter) -> str:
        """
        Create an HTML-styled table for embedding in text
        
        Args:
            table: Table data dictionary
            formatter: TableFormatter instance
            
        Returns:
            HTML-formatted table string
        """
        table_text = table['text']
        
        # Parse table structure
        lines = table_text.strip().split('\n')
        if not lines:
            return ""
        
        # Detect separator (tab or multiple spaces)
        first_line = lines[0]
        
        # Check if tab-separated
        if '\t' in first_line:
            separator = '\t'
        else:
            # Multiple spaces
            separator = '  '
        
        # Parse rows
        rows = []
        for line in lines:
            if separator == '\t':
                cells = [cell.strip() for cell in line.split('\t')]
            else:
                # Split by 2+ spaces
                cells = [cell.strip() for cell in re.split(r'  +', line)]
            
            if cells and any(cell for cell in cells):  # Ignore empty rows
                rows.append(cells)
        
        if not rows:
            return ""
        
        # Determine number of columns
        num_cols = max(len(row) for row in rows) if rows else 0
        
        # Pad rows to same length
        for row in rows:
            while len(row) < num_cols:
                row.append("")
        
        # Create HTML table with styling
        html_lines = [
            '<div style="margin: 15px 0; border: 1px solid #ddd; border-radius: 5px;">',
            '<table style="width: 100%; border-collapse: collapse; font-family: monospace; font-size: 12px;">',
        ]
        
        for i, row in enumerate(rows):
            is_header = i == 0  # First row is header
            
            html_lines.append('  <tr style="border-bottom: 1px solid #ddd;">')
            
            for cell in row:
                cell_tag = 'th' if is_header else 'td'
                bg_color = '#f0f0f0' if is_header else 'white'
                font_weight = 'bold' if is_header else 'normal'
                
                cell_html = (
                    f'    <{cell_tag} style="'
                    f'padding: 10px; '
                    f'background-color: {bg_color}; '
                    f'border-right: 1px solid #ddd; '
                    f'text-align: left; '
                    f'font-weight: {font_weight};">'
                    f'{cell}'
                    f'</{cell_tag}>'
                )
                html_lines.append(cell_html)
            
            html_lines.append('  </tr>')
        
        html_lines.extend([
            '</table>',
            '</div>',
        ])
        
        return '\n'.join(html_lines)
    
    @staticmethod
    def get_table_blocks(text: str, tables: List[Dict]) -> List[Dict]:
        """
        Get table blocks with position info for display purposes
        
        Args:
            text: Original text
            tables: List of extracted tables
            
        Returns:
            List of dicts with table info and position
        """
        positions = TablePositioner.find_table_positions(text, tables)
        
        blocks = []
        for start_pos, end_pos, table in positions:
            blocks.append({
                'start': start_pos,
                'end': end_pos,
                'text': table['text'],
                'type': 'table'
            })
        
        return blocks
    
    @staticmethod
    def split_text_by_tables(text: str, tables: List[Dict]) -> List[Dict]:
        """
        Split text into segments with tables interspersed
        Useful for display purposes
        
        Args:
            text: Original text
            tables: List of extracted tables
            
        Returns:
            List of dicts with 'type' ('text' or 'table') and 'content'
        """
        if not tables:
            return [{'type': 'text', 'content': text}]
        
        positions = TablePositioner.find_table_positions(text, tables)
        
        segments = []
        last_pos = 0
        
        for start_pos, end_pos, table in positions:
            # Add text segment before table
            if start_pos > last_pos:
                text_content = text[last_pos:start_pos].strip()
                if text_content:
                    segments.append({
                        'type': 'text',
                        'content': text_content
                    })
            
            # Add table segment
            segments.append({
                'type': 'table',
                'content': table['text'],
                'data': table
            })
            
            last_pos = end_pos
        
        # Add remaining text
        if last_pos < len(text):
            text_content = text[last_pos:].strip()
            if text_content:
                segments.append({
                    'type': 'text',
                    'content': text_content
                })
        
        return segments
