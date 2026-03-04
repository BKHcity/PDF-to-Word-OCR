#!/usr/bin/env python3
"""
Advanced Vietnamese OCR postprocessing:
- Character-level correction (OCR typo fix)
- Word-level merging (space insertion correction)
- Diacritic restoration (dictionary + context)
"""
import re
from typing import List, Dict, Tuple

# Common OCR character mistakes (shape-similar confusion)
OCR_CHAR_MISTAKES = {
    '0': 'O',  # number zero → letter O
    '1': 'l',  # number one → letter l
    'l': 'I',  # lowercase L → uppercase i
    'rn': 'm',  # rn → m
    'cl': 'd',  # cl → d
}

# Common word boundary errors (space missing/extra space)
COMMON_MERGES = [
    (r'nhan\s*vien', 'nhân viên'),
    (r'cong\s*viec', 'công việc'),
    (r'he\s*thong', 'hệ thống'),
    (r'may\s*tinh', 'máy tính'),
    (r'mang\s*lan', 'mạng LAN'),
    (r'thanh\s*pho', 'thành phố'),
    (r'hang\s*thang', 'hàng tháng'),
    # Curated phrase fixes for common OCR mis-transcriptions
    (r'noi\s*lam', 'nơi làm việc'),
    (r'noilam', 'nơi làm việc'),
    (r'noilam\s*vie[eê]?', 'nơi làm việc'),
]

# Common undiacritized words & replacements (curated from frequency)
COMMON_UNDIACRITIZED = {
    'hang': 'hàng',
    'dung': 'dùng',
    'trang': 'trang',
    'long': 'lòng',
    'dong': 'đông',
    'cai': 'cái',
    'day': 'dây',
    'dau': 'dấu',
    'tim': 'tìm',
    'y': 'ý',
    'chi': 'chí',
    'mon': 'môn',
    'ban': 'bản',
    'tay': 'tay',
    'chay': 'chạy',
    # Additional curated single-word corrections seen in OCR output
    'noilam': 'nơi làm việc',
    'he': 'hệ',
    'hé': 'hệ',
    'thir': 'thứ',
    'bfe': 'bậc',
    'gdm': 'gồm',
}

class VietnamesePostprocessor:
    def __init__(self, diacritic_map: Dict[str, List[str]] = None):
        """Initialize with optional diacritic dictionary."""
        self.diacritic_map = diacritic_map or {}
        self.diacritic_map.update(COMMON_UNDIACRITIZED)
    
    def fix_ocr_typos(self, text: str) -> str:
        """Fix common character confusions from OCR."""
        for mistake, fix in OCR_CHAR_MISTAKES.items():
            text = text.replace(mistake, fix)
        return text
    
    def merge_broken_words(self, text: str) -> str:
        """Fix space-broken words like 'nhan vien' → 'nhân viên'."""
        for pattern, replacement in COMMON_MERGES:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def restore_diacritics_context(self, text: str) -> str:
        """Restore diacritics using dictionary & simple context."""
        words = text.split()
        restored = []
        
        for i, word in enumerate(words):
            # Check for exact match in diacritic map
            lower_word = word.lower()
            if lower_word in self.diacritic_map:
                # Get diacritized form & preserve original case
                dia_form = self.diacritic_map[lower_word][0] if isinstance(self.diacritic_map[lower_word], list) else self.diacritic_map[lower_word]
                if word[0].isupper() and dia_form:
                    restored.append(dia_form[0].upper() + dia_form[1:])
                else:
                    restored.append(dia_form)
            else:
                restored.append(word)
        
        return ' '.join(restored)
    
    def postprocess(self, text: str) -> str:
        """Apply full postprocessing pipeline."""
        if not text:
            return text
        
        # Step 1: Fix OCR typos
        text = self.fix_ocr_typos(text)
        
        # Step 2: Merge broken words
        text = self.merge_broken_words(text)
        
        # Step 3: Restore diacritics
        text = self.restore_diacritics_context(text)
        
        # Step 4: Basic cleanup (extra spaces)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


if __name__ == '__main__':
    pp = VietnamesePostprocessor()
    test = "nhan vien cong viec he thong may tinh"
    print(f"Before: {test}")
    print(f"After:  {pp.postprocess(test)}")
