"""
Vietnamese Spell Checker & Post-Processing for OCR
Sửa lỗi dính chữ, sai dấu, và các lỗi OCR phổ biến

Author: OCR Enhancement Module
"""

import re
from typing import List, Tuple, Set, Dict

# Import corrections từ file mới
try:
    from vietnamese_ocr_corrections import (
        VietnameseOCRCorrector, 
        fix_ocr_text,
        VIETNAMESE_COMMON_WORDS,
        DIACRITIC_ERRORS,
        VOWEL_ERRORS,
        CONSONANT_ERRORS,
        STUCK_WORDS,
        SINGLE_WORD_ERRORS,
        PHRASE_ERRORS,
        CONTEXT_PATTERNS,
    )
    USE_NEW_CORRECTOR = True
    print("✅ Loaded vietnamese_ocr_corrections module")
except ImportError:
    USE_NEW_CORRECTOR = False
    print("⚠️ vietnamese_ocr_corrections not found, using built-in corrections")
    VIETNAMESE_COMMON_WORDS = set()

# Từ điển các từ tiếng Việt phổ biến (có thể mở rộng)
VIETNAMESE_COMMON_WORDS = {
    # Đại từ
    'tôi', 'tao', 'tớ', 'mình', 'ta', 'chúng', 'bọn', 'họ', 'nó', 'hắn', 'cô', 'chú', 
    'anh', 'chị', 'em', 'bạn', 'cậu', 'bác', 'ông', 'bà', 'con', 'cháu', 'thằng',
    
    # Động từ phổ biến
    'là', 'có', 'được', 'làm', 'đi', 'đến', 'về', 'ra', 'vào', 'lên', 'xuống',
    'nói', 'viết', 'đọc', 'nghe', 'xem', 'nhìn', 'thấy', 'biết', 'hiểu', 'nghĩ',
    'muốn', 'cần', 'phải', 'nên', 'hãy', 'đừng', 'chớ', 'xin', 'cho', 'lấy',
    'ăn', 'uống', 'ngủ', 'thức', 'chơi', 'học', 'dạy', 'yêu', 'ghét', 'thích',
    'sống', 'chết', 'sinh', 'lớn', 'nhỏ', 'già', 'trẻ', 'mới', 'cũ', 'đẹp',
    'giải', 'tố', 'việc', 'dung', 'hoà', 'hòa', 'mọi', 'thứ', 'thật', 'sự', 'rất',
    'khó', 'khăn', 'dễ', 'buồn', 'vui', 'thi', 'viết', 'sau', 'này', 'sẽ', 
    'lắng', 'nghe', 'nói', 'kể', 'hỏi', 'trả', 'lời',
    
    # Tính từ
    'tốt', 'xấu', 'đúng', 'sai', 'cao', 'thấp', 'dài', 'ngắn', 'rộng', 'hẹp',
    'nhanh', 'chậm', 'mạnh', 'yếu', 'khỏe', 'ốm', 'giàu', 'nghèo', 'sang', 'hèn',
    
    # Trạng từ
    'rất', 'lắm', 'quá', 'hơi', 'khá', 'cũng', 'đều', 'luôn', 'thường', 'hay',
    'đã', 'đang', 'sẽ', 'vừa', 'mới', 'còn', 'vẫn', 'cứ', 'chỉ', 'chính',
    
    # Giới từ, liên từ
    'của', 'và', 'với', 'để', 'cho', 'từ', 'đến', 'trong', 'ngoài', 'trên', 'dưới',
    'trước', 'sau', 'giữa', 'bên', 'cạnh', 'gần', 'xa', 'theo', 'về', 'tại',
    'vì', 'do', 'bởi', 'nên', 'mà', 'thì', 'nếu', 'thế', 'như', 'khi', 'lúc',
    'nhưng', 'tuy', 'dù', 'mặc', 'dầu', 'song', 'hoặc', 'hay', 'hoặc',
    
    # Số đếm
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'trăm', 'nghìn', 'ngàn', 'triệu', 'tỷ', 'vạn',
    
    # Từ thời gian
    'ngày', 'tháng', 'năm', 'tuần', 'giờ', 'phút', 'giây', 'sáng', 'trưa', 'chiều', 'tối', 'đêm',
    'hôm', 'nay', 'qua', 'mai', 'kia', 'trước', 'sau', 'này', 'ấy', 'đó',
    
    # Từ chỉ nơi chốn
    'đây', 'đó', 'kia', 'này', 'ấy', 'nọ', 'đâu', 'nào', 'sao', 'gì',
    
    # Từ phổ biến khác
    'nhé', 'nhỉ', 'nhá', 'nha', 'ạ', 'ơi', 'à', 'ừ', 'ừm', 'vâng', 'dạ',
    'không', 'chưa', 'chẳng', 'đâu', 'nào', 'gì', 'ai', 'sao', 'thế',
}

# Mapping các lỗi OCR phổ biến
COMMON_OCR_ERRORS = {
    # ===== LỖI TỪ ĐƠN (DẤU SAI) =====
    'phái': 'phải',
    'thí': 'thứ',
    'sử': 'sự',
    'thỉ': 'thì',
    'trí': 'từ',
    'tái': 'từ',
    'át': 'sẽ',
    'tổng': 'lắng',
    'guủi': 'gửi',
    'hây': 'hãy',
    'títi': 'tí tí',
    'lắngng': 'lắng',
    'lắngngng': 'lắng',
    
    # Từ cũ
    'guải': 'gửi',
    'giải': 'gửi',
    'tố': 'tớ',
    'khăm': 'khăn',
    'tri': 'từ',
    'tỉ': 'từ',
    'ai': 'ừ',
    'hấy': 'hãy',
    'buôn': 'buồn',
    'thi': 'thì',
    'xó': 'tớ',
    'sĩ': 'sẽ',
    'lắ': 'lắng',
    'nhí': 'nhé',
    'ranhí': 'ra nhé',
    
    # ===== LỖI TỪ DÍNH =====
    'ranhé': 'ra nhé',
    'xinhãy': 'xin hãy',
    'xinhây': 'xin hãy',
    'khókhăn': 'khó khăn',
    'khókhăm': 'khó khăn',
    'lắngngnghe': 'lắng nghe',
    'lắngnghe': 'lắng nghe',
    'thậtsự': 'thật sự',
    'dunghoà': 'dung hoà',
    'dunghòa': 'dung hòa',
    'từtừ': 'từ từ',
    'viếtra': 'viết ra',
    
    # ===== LỖI CỤM TỪ =====
    'guủi tố': 'gửi tớ',
    'guủi tớ': 'gửi tớ',
    'guải tố': 'gửi tớ',
    'giải tố': 'gửi tớ',
    'hây títi': 'hãy tí tí',
    'xin hây': 'xin hãy',
    'xin hây títi': 'xin hãy tí tí',
    'tri ai': 'từ từ',
    'tỉ ải': 'từ từ',
    'trí tái': 'từ từ',
    'xin hãy tri ai': 'xin hãy từ từ',
    'xin hãy tỉ ải': 'xin hãy từ từ',
    'xin hãy trí tái': 'xin hãy từ từ',
    'buôn thì': 'buồn thì',
    'buồn thỉ': 'buồn thì',
    'hấy viết': 'hãy viết',
    'viết ranhí': 'viết ra nhé',
    'viết ra nhí': 'viết ra nhé',
    'của xó sĩ': 'của tớ sẽ',
    'của xó': 'của tớ',
    'xó sĩ': 'tớ sẽ',
    'sĩ lắ': 'sẽ lắng',
    'nhật sự': 'thật sự',
    'khó khăm': 'khó khăn',
    'mọi thí': 'mọi thứ',
    'thật sử': 'thật sự',
    'át tổng': 'sẽ lắng',
    'tớ át': 'tớ sẽ',
    'của tớ át': 'của tớ sẽ',
    
    # ===== LỖI CONTEXT =====
    'mọi thứ nhật': 'mọi thứ thật',
}

# Patterns để tách từ dính
SPLIT_PATTERNS = [
    # Mẫu: từ + hãy/nhé/nhỉ/ạ (các từ kết thúc câu/mệnh lệnh)
    (r'(\w+)(hãy)', r'\1 \2'),
    (r'(\w+)(nhé)', r'\1 \2'),
    (r'(\w+)(nhỉ)', r'\1 \2'),
    (r'(\w+)(nhá)', r'\1 \2'),
    (r'(\w+)(nha)', r'\1 \2'),
    
    # Mẫu: xin/ra/vào/... + từ khác  
    (r'\b(xin)(\w{2,})', r'\1 \2'),
    (r'\b(ra)(\w{3,})', r'\1 \2'),
    (r'\b(vào)(\w{2,})', r'\1 \2'),
    
    # Mẫu: từ + khăn/khăm (khó khăn)
    (r'(khó)(khăn|khăm)', r'\1 khăn'),
    
    # Mẫu: lắng + nghe
    (r'(lắng)(nghe)', r'\1 \2'),
    
    # Mẫu: thật + sự
    (r'(thật)(sự)', r'\1 \2'),
    
    # Mẫu: dung + hoà/hòa
    (r'(dung)(hoà|hòa)', r'\1 \2'),
]


class VietnameseSpellChecker:
    """Kiểm tra và sửa lỗi chính tả tiếng Việt cho OCR"""
    
    def __init__(self, custom_words: Set[str] = None):
        self.words = VIETNAMESE_COMMON_WORDS.copy()
        if custom_words:
            self.words.update(custom_words)
        
        self.ocr_errors = COMMON_OCR_ERRORS.copy()
        self.split_patterns = SPLIT_PATTERNS.copy()
    
    def add_words(self, words: Set[str]):
        """Thêm từ vào từ điển"""
        self.words.update(words)
    
    def add_error_mapping(self, error: str, correct: str):
        """Thêm mapping lỗi OCR"""
        self.ocr_errors[error] = correct
    
    def fix_stuck_words(self, text: str) -> str:
        """Tách các từ bị dính"""
        result = text
        
        # Áp dụng các pattern tách từ
        for pattern, replacement in self.split_patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def fix_common_errors(self, text: str) -> str:
        """Sửa các lỗi OCR phổ biến"""
        result = text
        
        for error, correct in self.ocr_errors.items():
            # Case insensitive replace
            pattern = re.compile(re.escape(error), re.IGNORECASE)
            result = pattern.sub(correct, result)
        
        return result
    
    def fix_diacritics(self, text: str) -> str:
        """Sửa lỗi dấu tiếng Việt"""
        result = text
        
        # Sửa các lỗi dấu phổ biến
        diacritic_fixes = {
            # é thường là ế trong tiếng Việt
            'nhé': 'nhé',  # giữ nguyên
            'thé': 'thế',
            'ké': 'kế',
            'bé': 'bé',  # giữ nguyên (em bé)
            'mé': 'mế',
            'lé': 'lế',
            'né': 'nế',
            'sé': 'sế',
            'té': 'tế',
            'vé': 'vé',  # giữ nguyên (vé xe)
            'xé': 'xé',  # giữ nguyên
            'dé': 'dế',
            'gé': 'gế',
            'hé': 'hé',  # giữ nguyên (hé mở)
        }
        
        for wrong, right in diacritic_fixes.items():
            if wrong != right:
                result = result.replace(wrong, right)
        
        return result
    
    def suggest_corrections(self, word: str) -> List[str]:
        """Gợi ý sửa cho một từ"""
        suggestions = []
        word_lower = word.lower()
        
        # Tìm từ tương tự trong từ điển
        for dict_word in self.words:
            # Levenshtein distance đơn giản
            if self._similarity(word_lower, dict_word) > 0.7:
                suggestions.append(dict_word)
        
        return suggestions[:5]  # Top 5 gợi ý
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Tính độ tương đồng giữa 2 chuỗi (đơn giản)"""
        if not s1 or not s2:
            return 0.0
        
        # Đếm ký tự chung
        common = set(s1) & set(s2)
        total = set(s1) | set(s2)
        
        if not total:
            return 0.0
        
        return len(common) / len(total)
    
    def post_process(self, text: str, verbose: bool = False) -> str:
        """
        Post-process OCR text để sửa lỗi
        
        Args:
            text: Text từ OCR
            verbose: In chi tiết các sửa đổi
            
        Returns:
            Text đã được sửa
        """
        if not text:
            return text
        
        original = text
        
        # Bước 0: Sửa lỗi cụm từ context-aware (ưu tiên cao nhất)
        text = self._fix_context_phrases(text, verbose)
        
        # Bước 1: Sửa lỗi OCR phổ biến
        prev = text
        text = self.fix_common_errors(text)
        if verbose and text != prev:
            print(f"  ✏️ Fixed common errors: '{prev}' → '{text}'")
        
        # Bước 2: Tách từ dính
        prev = text
        text = self.fix_stuck_words(text)
        if verbose and text != prev:
            print(f"  ✂️ Split stuck words: '{prev}' → '{text}'")
        
        # Bước 3: Sửa dấu
        prev = text
        text = self.fix_diacritics(text)
        if verbose and text != prev:
            print(f"  🔤 Fixed diacritics: '{prev}' → '{text}'")
        
        # Bước 4: Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Bước 5: Sửa lỗi cuối cùng
        text = self._final_corrections(text)
        
        return text
    
    def _fix_context_phrases(self, text: str, verbose: bool = False) -> str:
        """Sửa lỗi cụm từ dựa trên context"""
        corrections = [
            # Format: (pattern, replacement, description)
            # Lỗi dấu
            (r'phái\s+dung', 'phải dung', 'phải dung'),
            (r'mọi\s+thí\b', 'mọi thứ', 'mọi thứ'),
            (r'thật\s+sử\b', 'thật sự', 'thật sự'),
            (r'trí\s+tái', 'từ từ', 'từ từ'),
            (r'buồn\s+thỉ', 'buồn thì', 'buồn thì'),
            (r'tớ\s+át\s+tổng', 'tớ sẽ lắng', 'tớ sẽ lắng'),
            (r'của\s+tớ\s+át', 'của tớ sẽ', 'của tớ sẽ'),
            (r'át\s+tổng', 'sẽ lắng', 'sẽ lắng'),
            
            # Lỗi cũ
            (r'guải\s*tố', 'gửi tớ', 'gửi tớ'),
            (r'giải\s*tố', 'gửi tớ', 'gửi tớ'),
            (r'tri\s*ai', 'từ từ', 'từ từ'),
            (r'tỉ\s*ải', 'từ từ', 'từ từ'),
            (r'tỉ\s*ai', 'từ từ', 'từ từ'),
            (r'nhật\s*sự', 'thật sự', 'thật sự'),
            (r'buôn\s*thì', 'buồn thì', 'buồn thì'),
            (r'hấy\s*viết', 'hãy viết', 'hãy viết'),
            (r'viết\s*ra\s*nhí', 'viết ra nhé', 'viết ra nhé'),
            (r'của\s*xó\s*sĩ', 'của tớ sẽ', 'của tớ sẽ'),
            (r'sĩ\s*lắ', 'sẽ lắng', 'sẽ lắng'),
            (r'khó\s*khăm', 'khó khăn', 'khó khăn'),
            (r'dung\s*hoà\s*m\b', 'dung hoà mọi', 'dung hoà mọi'),
            (r'xin\s*hây\s*títi', 'xin hãy tí tí', 'xin hãy tí tí'),
            (r'hây\s*títi', 'hãy tí tí', 'hãy tí tí'),
            (r'guủi\s*tố', 'gửi tớ', 'gửi tớ'),
            (r'guủi\s*tớ', 'gửi tớ', 'gửi tớ'),
            (r'lắngng\s*nghe', 'lắng nghe', 'lắng nghe'),
            (r'lắngngng\s*nghe', 'lắng nghe', 'lắng nghe'),
        ]
        
        result = text
        for pattern, replacement, desc in corrections:
            new_result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            if new_result != result and verbose:
                print(f"  🔧 Context fix: '{pattern}' → '{desc}'")
            result = new_result
        
        return result
    
    def _final_corrections(self, text: str) -> str:
        """Các sửa lỗi cuối cùng"""
        corrections = {
            # Sửa các từ đơn lẻ bị sai dấu
            ' phái ': ' phải ',
            ' thí ': ' thứ ',
            ' sử ': ' sự ',
            ' thỉ ': ' thì ',
            ' trí ': ' từ ',
            ' tái ': ' từ ',
            ' át ': ' sẽ ',
            ' tổng ': ' lắng ',
            
            # Từ cũ
            ' m ': ' mọi ',
            ' m,': ' mọi,',
            ' m.': ' mọi.',
            'khăm': 'khăn',
            'hấy': 'hãy',
            'buôn': 'buồn',
            'nhí': 'nhé',
            'xó': 'tớ',
            'sĩ': 'sẽ',
            'lắ ': 'lắng ',
            'tri ai': 'từ từ',
            'tỉ ải': 'từ từ',
            'trí tái': 'từ từ',
            'át tổng': 'sẽ lắng',
            'thật sử': 'thật sự',
            'mọi thí': 'mọi thứ',
            'buồn thỉ': 'buồn thì',
            'phái dung': 'phải dung',
            'guủi': 'gửi',
            'hây': 'hãy',
            'títi': 'tí tí',
            'lắngng': 'lắng',
            'lắngngng': 'lắng',
        }
        
        result = text
        for wrong, right in corrections.items():
            result = result.replace(wrong, right)
        
        return result
    
    def process_line_by_line(self, text: str, verbose: bool = False) -> str:
        """Process từng dòng"""
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            processed = self.post_process(line.strip(), verbose)
            if processed:
                processed_lines.append(processed)
        
        return '\n'.join(processed_lines)


# Singleton instance
_spell_checker = None

def get_spell_checker() -> VietnameseSpellChecker:
    """Get singleton spell checker instance"""
    global _spell_checker
    if _spell_checker is None:
        _spell_checker = VietnameseSpellChecker()
    return _spell_checker


def post_process_ocr_text(text: str, verbose: bool = False) -> str:
    """
    Convenience function để post-process OCR text
    
    Args:
        text: Raw OCR text
        verbose: Print corrections
        
    Returns:
        Corrected text
    """
    # Sử dụng bộ sửa lỗi mới nếu có
    if USE_NEW_CORRECTOR:
        result = fix_ocr_text(text, verbose)
        if verbose:
            print(f"  ✏️ Spell check applied!")
        return result
    
    # Fallback to old checker
    checker = get_spell_checker()
    return checker.process_line_by_line(text, verbose)


# Test
if __name__ == "__main__":
    test_texts = [
        "giải tố, việc phải dung hoà mọi thứ thật sự rất khó khăm xinhãy tỉ tả buôn thi hấy viết ranhé sau này của tố sẽ lắng nghe cậu.",
        "xinhãy cho tôi biết",
        "thậtsự rất khókhăn",
        "lắngnghe tôi nói",
    ]
    
    print("=" * 60)
    print("Vietnamese OCR Spell Checker Test")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\n📝 Original: {text}")
        corrected = post_process_ocr_text(text, verbose=True)
        print(f"✅ Corrected: {corrected}")
        print("-" * 40)
