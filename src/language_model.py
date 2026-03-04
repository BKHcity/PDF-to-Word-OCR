"""
Vietnamese Language Model for OCR Post-correction
Uses context-aware corrections for Vietnamese text
"""

import re
from typing import List, Dict, Set
from collections import Counter
import unicodedata

class VietnameseLanguageModel:
    """Context-aware language model for Vietnamese OCR correction"""

    def __init__(self):
        # Common Vietnamese words and patterns
        self.common_words = {
            'xin', 'chào', 'cảm', 'ơn', 'tôi', 'bạn', 'hôm', 'nay',
            'trời', 'nắng', 'đẹp', 'đang', 'học', 'làm', 'việc',
            'thời', 'gian', 'người', 'cuộc', 'sống', 'yêu', 'thương',
            'gia', 'đình', 'bố', 'mẹ', 'con', 'cái', 'nhà', 'đi',
            'đến', 'từ', 'với', 'cho', 'của', 'là', 'có', 'không',
            'sao', 'thế', 'nào', 'đây', 'kia', 'ấy', 'mình', 'tớ'
        }

        # Vietnamese character patterns
        self.vietnamese_patterns = [
            r'quyển', r'quyết', r'quốc', r'quân', r'quản',
            r'người', r'nghe', r'nghĩ', r'nghĩa', r'nghề',
            r'thương', r'thường', r'thực', r'thuyết', r'thắng'
        ]

    def correct_text(self, text: str) -> str:
        """
        Apply language model corrections to OCR text

        Args:
            text: Raw OCR text

        Returns:
            Corrected text
        """
        if not text or not text.strip():
            return text

        corrected = text.lower().strip()

        # Step 1: Fix common OCR errors
        corrected = self._fix_common_errors(corrected)

        # Step 2: Apply Vietnamese-specific corrections
        corrected = self._fix_vietnamese_patterns(corrected)

        # Step 3: Capitalize properly
        corrected = self._capitalize_sentences(corrected)

        return corrected

    def _fix_common_errors(self, text: str) -> str:
        """Fix common OCR recognition errors"""
        # Common substitution errors
        fixes = {
            '1': 'i',  # Number 1 often misread as i
            '0': 'o',  # Number 0 often misread as o
            'l': 'i',  # lowercase L often misread as i
            'rn': 'm', # rn often misread as m
            'cl': 'd', # cl often misread as d
            'vv': 'w', # vv often misread as w
        }

        for old, new in fixes.items():
            text = text.replace(old, new)

        return text

    def _fix_vietnamese_patterns(self, text: str) -> str:
        """Apply Vietnamese-specific corrections"""
        words = text.split()

        # Score each word against common Vietnamese words
        corrected_words = []
        for word in words:
            # Check if word exists in common words
            if word in self.common_words:
                corrected_words.append(word)
            else:
                # Try fuzzy matching with common words
                candidates = self._find_similar_words(word, self.common_words)
                if candidates:
                    corrected_words.append(candidates[0])
                else:
                    corrected_words.append(word)

        return ' '.join(corrected_words)

    def _find_similar_words(self, word: str, word_list: Set[str], threshold: float = 0.8) -> List[str]:
        """Find similar words using simple edit distance"""
        similar = []
        for candidate in word_list:
            distance = self._levenshtein_distance(word, candidate)
            max_len = max(len(word), len(candidate))
            similarity = 1 - (distance / max_len) if max_len > 0 else 0
            if similarity >= threshold:
                similar.append((candidate, similarity))

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in similar[:3]]  # Return top 3

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize the first letter of sentences"""
        sentences = re.split(r'([.!?]\s*)', text)
        result = []

        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Actual sentence content
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
            result.append(sentence)

        return ''.join(result)

    def get_confidence_score(self, text: str) -> float:
        """Calculate confidence score based on language model"""
        if not text:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        # Count known words
        known_words = sum(1 for word in words if word in self.common_words)
        coverage = known_words / len(words)

        # Vietnamese character ratio
        vietnamese_chars = sum(1 for char in text if self._is_vietnamese_char(char))
        char_ratio = vietnamese_chars / len(text) if text else 0

        # Combine scores
        return (coverage * 0.6) + (char_ratio * 0.4)

    def _is_vietnamese_char(self, char: str) -> bool:
        """Check if character is Vietnamese-specific"""
        vietnamese_chars = 'áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        return char.lower() in vietnamese_chars






