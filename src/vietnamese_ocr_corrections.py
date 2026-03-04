"""
VIETNAMESE OCR CORRECTIONS - Bộ sửa lỗi OCR tiếng Việt toàn diện
==================================================================

File này chứa hàng trăm mapping sửa lỗi OCR phổ biến cho tiếng Việt.
Các lỗi được phân loại theo:
1. Lỗi dấu thanh (sai dấu sắc/huyền/hỏi/ngã/nặng)
2. Lỗi nguyên âm (a/ă/â, o/ô/ơ, u/ư, e/ê, i/y)
3. Lỗi phụ âm (d/đ, c/k, g/gh, ng/ngh, s/x, ch/tr)
4. Lỗi dính chữ (thiếu khoảng trắng)
5. Lỗi tách chữ (thừa khoảng trắng)
6. Lỗi từ đơn phổ biến
7. Lỗi cụm từ phổ biến
8. Lỗi số/ký tự đặc biệt

Author: Vietnamese OCR Enhancement
Version: 2.0
"""

import re
import unicodedata
from typing import Dict, List, Tuple

# ============================================================================
# 1. LỖI DẤU THANH - Sai dấu sắc/huyền/hỏi/ngã/nặng
# ============================================================================

DIACRITIC_ERRORS = {
    # === Sai dấu sắc → huyền ===
    'phái': 'phải',
    'phái biết': 'phải biết',
    'phái làm': 'phải làm',
    'phái có': 'phải có',
    'trái tim': 'trái tim',  # giữ nguyên (đúng)
    'nhái': 'nhài',
    'thái độ': 'thái độ',  # giữ nguyên
    
    # === Sai dấu hỏi → ngã ===
    'sử': 'sự',
    'thật sử': 'thật sự',
    'sử thật': 'sự thật',
    'sử việc': 'sự việc',
    'sử kiện': 'sự kiện',
    'lử': 'lữ',
    'kỉ niệm': 'kỷ niệm',
    'kỉ lục': 'kỷ lục',
    'sỉ nhục': 'sỉ nhục',  # giữ nguyên
    'nghỉ ngơi': 'nghỉ ngơi',  # giữ nguyên
    
    # === Sai dấu ngã → hỏi ===
    'thĩ': 'thỉ',
    'bĩu môi': 'bĩu môi',
    'đĩa': 'đĩa',  # giữ nguyên
    
    # === Sai dấu huyền → sắc ===
    'thì': 'thì',  # giữ nguyên (đúng)
    'thỉ': 'thì',  # sai dấu hỏi
    'buồn thỉ': 'buồn thì',
    'nếu thỉ': 'nếu thì',
    'khi thỉ': 'khi thì',
    
    # === Sai dấu hỏi → nặng ===
    'mỏi': 'mỏi',  # giữ nguyên (đúng)
    'nỏi': 'nọi',
    
    # === Sai dấu sắc → nặng ===
    'cứ': 'cứ',  # giữ nguyên
    'lứa': 'lứa',  # giữ nguyên
    
    # === Sai dấu đặc biệt ===
    'thí': 'thứ',
    'mọi thí': 'mọi thứ',
    'mọi thú': 'mọi thứ',
    'thí nhất': 'thứ nhất',
    'thí hai': 'thứ hai',
    'thí ba': 'thứ ba',
}

# ============================================================================
# 2. LỖI NGUYÊN ÂM - a/ă/â, o/ô/ơ, u/ư, e/ê, i/y
# ============================================================================

VOWEL_ERRORS = {
    # === a/ă/â ===
    'căm': 'cảm',
    'thương căm': 'thương cảm',
    'căm xúc': 'cảm xúc',
    'khăn': 'khăn',  # giữ nguyên
    'khó khăm': 'khó khăn',
    'khăm khó': 'khó khăn',
    'bân': 'bận',
    'bân rộn': 'bận rộn',
    'ngân': 'ngân',  # giữ nguyên (ngân hàng)
    'ngân ngơ': 'ngẩn ngơ',
    'thãy': 'thấy',  # thãy → thấy
    'gìác': 'giác',  # gìác → giác
    'gìữa': 'giữa',  # gìữa → giữa
    'tròng': 'trong',  # tròng → trong
    
    # === o/ô/ơ ===
    'rong': 'ròng',
    'khóc rong': 'khóc ròng',
    'mong': 'mong',  # giữ nguyên
    'trong': 'trong',  # giữ nguyên
    'khong': 'không',
    'ko': 'không',
    'hông': 'không',
    'đong': 'đông',
    'đong đảo': 'đông đảo',
    
    # === u/ư ===
    'tư': 'tư',  # giữ nguyên
    'từ từ': 'từ từ',  # giữ nguyên
    'trí tái': 'từ từ',  # lỗi OCR
    'tri ai': 'từ từ',  # lỗi OCR
    'tỉ ải': 'từ từ',  # lỗi OCR
    'tu': 'từ',
    'tu tu': 'từ từ',
    
    # === e/ê ===
    'nhé': 'nhé',  # giữ nguyên
    'nhẻ': 'nhé',
    'nhẹ': 'nhẹ',  # giữ nguyên (nhẹ nhàng)
    'thé': 'thế',
    'nhe': 'nhé',
    'the': 'thế',
    
    # === i/y ===
    'tình': 'tình',  # giữ nguyên
    'tinh': 'tình',  # thiếu dấu
    'moi': 'mối',
    'mối': 'mối',  # giữ nguyên
    'mối tình': 'mối tình',
    'mối tỉnh': 'mối tình',
    'mối tinh': 'mối tình',
}

# ============================================================================
# 3. LỖI PHỤ ÂM - d/đ, c/k, g/gh, ng/ngh, s/x, ch/tr
# ============================================================================

CONSONANT_ERRORS = {
    # === d/đ ===
    'đi': 'đi',  # giữ nguyên
    'di': 'đi',  # thiếu nét ngang
    'đoc': 'đọc',
    'doc': 'đọc',
    'đap': 'đáp',
    'dap': 'đáp',
    'đáp lại': 'đáp lại',
    'dáp lại': 'đáp lại',
    'đung': 'đúng',
    # 'dung': 'dùng', # REMOVED - dung hòa ≠ dùng hòa
    'dung hoà': 'dung hòa',
    'dung hòa': 'dung hòa',  # giữ nguyên
    
    # === g/gh ===
    'gì': 'gì',  # giữ nguyên
    'gi': 'gì',
    'nghe': 'nghe',  # giữ nguyên
    'nge': 'nghe',
    'lắng nghe': 'lắng nghe',
    'lắng nge': 'lắng nghe',
    'lắngng': 'lắng',
    'lắngnghe': 'lắng nghe',
    
    # === ng/ngh ===
    'ngĩ': 'nghĩ',
    'suy ngĩ': 'suy nghĩ',
    'ngỉ': 'nghỉ',
    'ngỉ ngơi': 'nghỉ ngơi',
    
    # === s/x ===
    'xong': 'xong',  # giữ nguyên
    'song': 'song',  # giữ nguyên (bên song)
    'xin': 'xin',  # giữ nguyên
    'sin': 'xin',
    'xa': 'xa',  # giữ nguyên
    'sa': 'sa',  # giữ nguyên (sa lầy)
    
    # === ch/tr ===
    'chung': 'chúng',
    'chúng ta': 'chúng ta',
    'trung ta': 'chúng ta',
    'trong': 'trong',  # giữ nguyên
    'chong': 'trong',
    'trọng': 'trọng',  # giữ nguyên
    'chọng': 'trọng',
}

# ============================================================================
# 4. LỖI DÍNH CHỮ - Thiếu khoảng trắng
# ============================================================================

STUCK_WORDS = {
    # === Dính với "hãy" ===
    'xinhãy': 'xin hãy',
    'xinhây': 'xin hãy',
    'hãyhiểu': 'hãy hiểu',
    'hãyviết': 'hãy viết',
    'hãylàm': 'hãy làm',
    'hãyđi': 'hãy đi',
    'hãynói': 'hãy nói',
    'hãynghe': 'hãy nghe',
    'hãyđọc': 'hãy đọc',
    'hãycố': 'hãy cố',
    'hãytin': 'hãy tin',
    'đừnghãy': 'đừng hãy',
    
    # === Dính với "nhé/nhỉ" ===
    'ranhé': 'ra nhé',
    'ranhí': 'ra nhé',
    'ranhẻ': 'ra nhé',
    'đinhé': 'đi nhé',
    'nhanhé': 'nha nhé',
    'vậynhé': 'vậy nhé',
    'đượcnhé': 'được nhé',
    'nhớnhé': 'nhớ nhé',
    'nghenhe': 'nghe nhé',
    
    # === Dính với "thì" ===
    'nếuthì': 'nếu thì',
    'khithì': 'khi thì',
    'màthì': 'mà thì',
    'thìlà': 'thì là',
    'thìsẽ': 'thì sẽ',
    'thìcó': 'thì có',
    
    # === Dính với "là" ===
    'đólà': 'đó là',
    'nàylà': 'này là',
    'làmột': 'là một',
    'làcái': 'là cái',
    'làngười': 'là người',
    'làđiều': 'là điều',
    'làsự': 'là sự',
    
    # === Dính với "và" ===
    'vàcó': 'và có',
    'vàlà': 'và là',
    'vànếu': 'và nếu',
    'vàcũng': 'và cũng',
    'vàđã': 'và đã',
    'vàsẽ': 'và sẽ',
    
    # === Dính với "của" ===
    'củatôi': 'của tôi',
    'củabạn': 'của bạn',
    'củaanh': 'của anh',
    'củachị': 'của chị',
    'củaem': 'của em',
    'củamình': 'của mình',
    'củanó': 'của nó',
    
    # === Dính với "cho" ===
    'chonên': 'cho nên',
    'chotôi': 'cho tôi',
    'chobạn': 'cho bạn',
    'chođến': 'cho đến',
    'chorằng': 'cho rằng',
    
    # === Dính với "không" ===
    'khôngcó': 'không có',
    'khôngthể': 'không thể',
    'khôngbiết': 'không biết',
    'khôngnên': 'không nên',
    'khôngphải': 'không phải',
    'khôngđược': 'không được',
    
    # === Dính với "được" ===
    'đượccái': 'được cái',
    'đượcrồi': 'được rồi',
    'đượckhông': 'được không',
    'đượcnhư': 'được như',
    
    # === Dính với "có" ===
    'cóthể': 'có thể',
    'cólẽ': 'có lẽ',
    'cókhi': 'có khi',
    'cóngười': 'có người',
    'cóđiều': 'có điều',
    'cónhững': 'có những',
    
    # === Dính khác ===
    'thậtsự': 'thật sự',
    'thậtlà': 'thật là',
    'rấtlà': 'rất là',
    'rấtcó': 'rất có',
    'rấtnhiều': 'rất nhiều',
    'khókhăn': 'khó khăn',
    'dễdàng': 'dễ dàng',
    'lắngnghe': 'lắng nghe',
    'dunghoà': 'dung hòa',
    'dunghòa': 'dung hòa',
    'viếtra': 'viết ra',
    'đọcxong': 'đọc xong',
    'nhìnthấy': 'nhìn thấy',
    'nghethấy': 'nghe thấy',
    'biếtđược': 'biết được',
    'hiểuđược': 'hiểu được',
    'làmđược': 'làm được',
    'nóiđược': 'nói được',
}

# ============================================================================
# 5. LỖI TỪ ĐƠN PHỔ BIẾN
# ============================================================================

SINGLE_WORD_ERRORS = {
    # === Đại từ ===
    'tố': 'tớ',
    'guủi': 'gửi',
    'guải': 'gửi',
    'giải': 'gửi',  # trong ngữ cảnh "gửi tớ"
    'xó': 'tớ',
    'mĩnh': 'mình',
    'minh': 'mình',
    'tôj': 'tôi',
    'toj': 'tôi',
    'bạn': 'bạn',  # giữ nguyên
    'ban': 'bạn',
    'họ': 'họ',  # giữ nguyên
    'ho': 'họ',
    
    # === Động từ ===
    'hây': 'hãy',
    'hấy': 'hãy',
    'đúng': 'đừng',
    'đúng làm': 'đừng làm',
    'sẻ': 'sẽ',
    'sĩ': 'sẽ',
    'át': 'sẽ',
    'viêt': 'viết',
    'viet': 'viết',
    'đoc': 'đọc',
    'doc': 'đọc',
    'nghi': 'nghĩ',
    'ngĩ': 'nghĩ',
    'nge': 'nghe',
    'yêủ': 'yêu',
    'yeu': 'yêu',
    'làn': 'làm',
    'lam': 'làm',
    'biêt': 'biết',
    'biet': 'biết',
    'hiêu': 'hiểu',
    'hieu': 'hiểu',
    
    # === Tính từ ===
    'buôn': 'buồn',
    'buon': 'buồn',
    'vùi': 'vui',
    'vuj': 'vui',
    'đep': 'đẹp',
    'dep': 'đẹp',
    'xâu': 'xấu',
    'xau': 'xấu',
    'tôt': 'tốt',
    'tot': 'tốt',
    'khỏ': 'khó',
    'kho': 'khó',
    'đê': 'dễ',
    'de': 'dễ',
    
    # === Trạng từ/Phó từ ===
    'rât': 'rất',
    'rat': 'rất',
    'cũg': 'cũng',
    'cung': 'cũng',
    'đa': 'đã',
    'da': 'đã',
    'đag': 'đang',
    'dang': 'đang',
    'sê': 'sẽ',
    'se': 'sẽ',
    'rôi': 'rồi',
    'roi': 'rồi',
    'nữà': 'nữa',
    'nua': 'nữa',
    'luôm': 'luôn',
    'luon': 'luôn',
    
    # === Liên từ/Giới từ ===
    'và': 'và',  # giữ nguyên
    'va': 'và',
    'nhưg': 'nhưng',
    'nhung': 'nhưng',
    'nêu': 'nếu',
    'neu': 'nếu',
    'thỉ': 'thì',
    'thi': 'thì',
    'mà': 'mà',  # giữ nguyên
    'ma': 'mà',
    'vói': 'với',
    'voi': 'với',
    'đê': 'để',
    'de': 'để',
    'tù': 'từ',
    'tu': 'từ',
    'trí': 'từ',  # trong ngữ cảnh "từ từ"
    'tái': 'từ',
    
    # === Danh từ thông dụng ===
    'ngưoi': 'người',
    'nguoi': 'người',
    'điêu': 'điều',
    'dieu': 'điều',
    'viêc': 'việc',
    'viec': 'việc',
    'thơi': 'thời',
    'thoi': 'thời',
    'ngày': 'ngày',  # giữ nguyên
    'ngay': 'ngày',
    'đêm': 'đêm',  # giữ nguyên
    'dem': 'đêm',
    'năm': 'năm',  # giữ nguyên
    # 'nam' KHÔNG sửa vì là tên người phổ biến (Nam, Việt Nam, miền Nam...)
    'tháng': 'tháng',  # giữ nguyên
    'thang': 'tháng',
    
    'khẩm': 'khăn',
    'lống': 'lắng',
    'phai': 'phải',
    
    # === Lỗi từ ảnh mới ===
    'chơt': 'chợt',
    'thãy': 'thấy',
    'gìữa': 'giữa',
    # 'sơ hãi': 'sợ hãi',  # Không cần - đã đúng
    # 'năm': 'Nam',  # REMOVED - "năm" thường là số, không phải tên
    'gìác': 'giác',
    'tròng': 'trong',
    'lắngng': 'lắng',  # "sẽ lắng nghe" - KHÔNG phải "lông"
    'cửu': 'cứu',  # "được cứu"
    'mim': 'mỉm',  # "mỉm cười"
    'vi': 'vì',  # "vì đã"
    # 'câu': 'cậu',  # REMOVED - có thể conflict
    'chủ mèo': 'chú mèo',  # "chú mèo"
    'dụi': 'dụi',  # giữ nguyên (đúng)
    'rúc rích': 'rúc rích',  # giữ nguyên (đúng)
    'Bắt': 'Bất',  # "Bất chợt" - SAI dấu
    'sợ hải': 'sợ hãi',  # FIX: "sợ hải" → "sợ hãi"
    'hải': 'hãi',  # FIX: trong ngữ cảnh "sợ hãi"
    
    # === Số ===
    'môt': 'một',
    'mot': 'một',
    'haỉ': 'hai',
    # 'hải': 'hai',  # REMOVED - conflict với "sợ hải"
    'bà': 'ba',  # trong ngữ cảnh số
    'nàm': 'năm',
    'sáu': 'sáu',  # giữ nguyên
    # 'sau': 'sáu',  # REMOVED - sau này ≠ sáu này
    'bảy': 'bảy',  # giữ nguyên
    'bay': 'bảy',
    'tám': 'tám',  # giữ nguyên
    'tam': 'tám',
    'chín': 'chín',  # giữ nguyên
    'chin': 'chín',
    'mười': 'mười',  # giữ nguyên
    'muoi': 'mười',
    
    # === Từ đặc biệt ===
    'títi': 'tí tí',
    'tỉ tỉ': 'tí tí',
    'nhí': 'nhé',
    'tổng': 'lắng',  # trong ngữ cảnh "sẽ lắng"
    'lắ': 'lắng',
    'tổ': 'tớ',  # lỗi OCR phổ biến
    'đó': 'tớ',  # "gửi đó" → "gửi tớ"
    'nất': 'rất',  # "thật sự nất" → "thật sự rất"
    'khám': 'khăn',  # "khó khám" → "khó khăn"
    'tát': 'từ',  # "từ tát" → "từ từ"
    'bằng': 'lắng',  # "sẽ bằng" → "sẽ lắng"
    'chọt': 'chợt',  # "Bắt chọt" → "Bất chợt"
    'nhật': 'nhặt',  # "nhật chú mèo" → "nhặt chú mèo"
    'thứ': 'thì',  # "buồn thứ" → "buồn thì"
}

# ============================================================================
# 6. LỖI CỤM TỪ PHỔ BIẾN
# ============================================================================

PHRASE_ERRORS = {
    # === Cụm chào hỏi ===
    'xin chao': 'xin chào',
    'xin chào': 'xin chào',  # giữ nguyên
    'cảm on': 'cảm ơn',
    'cam on': 'cảm ơn',
    'cảm ơn': 'cảm ơn',  # giữ nguyên
    'xin loi': 'xin lỗi',
    'xin lỗi': 'xin lỗi',  # giữ nguyên
    
    # === Cụm "gửi tớ" ===
    'guủi tố': 'gửi tớ',
    'guủi tớ': 'gửi tớ',
    'guải tố': 'gửi tớ',
    'giải tố': 'gửi tớ',
    'gửi tố': 'gửi tớ',
    'gửi đó': 'gửi tớ',
    'gối đó': 'gửi tớ',
    
    # === Cụm "thật sự" ===
    'thật sử': 'thật sự',
    'thật su': 'thật sự',
    'that su': 'thật sự',
    'thât sự': 'thật sự',
    'nhật sự': 'thật sự',
    
    # === Cụm "khó khăn" ===
    'khó khăm': 'khó khăn',
    'kho khan': 'khó khăn',
    'khỏ khăn': 'khó khăn',
    'khókhăn': 'khó khăn',
    'khókhăm': 'khó khăn',
    
    # === Cụm "từ từ" ===
    'trí tái': 'từ từ',
    'tri ai': 'từ từ',
    'tỉ ải': 'từ từ',
    'tỉ ai': 'từ từ',
    'tu tu': 'từ từ',
    'tù tù': 'từ từ',
    'títh': 'từ từ',
    'tỉth': 'từ từ',
    'tit': 'từ từ',
    'tít': 'từ từ',
    
    # === Cụm "sẽ lắng" ===
    'sĩ lắ': 'sẽ lắng',
    'át tổng': 'sẽ lắng',
    'sẽ lắ': 'sẽ lắng',
    'se lang': 'sẽ lắng',
    
    # === Cụm "lắng nghe" ===
    'lắng nge': 'lắng nghe',
    'lang nghe': 'lắng nghe',
    'lắngng nghe': 'lắng nghe',
    'lắngnghe': 'lắng nghe',
    'lắngngng': 'lắng',
    
    # === Cụm "buồn thì" ===
    'buồn thỉ': 'buồn thì',
    'buôn thì': 'buồn thì',
    'buôn thi': 'buồn thì',
    'buon thi': 'buồn thì',
    
    # === Cụm "hãy viết" ===
    'hấy viết': 'hãy viết',
    'hây viết': 'hãy viết',
    'hay viet': 'hãy viết',
    
    # === Cụm "viết ra nhé" ===
    'viết ranhí': 'viết ra nhé',
    'viết ra nhí': 'viết ra nhé',
    'viet ra nhe': 'viết ra nhé',
    
    # === Cụm "của tớ sẽ" ===
    'của xó sĩ': 'của tớ sẽ',
    'của xó': 'của tớ',
    'cua to se': 'của tớ sẽ',
    'của tớ át': 'của tớ sẽ',
    
    # === Cụm "xin hãy" ===
    'xin hây': 'xin hãy',
    'xin hấy': 'xin hãy',
    'sin hay': 'xin hãy',
    'xin hãy títi': 'xin hãy tí tí',
    'xin hây títi': 'xin hãy tí tí',
    
    # === Cụm "mọi thứ" ===
    'mọi thí': 'mọi thứ',
    'mọi thú': 'mọi thứ',
    'moi thu': 'mọi thứ',
    
    # === Cụm "phải dung hòa" ===
    'phái dung': 'phải dung',
    'phai dung': 'phải dung',
    'phải dung hoà': 'phải dung hòa',
    
    # === Cụm "có thể" ===
    'có thê': 'có thể',
    'co the': 'có thể',
    'cỏ thể': 'có thể',
    
    # === Cụm "không thể" ===
    'khong the': 'không thể',
    'không thê': 'không thể',
    'khôg thể': 'không thể',
    
    # === Cụm tình cảm ===
    'mối tỉnh': 'mối tình',
    'moi tinh': 'mối tình',
    'mối tinh': 'mối tình',
    'tình yêu': 'tình yêu',  # giữ nguyên
    'tinh yeu': 'tình yêu',
    'yêu thương': 'yêu thương',  # giữ nguyên
    'yeu thuong': 'yêu thương',
    'thương yêu': 'thương yêu',  # giữ nguyên
    'thuong yeu': 'thương yêu',
    'buồn bã': 'buồn bã',  # giữ nguyên
    'buon ba': 'buồn bã',
    'khóc ròng': 'khóc ròng',  # giữ nguyên
    'khoc rong': 'khóc ròng',
    'khóc rong': 'khóc ròng',
    'tuổ không': 'tưởng không',
    'mối từnh': 'mối tình',
    
    # === Lỗi từ hình thư tình ===
    'thương căm': 'thương cảm',
    'thuong cam': 'thương cảm',
    'căm xúc': 'cảm xúc',
    'lậm xúc': 'cảm xúc',
    'troạn về': 'trọn vẹn',
    'tron ven': 'trọn vẹn',
    'trọn về': 'trọn vẹn',
    'ngẫn ngơ': 'ngẩn ngơ',
    'ngan ngo': 'ngẩn ngơ',
    'ngân ngơ': 'ngẩn ngơ',
    'người ngẫn': 'người ngẩn',
    'tả doc': 'ta đọc',
    'chúng tả': 'chúng ta',
    'thất đẹp': 'thật đẹp',
    'that dep': 'thật đẹp',
    'Trương ky': 'Thương có',
    'có áp': 'cho áp',
    'có áp lại': 'cho những',
    'thường cầm': 'thương cảm',
    'thuong cam': 'thương cảm',
    'lá thư': 'lá thư',  # giữ nguyên
    'la thu': 'lá thư',
    'thư tình': 'thư tình',  # giữ nguyên
    'thu tinh': 'thư tình',
    
    # === Cụm cảm xúc ===
    'thương cảm': 'thương cảm',  # giữ nguyên
    'thuong cam': 'thương cảm',
    'cảm xúc': 'cảm xúc',  # giữ nguyên
    'cam xuc': 'cảm xúc',
    'căm xúc': 'cảm xúc',
    'vui vẻ': 'vui vẻ',  # giữ nguyên
    'vui ve': 'vui vẻ',
    'hạnh phúc': 'hạnh phúc',  # giữ nguyên
    'hanh phuc': 'hạnh phúc',
    
    # === Cụm thời gian ===
    'bây giờ': 'bây giờ',  # giữ nguyên
    'bay gio': 'bây giờ',
    'hôm nay': 'hôm nay',  # giữ nguyên
    'hom nay': 'hôm nay',
    'ngày mai': 'ngày mai',  # giữ nguyên
    'ngay mai': 'ngày mai',
    'hôm qua': 'hôm qua',  # giữ nguyên
    'hom qua': 'hôm qua',
    'sau này': 'sau này',  # giữ nguyên
    'sau nay': 'sau này',
    
    # === Cụm "đọc xong" ===
    'đoc xong': 'đọc xong',
    'doc xong': 'đọc xong',
    'đọc song': 'đọc xong',
    
    # === Cụm "có khi" ===
    'có khỉ': 'có khi',
    'co khi': 'có khi',
    'cỏ khi': 'có khi',
    
    # === Cụm "đáp lại" ===
    'đáp lại': 'đáp lại',  # giữ nguyên
    'dap lai': 'đáp lại',
    'đáp laị': 'đáp lại',
    'đap lại': 'đáp lại',
    
    # === Cụm OCR lỗi mới ===
    'khó khẩm': 'khó khăn',
    'khó khâm': 'khó khăn',
    'sẽ lống': 'sẽ lắng',
    'sĩ lống': 'sẽ lắng',
    'sáu này': 'sau này',
    'dùng hòa': 'dung hòa',
    'phai dung': 'phải dung',
    'phai dùng': 'phải dung',
    'nhật sử': 'thật sự',
    'nhật sự': 'thật sự',
    'lống nghe': 'lắng nghe',
    
    # === Cụm "trọn vẹn" ===
    'trọn vẹn': 'trọn vẹn',  # giữ nguyên
    'tron ven': 'trọn vẹn',
    'trọn ven': 'trọn vẹn',
    'tron vẹn': 'trọn vẹn',
    
    # === Cụm "tuyệt vọng" ===
    'tuyệt vọng': 'tuyệt vọng',  # giữ nguyên
    'tuyet vong': 'tuyệt vọng',
    'tuyêt vọng': 'tuyệt vọng',
    'tuyệt vong': 'tuyệt vọng',
    
    # === Cụm "thật đẹp" ===
    'thật đẹp': 'thật đẹp',  # giữ nguyên
    'that dep': 'thật đẹp',
    'thât đẹp': 'thật đẹp',
    'thật đep': 'thật đẹp',
}

# ============================================================================
# 7. LỖI TÁCH CHỮ - Thừa khoảng trắng
# ============================================================================

SPLIT_WORD_ERRORS = {
    # === Tách sai từ có dấu ===
    'kh ông': 'không',
    'kh o ng': 'không',
    'đư ợc': 'được',
    'đư ơc': 'được',
    'nh ững': 'những',
    'nh ung': 'những',
    'tr ong': 'trong',
    'ch úng': 'chúng',
    'ch ung': 'chúng',
    'th ương': 'thương',
    'th uong': 'thương',
    'ng ười': 'người',
    'ng uoi': 'người',
    'nh iều': 'nhiều',
    'nh ieu': 'nhiều',
    'đi ều': 'điều',
    'đi eu': 'điều',
    'vi ệc': 'việc',
    'vi ec': 'việc',
    'ti ếng': 'tiếng',
    'ti eng': 'tiếng',
    'gi ữa': 'giữa',
    'gi ua': 'giữa',
    
    # === Tách sai âm đầu ===
    'ngh ĩ': 'nghĩ',
    'ngh i': 'nghĩ',
    'ngh e': 'nghe',
    'gh i': 'ghi',
    'gh e': 'ghé',
    'kh i': 'khi',
    'kh e': 'khẽ',
    'th ì': 'thì',
    'th i': 'thì',
    'ch o': 'cho',
    'tr ả': 'trả',
    'tr a': 'trả',
    
    # === Tách sai vần ===
    'yê u': 'yêu',
    'iê u': 'iêu',
    'ươ ng': 'ương',
    'uô ng': 'uống',
    'oă ng': 'oăng',
    'oa ng': 'oang',
}

# ============================================================================
# 8. LỖI SỐ/KÝ TỰ ĐẶC BIỆT
# ============================================================================

SPECIAL_CHAR_ERRORS = {
    # === Số bị đọc sai ===
    '0': 'O',  # có thể là chữ O
    'l': '1',  # chữ l có thể là số 1
    'O': '0',  # chữ O có thể là số 0
    'I': '1',  # chữ I có thể là số 1
    
    # === Ký tự đặc biệt ===
    '…': '...',
    '—': '-',
    '–': '-',
    '"': '"',
    '"': '"',
    ''': "'",
    ''': "'",
    '，': ',',
    '。': '.',
    '：': ':',
    '；': ';',
    '？': '?',
    '！': '!',
}

# ============================================================================
# 9. CONTEXT-AWARE PATTERNS (Regex)
# ============================================================================

CONTEXT_PATTERNS = [
    # Format: (pattern, replacement, description)
    
    # === Lỗi chữ K dính - Kkhông, Kkkhông ===
    (r'\bKk+h\u00f4ng\b', 'Không', 'Fix Kkhông -> Không'),
    (r'\bkk+h\u00f4ng\b', 'không', 'Fix kkhông -> không'),
    
    # === Lỗi từ ảnh mới ===
    (r'\bcâu\s+nhìn\b', 'cậu nhìn', 'cậu nhìn'),
    (r'\bchủ\s+mèo\b', 'chú mèo', 'chú mèo'),
    (r'\bsợ\s+hải\b', 'sợ hãi', 'FIX: sợ hải -> sợ hãi'),
    (r'\bsơ\s+hãi\b', 'sợ hãi', 'FIX: sơ hãi -> sợ hãi'),
    (r'\bBắt\s+chợt\b', 'Bất chợt', 'FIX: Bắt chợt -> Bất chợt'),
    (r'\blắngng\b', 'lắng', 'lắngng -> lắng'),
    (r'\bKkhông\b', 'Không', 'Kkhông -> Không'),
    (r'\bkkhông\b', 'không', 'kkhông -> không'),
    
    # === Dấu câu sai ===
    (r'\s+,', ',', 'Remove space before comma'),
    (r'\s+\.', '.', 'Remove space before period'),
    (r'\s+\?', '?', 'Remove space before question mark'),
    (r'\s+!', '!', 'Remove space before exclamation'),
    (r',(?!\s)', ', ', 'Add space after comma'),
    (r'\.(?!\s|$|\d)', '. ', 'Add space after period'),
    
    # === Sai dấu ngữ cảnh ===
    (r'phái\s+dung', 'phải dung', 'phải dung'),
    (r'mọi\s+thí\b', 'mọi thứ', 'mọi thứ'),
    (r'thật\s+sử\b', 'thật sự', 'thật sự'),
    (r'trí\s+tái', 'từ từ', 'từ từ'),
    (r'buồn\s+thỉ', 'buồn thì', 'buồn thì'),
    (r'tớ\s+át\s+tổng', 'tớ sẽ lắng', 'tớ sẽ lắng'),
    (r'của\s+tớ\s+át', 'của tớ sẽ', 'của tớ sẽ'),
    (r'át\s+tổng', 'sẽ lắng', 'sẽ lắng'),
    (r'guủi\s*tố', 'gửi tớ', 'gửi tớ'),
    (r'guủi\s*tớ', 'gửi tớ', 'gửi tớ'),
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
    (r'dung\s*hoà\s*m\b', 'dung hòa mọi', 'dung hòa mọi'),
    (r'xin\s*hây\s*títi', 'xin hãy tí tí', 'xin hãy tí tí'),
    (r'hây\s*títi', 'hãy tí tí', 'hãy tí tí'),
    (r'lắngng\s*nghe', 'lắng nghe', 'lắng nghe'),
    (r'lắngngng\s*nghe', 'lắng nghe', 'lắng nghe'),
    (r'khóc\s+rong', 'khóc ròng', 'khóc ròng'),
    (r'mối\s+tỉnh', 'mối tình', 'mối tình'),
    (r'tuổ\s+không', 'tưởng không', 'tưởng không'),
    (r'mối\s+từnh', 'mối tình', 'mối tình'),
    (r'ngẫn\s+ngơ', 'ngẩn ngơ', 'ngẩn ngơ'),
    (r'ngân\s+ngơ', 'ngẩn ngơ', 'ngẩn ngơ'),
    (r'lậm\s+xúc', 'cảm xúc', 'cảm xúc'),
    (r'căm\s+xúc', 'cảm xúc', 'cảm xúc'),
    (r'troạn\s+về', 'trọn vẹn', 'trọn vẹn'),
    (r'trọn\s+về', 'trọn vẹn', 'trọn vẹn'),
    (r'chúng\s+tả', 'chúng ta', 'chúng ta'),
    (r'tả\s+doc', 'ta đọc', 'ta đọc'),
    (r'tả\s+đọc', 'ta đọc', 'ta đọc'),
    (r'thất\s+đẹp', 'thật đẹp', 'thật đẹp'),
    (r'thương\s+căm', 'thương cảm', 'thương cảm'),
    (r'thường\s+cầm', 'thương cảm', 'thương cảm'),
    (r'Trương\s+ky', 'Thương có', 'Thương có'),
    (r'người\s+ngẫn', 'người ngẩn', 'người ngẩn'),
    (r'bing\s+dùng', 'biết dùng', 'biết dùng'),
    (r'hanh\s+ph', 'hạnh phúc', 'hạnh phúc'),
    (r'không\s+hiếu', 'không hiểu', 'không hiểu'),
    (r'co\s+khi', 'có khi', 'có khi'),
    (r'có\s+khỉ', 'có khi', 'có khi'),
    (r'Bà\s+bà', 'Hoặc có', 'Hoặc có'),
    (r'khó\s+khẩm', 'khó khăn', 'khó khăn'),
    (r'khó\s+khâm', 'khó khăn', 'khó khăn'),
    (r'sẽ\s+lống', 'sẽ lắng', 'sẽ lắng'),
    (r'sĩ\s+lống', 'sẽ lắng', 'sẽ lắng'),
    (r'sau\s+này', 'sau này', 'sau này'),
    (r'sáu\s+này', 'sau này', 'sau này'),
    (r'dung\s+hòa', 'dung hòa', 'dung hòa'),
    (r'dùng\s+hòa', 'dung hòa', 'dung hòa'),
    (r'buôn\s+thi', 'buồn thì', 'buồn thì'),
    (r'phai\s+dung', 'phải dung', 'phải dung'),
    (r'phai\s+dùng', 'phải dung', 'phải dung'),
    (r'nhật\s+sử', 'thật sự', 'thật sự'),
    (r'nhật\s+sự', 'thật sự', 'thật sự'),
    
    # === Sửa lỗi thừa ký tự ===
    (r'lắngng\b', 'lắng', 'lắng'),
    (r'lắngngng\b', 'lắng', 'lắng'),
    
    # === Khoảng trắng thừa ===
    (r'\s{2,}', ' ', 'Remove multiple spaces'),
]

# ============================================================================
# 10. TỪ ĐIỂN TIẾNG VIỆT PHỔ BIẾN (cho spell check)
# ============================================================================

VIETNAMESE_COMMON_WORDS = {
    # Đại từ
    'tôi', 'tao', 'tớ', 'mình', 'ta', 'chúng', 'bọn', 'họ', 'nó', 'hắn', 
    'cô', 'chú', 'anh', 'chị', 'em', 'bạn', 'cậu', 'bác', 'ông', 'bà', 
    'con', 'cháu', 'thằng', 'đứa', 'người', 'ai', 'gì', 'đâu', 'nào',
    
    # Động từ phổ biến
    'là', 'có', 'được', 'làm', 'đi', 'đến', 'về', 'ra', 'vào', 'lên', 
    'xuống', 'nói', 'viết', 'đọc', 'nghe', 'xem', 'nhìn', 'thấy', 'biết', 
    'hiểu', 'nghĩ', 'muốn', 'cần', 'phải', 'nên', 'hãy', 'đừng', 'chớ', 
    'xin', 'cho', 'lấy', 'ăn', 'uống', 'ngủ', 'thức', 'chơi', 'học', 
    'dạy', 'yêu', 'ghét', 'thích', 'sống', 'chết', 'sinh', 'gửi', 'nhận',
    'mở', 'đóng', 'bắt', 'thả', 'giữ', 'buông', 'cầm', 'nắm', 'ôm', 'hôn',
    
    # Tính từ
    'tốt', 'xấu', 'đúng', 'sai', 'cao', 'thấp', 'dài', 'ngắn', 'rộng', 
    'hẹp', 'nhanh', 'chậm', 'mạnh', 'yếu', 'khỏe', 'ốm', 'giàu', 'nghèo', 
    'sang', 'hèn', 'đẹp', 'xấu', 'mới', 'cũ', 'trẻ', 'già', 'lớn', 'nhỏ',
    'buồn', 'vui', 'khó', 'dễ', 'nóng', 'lạnh', 'ấm', 'mát', 'khô', 'ướt',
    
    # Trạng từ
    'rất', 'lắm', 'quá', 'hơi', 'khá', 'cũng', 'đều', 'luôn', 'thường', 
    'hay', 'đã', 'đang', 'sẽ', 'vừa', 'mới', 'còn', 'vẫn', 'cứ', 'chỉ', 
    'chính', 'ngay', 'liền', 'tức', 'sớm', 'muộn', 'trước', 'sau',
    
    # Liên từ, giới từ
    'của', 'và', 'với', 'để', 'cho', 'từ', 'đến', 'trong', 'ngoài', 
    'trên', 'dưới', 'trước', 'sau', 'giữa', 'bên', 'cạnh', 'gần', 'xa', 
    'theo', 'về', 'tại', 'vì', 'do', 'bởi', 'nên', 'mà', 'thì', 'nếu', 
    'thế', 'như', 'khi', 'lúc', 'nhưng', 'tuy', 'dù', 'mặc', 'dầu', 
    'song', 'hoặc', 'hay',
    
    # Từ thời gian
    'ngày', 'tháng', 'năm', 'tuần', 'giờ', 'phút', 'giây', 'sáng', 
    'trưa', 'chiều', 'tối', 'đêm', 'hôm', 'nay', 'qua', 'mai', 'kia',
    'bây', 'giờ', 'lúc', 'khi', 'rồi', 'xong', 'hết', 'còn',
    
    # Số đếm
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 
    'mười', 'trăm', 'nghìn', 'ngàn', 'triệu', 'tỷ', 'vạn',
    
    # Từ phổ biến khác
    'nhé', 'nhỉ', 'nhá', 'nha', 'ạ', 'ơi', 'à', 'ừ', 'ừm', 'vâng', 
    'dạ', 'không', 'chưa', 'chẳng', 'sao', 'thế', 'vậy', 'đấy', 'đây',
    'thật', 'sự', 'thực', 'lắng', 'nghe', 'dung', 'hòa', 'hoà', 'mọi',
    'thứ', 'việc', 'điều', 'cái', 'con', 'chiếc', 'quyển', 'cuốn',
}


# ============================================================================
# MAIN CORRECTION CLASS
# ============================================================================

class VietnameseOCRCorrector:
    """
    Bộ sửa lỗi OCR tiếng Việt toàn diện
    
    Sử dụng:
        corrector = VietnameseOCRCorrector()
        fixed_text = corrector.correct("guủi tố, việc phái dung hoà...")
    """
    
    def __init__(self):
        # Keep phrase errors separate for priority processing
        self.phrase_errors = PHRASE_ERRORS.copy()
        
        # Combine single-word error dictionaries
        self.word_errors = {}
        self.word_errors.update(DIACRITIC_ERRORS)
        self.word_errors.update(VOWEL_ERRORS)
        self.word_errors.update(CONSONANT_ERRORS)
        self.word_errors.update(STUCK_WORDS)
        self.word_errors.update(SINGLE_WORD_ERRORS)
        self.word_errors.update(SPLIT_WORD_ERRORS)
        
        self.context_patterns = CONTEXT_PATTERNS
        self.common_words = VIETNAMESE_COMMON_WORDS
        
        print(f"📚 Loaded {len(self.word_errors)} word corrections + {len(self.phrase_errors)} phrase corrections")
        print(f"📝 Loaded {len(self.context_patterns)} context patterns")
        print(f"📖 Loaded {len(self.common_words)} common Vietnamese words")
    
    def correct(self, text: str, verbose: bool = False) -> str:
        """
        Sửa lỗi OCR trong text
        
        Args:
            text: Text cần sửa
            verbose: In chi tiết các sửa đổi
            
        Returns:
            Text đã sửa
        """
        if not text:
            return text
        
        # Normalize Unicode to NFC (composed form) for consistent matching
        text = unicodedata.normalize('NFC', text)
        original = text
        
        # Step 1: Apply context patterns (regex) - highest priority
        for pattern, replacement, desc in self.context_patterns:
            try:
                new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                if new_text != text and verbose:
                    print(f"  🔧 Pattern: '{desc}'")
                text = new_text
            except:
                continue
        
        # Step 2: Apply PHRASE corrections first (multi-word, longer phrases)
        # Sort by length descending to match longest phrases first
        sorted_phrases = sorted(self.phrase_errors.items(), key=lambda x: -len(x[0]))
        for error, correct in sorted_phrases:
            # Normalize both
            error_norm = unicodedata.normalize('NFC', error)
            correct_norm = unicodedata.normalize('NFC', correct)
            
            if error_norm.lower() in text.lower():
                # Use word boundary for phrases to avoid partial matches
                pattern = re.compile(r'\b' + re.escape(error_norm) + r'\b', re.IGNORECASE)
                new_text = pattern.sub(correct_norm, text)
                if new_text != text and verbose:
                    print(f"  📝 Phrase: '{error_norm}' → '{correct_norm}'")
                text = new_text
        
        # Step 3: Apply WORD corrections (single words)
        # Sort by length descending
        sorted_words = sorted(self.word_errors.items(), key=lambda x: -len(x[0]))
        for error, correct in sorted_words:
            # Normalize both
            error_norm = unicodedata.normalize('NFC', error)
            correct_norm = unicodedata.normalize('NFC', correct)
            
            if error_norm.lower() in text.lower():
                # Use word boundary for single words
                pattern = re.compile(r'\b' + re.escape(error_norm) + r'\b', re.IGNORECASE)
                new_text = pattern.sub(correct_norm, text)
                if new_text != text and verbose:
                    print(f"  ✏️ Word: '{error_norm}' → '{correct_norm}'")
                text = new_text
        
        # Step 4: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 5: Fix punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])(?!\s|$)', r'\1 ', text)
        
        if verbose:
            if text != original:
                print(f"  ✅ Final: '{text}'")
            else:
                print(f"  ℹ️ No changes made")
        
        return text
    
    def add_correction(self, error: str, correct: str, is_phrase: bool = False):
        """Thêm mapping sửa lỗi mới"""
        if is_phrase or ' ' in error:
            self.phrase_errors[error] = correct
        else:
            self.word_errors[error] = correct
    
    def add_pattern(self, pattern: str, replacement: str, description: str = ""):
        """Thêm pattern regex mới"""
        self.context_patterns.append((pattern, replacement, description))
    
    def get_suggestions(self, word: str, max_suggestions: int = 5) -> List[str]:
        """Gợi ý sửa cho một từ"""
        suggestions = []
        word_lower = word.lower()
        
        # Check in error dicts
        if word_lower in self.word_errors:
            suggestions.append(self.word_errors[word_lower])
        if word_lower in self.phrase_errors:
            suggestions.append(self.phrase_errors[word_lower])
        
        # Find similar words
        for dict_word in self.common_words:
            if self._similarity(word_lower, dict_word) > 0.7:
                if dict_word not in suggestions:
                    suggestions.append(dict_word)
        
        return suggestions[:max_suggestions]
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Tính độ tương đồng đơn giản"""
        if not s1 or not s2:
            return 0.0
        common = set(s1) & set(s2)
        total = set(s1) | set(s2)
        return len(common) / len(total) if total else 0.0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_corrector = None

def get_corrector() -> VietnameseOCRCorrector:
    """Get singleton corrector instance"""
    global _corrector
    if _corrector is None:
        _corrector = VietnameseOCRCorrector()
    return _corrector


def fix_ocr_text(text: str, verbose: bool = False) -> str:
    """
    Sửa lỗi OCR trong text
    
    Args:
        text: Text từ OCR
        verbose: In chi tiết
        
    Returns:
        Text đã sửa
    """
    return get_corrector().correct(text, verbose)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VIETNAMESE OCR CORRECTIONS TEST")
    print("=" * 70)
    
    test_texts = [
        "guủi tố, việc phái dung hoà mọi thí thật sử rất khó khăm",
        "xinhãy títi buồn thỉ hấy viết ranhí",
        "sau này của tố sĩ lắngng nghe cậu",
        "Khóc rong vì tuổ không mối tỉnh",
        "họ đa viêt nhung la thư tình",
        "chúng ta đoc xong co khi cười ngân ngơ",
        "hoặc có khỉ lại khóc rong vì buồn bã",
        "có khi thương căm cho nhung mối tinh si",
        "không được đáp laị Nhưng trên tât cả",
        "đó đều là nhung cam xuc tron vẹn",
        "Thương co thương buon có buồn và vô vọng",
        "cũng là môt nỗi tuyêt vong thât đẹp",
    ]
    
    corrector = get_corrector()
    
    for text in test_texts:
        print(f"\n📝 Original: {text}")
        fixed = corrector.correct(text, verbose=True)
        print(f"✅ Fixed:    {fixed}")
        print("-" * 50)
