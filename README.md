# 🇻🇳 Vietnamese Handwriting & OCR Recognition System

**Hệ thống nhận dạng chữ viết tay & chữ in tiếng Việt tiên tiến**

Ứng dụng web Streamlit tích hợp 7 engine OCR mạnh nhất, hỗ trợ nhận dạng chữ viết tay, chữ in, PDF, ảnh và camera.

---

## ✨ Tính Năng

### 📤 Upload Ảnh

- Nhận dạng chữ viết tay & chữ in từ ảnh
- Hỗ trợ nhiều format: PNG, JPG, JPEG, BMP
- Xem kết quả thực tế + lưu độ tin cậy

### 📄 Xử Lý PDF

- Extract text trực tiếp từ PDF có text layer
- OCR scanned PDF (ảnh) với Tesseract + VietOCR
- Chuyển PDF → Word (.docx) giữ layout

### 📸 Camera

- Chụp ảnh trực tiếp qua webcam
- Nhận dạng real-time

### 📋 Batch Processing

- Xử lý hàng loạt ảnh / PDF
- Xuất kết quả CSV

### 🎯 Voting System

- Kết hợp 7 engine khác nhau
- Voting: Majority, Weighted, Unanimous
- Độ chính xác cao hơn 95%

---

## 🚀 Cài Đặt

### 1️⃣ Yêu Cầu Hệ Thống

- **Python 3.8+**
- **RAM: 4GB+** (8GB khuyến nghị)
- **Disk: 5GB+** (cho models + data)
- **Windows / Linux / macOS**

### 2️⃣ Clone Repository

```bash
git clone https://github.com/phuduong24h/do-an-tn.git
cd handwriting-Project
```

### 3️⃣ Tạo Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ Cài Đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5️⃣ Cài Tesseract (Bắt Buộc cho PDF)

**Windows:**

```bash
# Download từ: https://github.com/UB-Mannheim/tesseract/wiki
# Chạy .exe installer, chọn Vietnamese language, install vào C:\Program Files\Tesseract-OCR

# Cập nhật path trong app (nếu cần)
# Mở app_advanced_vietnamese.py, tìm:
# pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**Linux:**

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-vie
```

**macOS:**

```bash
brew install tesseract
```

---

## 🎯 Chạy Ứng Dụng

### Start Web App

```bash
# Đảm bảo venv được activate
streamlit run app_advanced_vietnamese.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

### First Time Setup (Tùy Chọn)

```bash
# Auto-check dependencies & models
python setup.py
```

---

## 🎓 Huấn Luyện Lại Mô Hình (CRNN)

### Chuẩn Bị Dữ Liệu

#### Option 1: Dùng VNOnDB Dataset

```bash
# 1. Download từ: http://on.cs.keio.ac.jp/vnondb/
# 2. Giải nén vào:
#    - data/VNOnDB_Line/InkData_line/
#    - data/VNOnDB_Paragraph/InkData_paragraph/

# 3. Convert InkML → PNG + Labels
python src/train_crnn.py --prepare-data
```

#### Option 2: Dùng Synthetic Data (Có sẵn)

```bash
# Synthetic data đã có tại: data/synthetic_100k/
# Chứa ~100k ảnh chữ viết tiếng Việt được tạo tự động
```

#### Option 3: Custom Dataset

```bash
# Cấu trúc folder:
# data/custom/
# ├── images/
# │   ├── 0001.png
# │   ├── 0002.png
# │   └── ...
# └── labels.txt  # 0001.png chữ viết
```

### Huấn Luyện CRNN

```bash
# Basic training
python train_complete_vietnamese.py

# Advanced training options
python train_complete_vietnamese.py \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --dataset synthetic_100k \
    --output-dir ./checkpoints/

# Với GPU
python train_complete_vietnamese.py --gpu 0 --epochs 100
```

### Output

```
checkpoints/
├── crnn_epoch_10.pth
├── crnn_epoch_20.pth
├── crnn_best.pth          # Mô hình tốt nhất
└── training_log.txt
```

### Sử Dụng Mô Hình Trained

```python
# Trong app, thay đường dẫn model:
# src/models.py -> CRNN_MODEL_PATH = './checkpoints/crnn_best.pth'
```

---

## 📊 Cấu Trúc Project

```
handwriting-Project/
├── app_advanced_vietnamese.py    # ⭐ Main Streamlit app
├── requirements.txt              # ⭐ Dependencies
├── setup.py                      # First-time setup
├── train_complete_vietnamese.py  # ⭐ Train CRNN
│
├── src/                          # ⭐ Core modules
│   ├── models.py                 # CRNN, model loader
│   ├── ocr_service.py            # OCR inference
│   ├── vietnamese_ocr_advanced.py # Main OCR pipeline
│   ├── pdf_utils.py              # PDF processing
│   ├── text_detector.py          # Text detection
│   ├── postprocessor_advanced.py # Text correction
│   ├── language_model.py         # Language model
│   ├── beam_search.py            # Beam search decoding
│   └── ui/                       # UI components
│       ├── styles.py
│       ├── sidebar.py
│       └── tabs/
│           ├── image_ocr.py
│           ├── pdf_ocr.py
│           ├── camera_ocr.py
│           └── batch_ocr.py
│
├── data/                         # ⭐ Training data
│   ├── synthetic_100k/           # 100k synthetic images
│   ├── VNOnDB_Line/              # Real VNOnDB data (download)
│   └── VNOnDB_Paragraph/         # VNOnDB paragraph data
│
├── models/                       # Trained models
│   ├── crnn_best.pth             # ⭐ Best CRNN model
│   ├── crnn_best_compat.pth      # Compatible version
│   └── trocr_vn_quick/           # TrOCR models (auto-download)
│
└── config.yaml                   # Configuration
```

---

## 🔧 Configuration

Sửa `config.yaml` để customize:

```yaml
# OCR Settings
ocr:
  default_engines: ["VietOCR", "PaddleOCR", "Tesseract"]
  voting_method: "majority"
  confidence_threshold: 0.5

# Model Paths
models:
  crnn_path: "./crnn_best.pth"
  language_model: "./models/language_model.pkl"

# PDF Settings
pdf:
  dpi: 300
  use_direct_extraction: true

# API Keys (nếu dùng external services)
api:
  # None currently
```

---

## 📖 Hướng Dẫn Chi Tiết

### 🎯 Use Case 1: Nhận Dạng Ảnh Chữ Viết Tay

1. Mở app: `streamlit run app_advanced_vietnamese.py`
2. Tab "📤 Upload Image"
3. Upload ảnh chữ viết tay
4. Sidebar chọn engines (VietOCR + PaddleOCR khuyến nghị)
5. Click "Recognize Text"
6. Xem kết quả & tin cậy độ

### 🎯 Use Case 2: Xử Lý PDF Scanned

1. Tab "📄 PDF Processing"
2. Upload PDF
3. Chọn DPI (300-400 cho chất lượng tốt)
4. Bật Tesseract + VietOCR
5. Click "Process PDF"
6. Xem text / Xuất Word

### 🎯 Use Case 3: Huấn Luyện Lại Với Data Của Bạn

1. Chuẩn bị data: `data/custom/images/` + `labels.txt`
2. Chạy: `python train_complete_vietnamese.py --dataset custom`
3. Chờ training hoàn tất
4. Update path model trong `src/models.py`
5. Restart app để dùng mô hình mới

---

## 🧪 Testing

### Quick Test

```bash
# Test OCR engines
python -c "
from src.ocr_service import OCRService
service = OCRService()
result = service.recognize('test_image.png', ['VietOCR', 'PaddleOCR'])
print(result)
"
```

### Full Test Suite

```bash
# (Nếu có)
pytest tests/ -v
```

---

## 🐛 Troubleshooting

### Problem: "Tesseract not found"

```
✅ Solution:
1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Install vào C:\Program Files\Tesseract-OCR
3. Update path trong code hoặc environment variable
```

### Problem: "CUDA out of memory"

```
✅ Solution:
1. Giảm batch size: --batch-size 16
2. Hoặc dùng CPU: --no-cuda
3. Reduce image resolution
```

### Problem: "Low OCR accuracy"

```
✅ Solution:
1. Improve image quality: DPI 300+
2. Use multiple engines & voting
3. Train CRNN với custom data
4. Check language model
```

### Problem: "Slow performance"

```
✅ Solution:
1. Use GPU: --gpu 0
2. Disable unnecessary engines
3. Reduce DPI cho PDF
4. Use TrOCR_quick thay TrOCR_full
```

---

## 📚 Tài Liệu

- **[VietOCR](https://github.com/pbcquoc/vietocr)** - Transformer OCR for Vietnamese
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** - Chinese-optimized OCR
- **[TrOCR](https://github.com/microsoft/unilm/tree/master/trocr)** - Microsoft Transformer OCR
- **[Tesseract](https://tesseract-ocr.github.io/)** - Open-source OCR engine
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** - Easy-to-use OCR

---

## 🤝 Contributing

Hoan nghênh issues & pull requests!

```bash
# Fork repo
git clone https://github.com/YOUR_FORK/do-an-tn.git

# Create feature branch
git checkout -b feature/your-feature

# Commit & Push
git add .
git commit -m "Add: your feature"
git push origin feature/your-feature

# Open PR
```

---

## 📄 License

MIT License - See LICENSE file

---

## 👨‍💼 Author

**Phạm Hu Dương** - phuduong24h@gmail.com

---

## 🙏 Acknowledgments

- VNOnDB Dataset: Keio University
- VietOCR: pbcquoc
- PaddleOCR: Baidu
- TrOCR: Microsoft

---

## 💬 Support

- 📧 Email: phuduong24h@gmail.com
- 🐛 Issues: GitHub Issues
- 💡 Discussions: GitHub Discussions

---

**Made with ❤️ for Vietnamese OCR**
