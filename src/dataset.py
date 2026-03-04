# src/dataset.py
import os, random, string
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from src.utils import resize_keep_aspect, pad_width, normalize_for_model, elastic_transform

# VIETNAMESE CHARSET - FULL SUPPORT (105 characters)
VN_CHARS = "aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789 "
VOCAB = ["<BLANK>"] + list(VN_CHARS)

# Maps for encoding/decoding
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
IDX2CHAR = {i: c for i, c in enumerate(VOCAB)}

def random_text(min_len=3, max_len=12, vietnamese=True):
    """Generate random text for training"""
    l = random.randint(min_len, max_len)
    return "".join(random.choice(VN_CHARS) for _ in range(l))

def random_vietnamese_text(min_len=3, max_len=15):
    """Generate random Vietnamese text with diacritics"""
    return random_text(min_len, max_len, vietnamese=True)

def random_english_text(min_len=3, max_len=12):
    """Generate random English text"""
    return random_text(min_len, max_len, vietnamese=False)

def load_system_fonts():
    fonts = []
    if os.name == 'nt':
        p = Path("C:/Windows/Fonts")
        if p.exists():
            fonts = list(p.glob("*.ttf"))
    else:
        fonts = list(Path("/usr/share/fonts").rglob("*.ttf"))
        fonts += list(Path("/Library/Fonts").rglob("*.ttf"))
    return [str(x) for x in fonts]

SYSTEM_FONTS = load_system_fonts()

# ===================================================================
# 1. Synthetic Dataset
# ===================================================================
class SyntheticTextDataset(Dataset):
    def __init__(self, n=2000, img_h=32, max_w=512, fonts=SYSTEM_FONTS, augment=False, vietnamese=True):
        self.n = n
        self.img_h = img_h
        self.max_w = max_w
        self.fonts = fonts if fonts else [None]
        self.augment = augment
        self.vietnamese = vietnamese
        self.samples = [(self._generate_text(), random.choice(self.fonts)) for _ in range(n)]

    def _generate_text(self):
        return random_vietnamese_text()

    def __len__(self): return self.n

    def render(self, txt, font_path):
        font_size = int(self.img_h * random.uniform(0.8, 2.2))
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        dummy = Image.new("L", (10,10), 255)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), txt, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        w = min(w, self.max_w - 4)
        img = Image.new("L", (w+4, max(self.img_h, h+4)), 255)
        draw = ImageDraw.Draw(img)
        y = max(0, (img.size[1] - h)//2)
        draw.text((2, y), txt, fill=0, font=font)
        arr = np.array(img)
        angle = random.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((arr.shape[1]//2, arr.shape[0]//2), angle, 1)
        arr = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), borderValue=255)
        return arr

    def __getitem__(self, idx):
        txt, font = self.samples[idx]
        arr = self.render(txt, font)
        if self.augment:
            if random.random() < 0.3:
                arr = elastic_transform(arr, alpha=30, sigma=4)
            if random.random() < 0.2:
                arr = cv2.GaussianBlur(arr, (3,3), 0)
            if random.random() < 0.2:
                noise = np.random.normal(0, 10, arr.shape).astype(np.int16)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        arr = resize_keep_aspect(arr, self.img_h, self.max_w)
        w = arr.shape[1]
        arr_norm = normalize_for_model(arr)
        arr_padded = pad_width((255 - (arr_norm*255)).astype(np.uint8), self.max_w)
        arr_norm = (255 - arr_padded).astype(np.float32) / 255.0

        tensor = torch.from_numpy(arr_norm).unsqueeze(0)
        label = torch.LongTensor([CHAR2IDX.get(c, 0) for c in txt])
        return tensor, label, w, len(txt)

# ===================================================================
# 2. VNOnDB Dataset
# ===================================================================
class VNOnDBDataset(Dataset):
    def __init__(self, root, label_file=None, img_h=32, max_w=512):
        self.root = Path(root)
        self.img_h, self.max_w = img_h, max_w
        self.samples = []

        if label_file is None:
            label_file = self.root / "labels.txt"
        else:
            label_file = Path(label_file)

        if not label_file.exists():
            pass # Fail silently if not found for now, or raise

        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "\t" not in line:
                        continue
                    fn, txt = line.strip().split("\t", 1)
                    self.samples.append((str(self.root / "images" / fn), txt))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fn, txt = self.samples[idx]
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_h, self.max_w), dtype=np.uint8)
            
        img = resize_keep_aspect(img, self.img_h, self.max_w)
        w = img.shape[1]
        arr_norm = (255 - img).astype(np.float32)/255.0
        tensor = torch.from_numpy(arr_norm).unsqueeze(0)

        # Filter text
        filtered_txt = "".join([c for c in txt.lower() if c in CHAR2IDX])
        label = torch.LongTensor([CHAR2IDX.get(c, 0) for c in filtered_txt])

        return tensor, label, w, len(label)
