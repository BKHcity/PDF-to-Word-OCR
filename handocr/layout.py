"""Simple layout and table extraction helpers.

This is a heuristic table extractor that uses OpenCV morphological
operations to detect table grid lines and extract cell bounding boxes.
It is intentionally simple and designed to work on reasonably clean
scanned tables. For complex tables consider integrating a dedicated
table detection model.
"""
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2


def image_to_cv2_gray(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("L"))
    return arr


def extract_tables(pil_img: Image.Image) -> List[List[List[Tuple[int,int,int,int]]]]:
    """Return a list of tables; each table is a list of rows, each row a list of cell bboxes (x1,y1,x2,y2).

    This function returns bounding boxes only. OCR per-cell can be applied afterwards.
    """
    gray = image_to_cv2_gray(pil_img)
    h, w = gray.shape

    # binarize
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # detect horizontal and vertical lines
    horizontal = bw.copy()
    vertical = bw.copy()

    scale = max(10, w // 50)
    horiz_size = scale
    vert_size = max(2, h // 50)

    horiz_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
    horizontal = cv2.erode(horizontal, horiz_structure)
    horizontal = cv2.dilate(horizontal, horiz_structure)

    vert_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
    vertical = cv2.erode(vertical, vert_structure)
    vertical = cv2.dilate(vertical, vert_structure)

    # combine
    grid = cv2.add(horizontal, vertical)

    # find contours of grid areas
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww < 30 or hh < 30:
            continue
        roi = bw[y:y+hh, x:x+ww]

        # find vertical lines in roi to get column boundaries
        cols = cv2.reduce(roi, 0, cv2.REDUCE_AVG).flatten()
        col_seps = np.where(cols > 10)[0]

        # find horizontal lines to get row boundaries
        rows = cv2.reduce(roi, 1, cv2.REDUCE_AVG).flatten()
        row_seps = np.where(rows > 10)[0]

        # cluster separators to boundaries
        def cluster_idxs(idxs):
            if len(idxs) == 0:
                return []
            groups = [[idxs[0]]]
            for i in idxs[1:]:
                if i - groups[-1][-1] <= 3:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            return [int(sum(g)/len(g)) for g in groups]

        col_bound = cluster_idxs(col_seps)
        row_bound = cluster_idxs(row_seps)

        # build cells list
        if not col_bound or not row_bound:
            continue

        # include edges
        xs = [0] + col_bound + [ww-1]
        ys = [0] + row_bound + [hh-1]

        table = []
        for r in range(len(ys)-1):
            row_cells = []
            for c in range(len(xs)-1):
                x1 = x + xs[c]
                y1 = y + ys[r]
                x2 = x + xs[c+1]
                y2 = y + ys[r+1]
                # skip tiny
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                row_cells.append((x1, y1, x2, y2))
            if row_cells:
                table.append(row_cells)

        if table:
            tables.append(table)

    return tables


def export_table_csv(pil_img: Image.Image, ocr_service, out_path: str):
    """Detect tables and OCR each cell, writing CSV files (one per detected table).

    `ocr_service` should expose `recognize_image` or `recognize_image_fused` that accepts PIL.Image.
    """
    import csv
    tables = extract_tables(pil_img)
    for ti, table in enumerate(tables):
        rows_text = []
        for row in table:
            row_text = []
            for bbox in row:
                x1, y1, x2, y2 = bbox
                cell = pil_img.crop((x1, y1, x2, y2))
                res = ocr_service.recognize_image_fused(cell)
                row_text.append(res.text.replace('\n', ' ').strip())
            rows_text.append(row_text)

        csv_out = f"{out_path.rstrip('.csv')}_table{ti}.csv"
        with open(csv_out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            for r in rows_text:
                w.writerow(r)

    return len(tables)
