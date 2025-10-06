import sys
import json
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import pytesseract

import os
from datetime import datetime

def writefile(output_path: str, image_data):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, image_data)
        if success:
            print(f"✅ Cleaned image saved to {output_path}")
        else:
            print(f"⚠️ Failed to write image file: {output_path}")
    except Exception as e:
        print(f"⚠️ Could not save cleaned image ({type(e).__name__}): {e}")

def _read_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    return image, w, h


def _preprocess(image: np.ndarray, target_width: int = 1200) -> tuple[np.ndarray, float]:
    """Preprocess image to produce a clean binary mask and scale factor."""
    h, w = image.shape[:2]

    # Scale image if it's too large
    scale = 1.0
    if w > target_width:
        scale = target_width / float(w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (normal + inverted)
    th1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    th2 = cv2.adaptiveThreshold(
        255 - gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    merged = cv2.bitwise_or(th1, th2)

    # Clean up small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned, scale

def is_index_candidate(box, img_shape):
    """
    Heuristic check for boxes that look like move indices.
    Works for any image size by using relative proportions.
    """
    x, y, w, h = box
    img_h, img_w = img_shape[:2]
    aspect = w / float(h)

    # Relative to image height (makes it scale-invariant)
    rel_h = h / float(img_h)
    rel_w = w / float(img_w)

    # Typically indices are small and tall
    return (0.2 < aspect < 1.0) and (0.015 < rel_h < 0.08) and (0.005 < rel_w < 0.05)


def process_chess_layout(image_path: str):
    """Detect chess move columns, rows, and potential index regions."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    binary, scale = _preprocess(image)
    img_h, img_w = binary.shape
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # === Step 1: Detect columns ===
    vertical_sum = np.sum(binary // 255, axis=0)
    vertical_sum = cv2.GaussianBlur(vertical_sum.astype(np.float32), (51, 1), 0)
    vertical_thresh = np.max(vertical_sum) * 0.3

    columns = []
    in_col = False
    start_x = 0
    for x, val in enumerate(vertical_sum):
        if val > vertical_thresh and not in_col:
            start_x = x
            in_col = True
        elif val <= vertical_thresh and in_col:
            columns.append((start_x, x))
            in_col = False
    if in_col:
        columns.append((start_x, img_w))

    # === Step 2: Detect rows *within* each column ===
    for (x1, x2) in columns:
        col_img = binary[:, x1:x2]
        horizontal_sum = np.sum(col_img // 255, axis=1)
        horizontal_sum = cv2.GaussianBlur(horizontal_sum.astype(np.float32), (1, 31), 0)
        horizontal_thresh = np.max(horizontal_sum) * 0.3

        in_row = False
        start_y = 0
        for y, val in enumerate(horizontal_sum):
            if val > horizontal_thresh and not in_row:
                start_y = y
                in_row = True
            elif val <= horizontal_thresh and in_row:
                y1, y2 = start_y, y
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
                in_row = False

        # === Step 3: Find small index-like contours within this column ===
        contours, _ = cv2.findContours(col_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if is_index_candidate((bx, by, bw, bh), col_img.shape):
                print('might bhe an index')
                # Offset x by column start
                cv2.rectangle(vis, (x1 + bx, by), (x1 + bx + bw, by + bh), (0, 255, 0), 1)
            else:
                print('no way')

    # === Save visualization ===
    timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    output_path = os.path.join("img", "generated", f"layoutLINES_{timestamp}.png")
    writefile(output_path, vis)
    print(f"Saved: {output_path}")

    # === Save visualization ===


def _ocr_words(image: np.ndarray) -> List[Dict[str, Any]]:
    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789.ABCDEFGHabcdefghNBRQKOx+-=#!?()[]{}.,;:"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i]) if data["conf"][i] not in ("", "-1") else -1.0
        except Exception:
            conf = -1.0
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        words.append(
            {
                "text": text,
                "conf": conf,
                "left": x,
                "top": y,
                "width": w,
                "height": h,
                "line_num": int(data.get("line_num", [0])[i] or 0),
                "block_num": int(data.get("block_num", [0])[i] or 0),
                "par_num": int(data.get("par_num", [0])[i] or 0),
                "word_num": int(data.get("word_num", [0])[i] or 0),
            }
        )

    # Sort by block->par->line->x
    words.sort(key=lambda w: (w["block_num"], w["par_num"], w["line_num"], w["left"]))
    return words


def _is_move_index_token(t: str) -> bool:
    if not t:
        return False
    if t.endswith(".") and t[:-1].isdigit():
        return True
    if t.isdigit():
        return True
    return False


def _normalize(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[float, float]:
    cx = x + w / 2.0
    cy = y + h / 2.0
    px = max(0.0, min(100.0, 100.0 * cx / img_w))
    py = max(0.0, min(100.0, 100.0 * cy / img_h))
    return px, py


def _segment_lines(binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Horizontal projection to find text lines
    rowsum = np.sum(binary_image > 0, axis=1)
    thresh = max(1, int(0.02 * np.max(rowsum)))
    in_run = False
    start = 0
    regions: List[Tuple[int, int, int, int]] = []
    height, width = binary_image.shape[:2]
    for y, val in enumerate(rowsum):
        if val >= thresh and not in_run:
            in_run = True
            start = y
        elif val < thresh and in_run:
            in_run = False
            end = y
            regions.append((0, start, width, max(1, end - start)))
    if in_run:
        regions.append((0, start, width, max(1, height - start)))
    return regions


def process_chess_image(image_path: str) -> List[Dict[str, Any]]:
    image, img_w, img_h = _read_image(image_path)

    # 1) Segmentation (ink-on-paper regions)
    binary = _preprocess(image)

    # Optional: use detected line regions to bias parsing (available if needed)
    _ = _segment_lines(binary)

    # 2) OCR and sequential move extraction
    words = _ocr_words(image)

    # Build index: line -> list of words (left-to-right)
    from collections import defaultdict
    line_to_words: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for w in words:
        key = (w["block_num"], w["par_num"], w["line_num"])  # stable grouping
        line_to_words[key].append(w)
    for key in line_to_words:
        line_to_words[key].sort(key=lambda w: w["left"])  # left-to-right within the line

    moves: List[Dict[str, Any]] = []
    seen_indices = set()

    # Parse each line for patterns: <num.> <white> <black>
    for key in sorted(line_to_words.keys()):
        tokens = line_to_words[key]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            t = tok["text"]
            if _is_move_index_token(t):
                # Normalize token (strip trailing dot for index number)
                num_str = t[:-1] if t.endswith(".") else t
                try:
                    idx_num = int(num_str)
                except Exception:
                    i += 1
                    continue

                # Find white and black tokens immediately following, skipping separators
                def next_token(j: int) -> int:
                    k = j + 1
                    while k < len(tokens):
                        tt = tokens[k]["text"]
                        if tt and tt not in {".", ",", ";", ":", "-", "—"} and not _is_move_index_token(tt):
                            return k
                        k += 1
                    return -1

                w_idx = next_token(i)
                b_idx = next_token(w_idx) if w_idx != -1 else -1

                if idx_num not in seen_indices and w_idx != -1 and b_idx != -1:
                    seen_indices.add(idx_num)
                    w_tok = tokens[w_idx]
                    b_tok = tokens[b_idx]

                    wxp, wyp = _normalize(w_tok["left"], w_tok["top"], w_tok["width"], w_tok["height"], img_w, img_h)
                    bxp, byp = _normalize(b_tok["left"], b_tok["top"], b_tok["width"], b_tok["height"], img_w, img_h)

                    moves.append(
                        {
                            "moveIndex": idx_num,
                            "white": {
                                "location": {"x": round(wxp, 2), "y": round(wyp, 2)},
                                "move": w_tok["text"],
                            },
                            "black": {
                                "location": {"x": round(bxp, 2), "y": round(byp, 2)},
                                "move": b_tok["text"],
                            },
                        }
                    )

                    # Advance cursor beyond black token to avoid duplicate grouping on same line
                    i = b_idx + 1
                    continue

            i += 1

    # Sort by move index to enforce sequential expectation
    moves.sort(key=lambda m: m["moveIndex"]) 
    return moves


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python chess_segmenter.py <image_path>")
    #     sys.exit(1)
    good_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/1759549223658-IMG_293F39624621-1.jpeg'
    bad_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/IMG_2F54EA6661FD-1.jpeg'

    # img_path = good_img
    img_path = bad_img
    result = process_chess_layout(img_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()




