import sys
import json
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import pytesseract
from sklearn.cluster import DBSCAN

import os
from datetime import datetime
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig

def writefile(output_path: str, image_data):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, image_data)
        if success:
            return output_path
            print(f"✅ Cleaned image saved to {output_path}")
        else:
            return ''
            print(f"⚠️ Failed to write image file: {output_path}")
    except Exception as e:
        print(f"⚠️ Could not save cleaned image ({type(e).__name__}): {e}")
        return ''

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
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 20
    )
    th2 = cv2.adaptiveThreshold(
        255 - gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 20
    )
    merged = cv2.bitwise_or(th1, th2)

    # Clean up small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel, iterations=1)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.bitwise_not(cleaned)

    timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    output_path = os.path.join("img", "generated", f"preprocess_{timestamp}.png")
    writefile(output_path, cleaned)
    return cleaned, scale, output_path

def merge_close_boxes(words, margin_ratio=0.02, y_thresh=5, x_gap=1):
    """
    Merge nearby word bounding boxes and expand them slightly.
    - margin_ratio: how much to expand the final boxes (e.g., 0.02 = 2%)
    - y_thresh: how close in Y boxes must be to belong to the same line
    - x_gap: max horizontal gap to merge into same group
    """
    # Extract all bounding boxes
    boxes = np.array([[w.aabb.xmin, w.aabb.ymin, w.aabb.xmax, w.aabb.ymax] for line in words for w in line])
    if len(boxes) == 0:
        return []

    # Sort boxes top-to-bottom, left-to-right
    boxes = boxes[np.lexsort((boxes[:,0], boxes[:,1]))]

    merged = []
    current = boxes[0].copy()

    for b in boxes[1:]:
        same_line = abs(b[1] - current[1]) < y_thresh
        close_x = b[0] - current[2] < x_gap
        if same_line and close_x:
            # merge horizontally
            current[2] = max(current[2], b[2])
            current[3] = max(current[3], b[3])
        else:
            merged.append(current)
            current = b.copy()
    merged.append(current)

    # Add margin
    expanded = []
    for (xmin, ymin, xmax, ymax) in merged:
        w, h = xmax - xmin, ymax - ymin
        dx, dy = w * margin_ratio, h * margin_ratio
        expanded.append([
            int(xmin - dx), int(ymin - dy),
            int(xmax + dx), int(ymax + dy)
        ])

    return expanded


def process_htr_page(image_path):
    image = cv2.imread(image_path)
    # original_image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Needs image image")

    # Run HTR pipeline
    detector_cfg = DetectorConfig()
    line_cfg = LineClusteringConfig()
    page = read_page(image, detector_config=detector_cfg, line_clustering_config=line_cfg)
    
    vis_img = image.copy()
    if len(vis_img.shape) == 2:  # grayscale -> BGR
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    # Draw lines and characters
    for line in page:
        for word in line:
            x1, y1, x2, y2 = word.aabb.xmin, word.aabb.ymin, word.aabb.xmax, word.aabb.ymax
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(vis_img, word.text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


    timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    output_path = os.path.join("img", "generated", f"htr_overlay_{timestamp}.png")
    writefile(output_path, vis_img)

    return page

def _ocr_words(image: np.ndarray, visualize: bool = True) -> List[Dict[str, Any]]:
    print('im in this bitch')
    config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data.get("text", []))
    print(f"OCR detected {n} entries")

    # make a copy for visualization

    vis_img = image.copy()
    if len(vis_img.shape) == 2:  # grayscale -> BGR
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

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

        word = {
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
        words.append(word)

        # Draw bounding boxes and text label
        if visualize:
            print('write: ' + text)
            color = (0, 255, 0) if conf > 0 else (0, 0, 255)  # green for confident, red for low conf
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(vis_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Sort by block->par->line->x
    words.sort(key=lambda w: (w["block_num"], w["par_num"], w["line_num"], w["left"]))

    cv2.rectangle(vis_img, (10, 10), (100, 100), (0,0,0), 1)

    # Display overlay
    if visualize:
        timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
        output_path = os.path.join("img", "generated", f"ocr_{timestamp}.png")
        writefile(output_path, vis_img)
        cv2.imshow("OCR Overlay", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return words



def process_chess_image(image_path: str) -> List[Dict[str, Any]]:
    image, img_w, img_h = _read_image(image_path)
    binary, scale = _preprocess(image)
    print(binary)
    words = _ocr_words(binary)
    print(words)



def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python chess_segmenter.py <image_path>")
    #     sys.exit(1)
    good_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/1759549223658-IMG_293F39624621-1.jpeg'
    bad_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/IMG_2F54EA6661FD-1.jpeg'

    img_path = good_img
    # img_path = bad_img
    # mask_file = process_chess_image(img_path)
    image, image_w, image_h = _read_image(img_path)
    cleaned, scale, cleaned_path = _preprocess(image)
    process_htr_page(img_path)


if __name__ == "__main__":
    main()




