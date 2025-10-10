import sys
import re
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
    if image is None:
        raise ValueError(f"Needs image image")

    # Run HTR pipeline
    detector_cfg = DetectorConfig()
    line_cfg = LineClusteringConfig()
    page = read_page(image, detector_config=detector_cfg, line_clustering_config=line_cfg)

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


def _extract_boxes_from_page(page):
    boxes = []
    for line in page:
        for w in line:
            boxes.append([w.aabb.xmin, w.aabb.ymin, w.aabb.xmax, w.aabb.ymax])
    return np.array(boxes, dtype=np.int32)

def cluster_boxes_dbscan(boxes: np.ndarray):
    if len(boxes) == 0:
        return []
    cx = (boxes[:,0] + boxes[:,2]) / 2.0
    cy = (boxes[:,1] + boxes[:,3]) / 2.0
    w  = (boxes[:,2] - boxes[:,0])
    h  = (boxes[:,3] - boxes[:,1])
    med_w = np.median(w) if len(w) else 20
    med_h = np.median(h) if len(h) else 20

    X = np.stack([cx/ (med_w*1.8), cy/(med_h*1.2)], axis=1)
    db = DBSCAN(eps=1.0, min_samples=1).fit(X)
    labels = db.labels_

    clusters = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        group = boxes[idx]
        xmin = int(np.min(group[:,0]))
        ymin = int(np.min(group[:,1]))
        xmax = int(np.max(group[:,2]))
        ymax = int(np.max(group[:,3]))
        clusters.append(np.array([xmin,ymin,xmax,ymax], dtype=np.int32))
    clusters.sort(key=lambda b: (b[1], b[0]))
    return clusters

def prep_roi(img: np.ndarray, box: np.ndarray, pad_ratio: float=0.02):
    H, W = img.shape[:2]
    xmin,ymin,xmax,ymax = box.tolist()
    w = xmax - xmin
    h = ymax - ymin
    dx = int(w*pad_ratio)
    dy = int(h*pad_ratio)
    xmin = max(0, xmin - dx)
    ymin = max(0, ymin - dy)
    xmax = min(W, xmax + dx)
    ymax = min(H, ymax + dy)
    roi = img[ymin:ymax, xmin:xmax]

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    scale = 2.0 if max(gray.shape) < 120 else 1.5
    roi_big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(roi_big, (3,3), 0)
    th  = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 21, 10)
    th_inv = cv2.bitwise_not(th)
    return th, th_inv

def normalize_move(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s.replace("0","O").replace("o","O")
    s = s.replace("–","-").replace("—","-")
    s = s.upper()
    s = re.sub(r"[^KQRNBA-H1-8X=+\-O#]", "", s)
    if s in {"OO","O-O"}: return "O-O"
    if s in {"OOO","O-O-O"}: return "O-O-O"
    return s

def ocr_move_from_roi(roi: np.ndarray) -> str:
    CHESS_WHITELIST = "KQRNBabcdefgh12345678xX=+O-#o"
    CONFIGS = [
        f"--oem 1 --psm 8 -c tessedit_char_whitelist={CHESS_WHITELIST}",
        f"--oem 1 --psm 7 -c tessedit_char_whitelist={CHESS_WHITELIST}",
        f"--oem 1 --psm 13 -c tessedit_char_whitelist={CHESS_WHITELIST}",
    ]
    SAN_REGEX = re.compile(
        r"^(?:O-O(?:-O)?|[KQRNB]?[A-H]?[1-8]?x?[A-H][1-8](?:=[QRNB])?)[+#]?$",
        re.IGNORECASE
    )
    candidates = []
    for cfg in CONFIGS:
        txt = pytesseract.image_to_string(roi, config=cfg).strip()
        if txt: candidates.append(txt)
    roi_inv = cv2.bitwise_not(roi)
    for cfg in CONFIGS:
        txt = pytesseract.image_to_string(roi_inv, config=cfg).strip()
        if txt: candidates.append(txt)
    for raw in candidates:
        mv = normalize_move(raw)
        if SAN_REGEX.match(mv):
            return mv
    if candidates:
        return normalize_move(max(candidates, key=len))
    return ""

def ocr_chess_moves_from_htr(image: np.ndarray, page):
    boxes = _extract_boxes_from_page(page)
    print('boxes are:')
    print(boxes)
    # clusters = cluster_boxes_dbscan(boxes)
    # print('clusters are:')
    # print(clusters)
    results = []
    for box in boxes:
        th, th_inv = prep_roi(image, box, pad_ratio=0.02)
        move = ocr_move_from_roi(th) or ocr_move_from_roi(th_inv)
        results.append((tuple(box.tolist()), move))
    return results

def overlay_moves(img_path, moves):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    # Loop through boxes and labels
    for (xmin, ymin, xmax, ymax), label in moves:
        color = (0, 255, 0) if label else (0, 0, 255)  # green if text found, red if blank
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        if label:
            cv2.putText(
                image,
                label,
                (xmin, max(20, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
    timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    ocrwithmoves_path = os.path.join("img", "generated", f"ocrwithmoves_{timestamp}.png")
    writefile(ocrwithmoves_path, image)


def main():
    good_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/1759549223658-IMG_293F39624621-1.jpeg'
    bad_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/IMG_2F54EA6661FD-1.jpeg'

    img_path = good_img
    # img_path = bad_img
    image, image_w, image_h = _read_image(img_path)
    cleaned, scale, cleaned_path = _preprocess(image)
    page = process_htr_page(img_path)
    moves = ocr_chess_moves_from_htr(image, page)
    # print('moves are:')
    # print(moves)
    # good_moves = 
    # bad_moves = [((864, 105, 934, 138), 'EO'), ((255, 145, 432, 160), 'A'), ((708, 189, 893, 253), '225'), ((616, 193, 654, 245), ''), ((263, 201, 357, 250), 'E'), ((363, 201, 553, 260), 'EEK'), ((169, 222, 241, 271), 'CE'), ((888, 297, 1151, 439), 'AED'), ((176, 305, 249, 377), ''), ((450, 312, 556, 368), 'BB-#'), ((568, 314, 688, 373), 'QA'), ((262, 315, 335, 383), 'A'), ((743, 316, 819, 376), 'A'), ((380, 319, 424, 363), ''), ((84, 436, 101, 478), ''), ((1083, 500, 1126, 565), ''), ((949, 505, 1036, 577), '-F'), ((473, 507, 512, 570), 'X'), ((280, 510, 368, 576), '+'), ((544, 510, 575, 578), '5'), ((823, 514, 907, 577), '2'), ((150, 528, 204, 601), ''), ((85, 544, 103, 623), ''), ((275, 584, 325, 685), ''), ((547, 588, 737, 691), 'NO'), ((345, 602, 439, 677), '1'), ((146, 616, 208, 679), ''), ((999, 616, 1060, 686), 'BD'), ((787, 617, 949, 684), '5N'), ((628, 704, 679, 784), 'F-'), ((995, 704, 1022, 776), ''), ((868, 705, 960, 775), 'NE'), ((1035, 706, 1118, 772), 'CE'), ((713, 711, 745, 776), '5'), ((784, 711, 842, 778), 'BE'), ((533, 722, 574, 791), ''), ((279, 730, 379, 800), 'EO'), ((149, 734, 182, 790), '#'), ((993, 801, 1130, 888), 'BX'), ((876, 815, 958, 868), 'X'), ((759, 819, 868, 967), 'EO'), ((277, 822, 427, 913), 'E-3'), ((530, 842, 564, 902), 'N'), ((149, 845, 201, 905), '-'), ((609, 846, 668, 912), '+'), ((686, 855, 726, 914), ''), ((884, 884, 997, 966), 'E'), ((1020, 896, 1137, 962), 'K-24'), ((276, 925, 430, 1041), 'A3'), ((629, 945, 676, 1016), 'KE'), ((543, 965, 606, 1022), 'BE'), ((165, 967, 218, 1033), 'O-'), ((1025, 982, 1131, 1053), 'OR'), ((883, 993, 984, 1065), 'A4'), ((801, 1003, 834, 1058), 'C'), ((288, 1063, 337, 1150), ''), ((366, 1086, 467, 1152), 'F-D-'), ((812, 1086, 973, 1165), 'K6'), ((164, 1091, 227, 1156), '6-'), ((1009, 1096, 1078, 1152), 'Q'), ((537, 1099, 580, 1190), '4'), ((627, 1108, 702, 1177), '6'), ((401, 1174, 548, 1272), 'HEE'), ((1085, 1182, 1106, 1244), ''), ((263, 1183, 386, 1279), 'EE'), ((794, 1183, 963, 1342), 'EE'), ((1000, 1186, 1056, 1270), 'K'), ((558, 1193, 630, 1266), 'RE'), ((665, 1195, 714, 1242), '-'), ((158, 1202, 228, 1274), '7-'), ((995, 1275, 1106, 1401), 'E'), ((553, 1293, 639, 1373), 'X'), ((296, 1299, 340, 1371), '-'), ((675, 1299, 761, 1349), 'E'), ((157, 1303, 225, 1383), 'F-'), ((372, 1306, 398, 1371), '4'), ((902, 1355, 979, 1421), 'BX'), ((793, 1359, 878, 1417), '1-'), ((1040, 1362, 1106, 1422), ''), ((677, 1399, 1116, 1559), 'E2ONE'), ((551, 1406, 629, 1475), 'RAE'), ((278, 1407, 427, 1480), 'AE'), ((167, 1426, 234, 1579), 'A'), ((612, 1501, 658, 1557), ''), ((546, 1509, 596, 1586), ''), ((366, 1514, 401, 1564), ''), ((251, 1525, 319, 1598), 'RE'), ((780, 1585, 812, 1619), 'C'), ((813, 1586, 1014, 1650), 'E=AE'), ((304, 1604, 440, 1673), 'O'), ((799, 1605, 1038, 1780), 'EK'), ((515, 1606, 654, 1672), 'NO6'), ((191, 1621, 262, 1683), ''), ((111, 1645, 123, 1681), '4'), ((795, 1652, 840, 1695), 'B'), ((1053, 1654, 1121, 1716), 'AE'), ((1004, 1722, 1091, 1772), 'ER'), ((174, 1781, 253, 1848), 'A'), ((285, 1811, 305, 1841), '-'), ((30, 1827, 84, 1853), 'A')]
    # moves = bad_moves
    
    overlay_moves(img_path, moves)



if __name__ == "__main__":
    main()




