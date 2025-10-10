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
from collections import defaultdict

def writefile(output_path: str, image_data):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, image_data)
        if success:
            return output_path
            print(f"Cleaned image saved to {output_path}")
        else:
            return ''
            print(f"Failed to write image file: {output_path}")
    except Exception as e:
        print(f"Could not save cleaned image ({type(e).__name__}): {e}")
        return ''

def process_htr_page(image):
    
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


    # timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    # output_path = os.path.join("img", "generated", f"htr_overlay_{timestamp}.png")
    # writefile(output_path, vis_img)

    return page

def cluster_rows(aabbs, eps=25):
    """Cluster boxes into rows based on vertical proximity."""
    if not aabbs:
        return []
    ys = np.array([[ (ymin + ymax) / 2 ] for (_, ymin, _, ymax, _) in aabbs])
    clustering = DBSCAN(eps=eps, min_samples=1).fit(ys)
    clusters = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        clusters[label].append(aabbs[i])
    return list(clusters.values())

def cluster_columns(rows, eps=50):
    """Cluster horizontally within each row to find column groups."""
    col_positions = []
    for row in rows:
        xs = [ (xmin + xmax) / 2 for (xmin, _, xmax, _, _) in row ]
        col_positions.extend(xs)
    col_positions = np.array(col_positions).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(col_positions)
    unique_cols = sorted([np.mean([x for x, c in zip(col_positions.flatten(), clustering.labels_) if c == i])
                          for i in set(clustering.labels_)])
    return unique_cols

def draw_table(image, page):
    aabbs = []

    for line in page:
        for word in line:
            xmin, ymin = word.aabb.xmin, word.aabb.ymin
            xmax, ymax = word.aabb.xmax, word.aabb.ymax
            text = word.text.strip()
            aabbs.append((xmin, ymin, xmax, ymax, text))
    
    row_clusters = cluster_rows(aabbs, eps=100)
    col_centers = cluster_columns(row_clusters, eps=100)

    out = image.copy()
    for row_idx, row in enumerate(row_clusters):
        color = (0,0,0)
        for (xmin, ymin, xmax, ymax, _) in row:
            cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)
    for x in col_centers:
        cv2.line(out, (int(x), 0), (int(x), out.shape[0]), color, 1)

    timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    output_path = os.path.join("img", "generated", f"htr_overlay_{timestamp}.png")
    writefile(output_path, out)





def main():
    good_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/1759549223658-IMG_293F39624621-1.jpeg'
    bad_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/IMG_2F54EA6661FD-1.jpeg'

    img_path = good_img
    # img_path = bad_img

    image = cv2.imread(img_path)
    page = process_htr_page(image)
    draw_table(image,page)




if __name__ == "__main__":
    main()