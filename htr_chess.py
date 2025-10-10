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


def main():
    good_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/1759549223658-IMG_293F39624621-1.jpeg'
    bad_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/IMG_2F54EA6661FD-1.jpeg'

    img_path = good_img
    process_htr_page(img_path)


if __name__ == "__main__":
    main()