import sys
import json
import uuid
import os
import cv2
import numpy as np

import logging
from flask import Flask, request, jsonify
from datetime import datetime


from typing import List, Dict, Any, Tuple

from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig
from collections import defaultdict

# --- HTR dependencies (import your own modules here) ---
# from your_module import DetectorConfig, LineClusteringConfig, read_page, writefile

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Secret API key for Express-to-Flask auth
API_KEY = os.getenv("HTR_API_KEY", "change_me_for_production")

# Max upload size (in bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME = {"image/png", "image/jpeg"}

# Output directory for debug overlays
OUTPUT_DIR = os.path.join("img", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Flask app setup
# -------------------------------------------------------------------

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Security & validation middleware
# -------------------------------------------------------------------

@app.before_request
def verify_api_key():
    """Verify API key header."""
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        logger.warning("Unauthorized request blocked")
        return jsonify({"error": "Unauthorized"}), 401


@app.before_request
def limit_upload_size():
    """Enforce maximum upload size."""
    if request.content_length and request.content_length > MAX_FILE_SIZE:
        logger.warning("File too large")
        return jsonify({"error": "File too large"}), 413


# -------------------------------------------------------------------
# Core processing logic
# -------------------------------------------------------------------

def process_htr_page(image):
    """Run HTR pipeline and return word bounding boxes in percentages."""
    if image is None:
        raise ValueError("Missing image data")

    # Example HTR flow (replace with your real pipeline)
    detector_cfg = DetectorConfig()
    line_cfg = LineClusteringConfig()
    page = read_page(image, detector_config=detector_cfg, line_clustering_config=line_cfg)

    vis_img = image.copy()
    if len(vis_img.shape) == 2:  # grayscale -> BGR
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    img_h, img_w = vis_img.shape[:2]
    boxes = []

    for line in page:
        for word in line:
            x1, y1, x2, y2 = word.aabb.xmin, word.aabb.ymin, word.aabb.xmax, word.aabb.ymax
            w = x2 - x1
            h = y2 - y1

            # Convert to percentage
            left_pct = (x1 / img_w) * 100
            top_pct = (y1 / img_h) * 100
            width_pct = (w / img_w) * 100
            height_pct = (h / img_h) * 100

            boxes.append({
                "id": str(uuid.uuid4()),
                "text": word.text,
                "top": round(top_pct, 4),
                "left": round(left_pct, 4),
                "width": round(width_pct, 4),
                "height": round(height_pct, 4)
            })

            # Optional: draw overlay
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(vis_img, word.text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Save debug overlay image
    # timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
    # output_path = os.path.join(OUTPUT_DIR, f"htr_overlay_{timestamp}.png")
    # writefile(output_path, vis_img)

    return {"boxes": boxes, "imageSize": {"width": img_w, "height": img_h}}


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/process", methods=["POST"])
def process_image():
    """Main endpoint for image upload and processing."""
    print('entering process image')
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    if file.mimetype not in ALLOWED_MIME:
        return jsonify({"error": f"Unsupported file type: {file.mimetype}"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = process_htr_page(image)
        logger.info(f"Processed image: {file.filename}, {len(result['boxes'])} boxes found")
        return jsonify(result)
    except Exception as e:
        logger.exception("Error processing image")
        return jsonify({"error": str(e)}), 500


@app.after_request
def log_request(response):
    """Log all requests."""
    logger.info(f"{request.remote_addr} {request.method} {request.path} -> {response.status}")
    return response


# -------------------------------------------------------------------
# Entry point (for local debugging only — not production)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Local-only dev server; use Gunicorn in production!
    app.run(host="127.0.0.1", port=5001, debug=False)

