import os
import re
import cv2
import pytesseract
from datetime import datetime
from pytesseract import Output

def main():
    # --- Environment-aware configuration ---
    # Will use TESSERACT_CMD if set, otherwise default to "tesseract"
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
    print(TESSERACT_CMD)
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    good_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/1759549223658-IMG_293F39624621-1.jpeg'
    bad_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/IMG_2F54EA6661FD-1.jpeg'
    cropped_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/cropped.png'
    IMAGE_PATH = cropped_img
    image = cv2.imread(IMAGE_PATH)

    preprocessed = preprocess_img(image)


    # --- Run Tesseract OCR ---
    data = pytesseract.image_to_data(preprocessed, output_type=Output.DICT)
    print(data)

    # --- Regex pattern for chess move indexes ---
    # Matches 1., 2., (1), [1], or just 1 / 2 / 10 etc.
    index_pattern = re.compile(r"^\(?\[?\d{1,2}\.?\)?\]?$")

    # --- Iterate through OCR results and find indexes ---
    for i, text in enumerate(data["text"]):
        text_clean = text.strip()
        if not text_clean:
            continue

        if index_pattern.match(text_clean):
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                image,
                text_clean,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )

    # --- Display or save results ---
    # cv2.imshow("Detected Chess Indexes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optionally save the annotated image
    cv2.imwrite("chess_page_detected.jpg", image)

def writefile(name: str, image_data):
    try:
        timestamp = datetime.now().strftime("%H%M%S_%Y%m%d")
        output_path = os.path.join("img", "generated", f"{name}_{timestamp}.png")
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

def preprocess_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (normal + inverted)
    th1 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    th2 = cv2.adaptiveThreshold(
        255 - gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    merged = cv2.bitwise_or(th1, th2)

    # Clean up small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel, iterations=1)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    writefile('fuckkk', cleaned)
    return cleaned


if __name__ == "__main__":
    main()
