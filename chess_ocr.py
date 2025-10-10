from paddleocr import PaddleOCR
import cv2
import numpy as np
import re

def main():
    

    good_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/1759549223658-IMG_293F39624621-1.jpeg'
    bad_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/IMG_2F54EA6661FD-1.jpeg'
    cropped_img = '/Users/adlaiabdelrazaq/Documents/code/personal/25/chess_image_analyzer/img/src/cropped.png'
    IMAGE_PATH = cropped_img
    image = cv2.imread(IMAGE_PATH)
    print(' got the image..')


    process_img(image)
    


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

def process_img(image):
    print('in processing')
    ocr = PaddleOCR(
        # det_model_dir='models/det',
        # rec_model_dir='models/rec',
        lang="en",
        use_angle_cls=True
    )
    print('about to run ocr')
    results = ocr.ocr(image, cls=True)

    number_pattern = re.compile(r"^\(?\[?\d{1,2}\.?\)?\]?$")
    print("results")
    print(results)

    for box, (text, conf) in results[0]:
        text = text.strip()
        if number_pattern.match(text) and conf > 0.5:
            pts = np.array(box).astype(int)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(
                image,
                text,
                (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            print(f"Found index '{text}' (conf={conf:.2f}) at {pts.tolist()}")

    writefile('yeaa', image)






if __name__ == "__main__":
    main()
