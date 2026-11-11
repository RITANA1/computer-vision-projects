
import cv2
import pytesseract
from pytesseract import Output
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Tesseract executable
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

TESSDATA_DIR = BASE_DIR / "tessdata"
IMAGE_PATH = BASE_DIR / "images" / "sample1.png"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Check image
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Check Chinese language data
if not (TESSDATA_DIR / "chi_sim.traineddata").exists():
    raise FileNotFoundError(
        f"Chinese language data not found: {TESSDATA_DIR / 'chi_sim.traineddata'}"
    )

# Read image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError(f"OpenCV cannot read this image: {IMAGE_PATH}")


# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Remove noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply Otsu thresholding
_, processed = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)


# ==========================================
# 3. OCR CONFIGURATION
# ==========================================

# Simplified Chinese recognition
LANGUAGE = "chi_sim"

# Configure Tesseract
config = f"--tessdata-dir {TESSDATA_DIR.as_posix()} --psm 6"


# ==========================================
# 4. TEXT RECOGNITION
# ==========================================

text = pytesseract.image_to_string(
    processed,
    lang=LANGUAGE,
    config=config
)


# ==========================================
# 5. TEXT DETECTION AND CONFIDENCE
# ==========================================

data = pytesseract.image_to_data(
    processed,
    lang=LANGUAGE,
    config=config,
    output_type=Output.DICT
)

result_image = image.copy()
confidences = []

for i, word in enumerate(data["text"]):

    word = word.strip()

    if not word:
        continue

    confidence = float(data["conf"][i])

    if confidence >= 0:
        confidences.append(confidence)

    # Draw bounding boxes around recognized words
    if confidence >= 30:

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


# ==========================================
# 6. CALCULATE AVERAGE CONFIDENCE
# ==========================================

average_confidence = (
    sum(confidences) / len(confidences)
    if confidences else 0
)


# ==========================================
# 7. SAVE RESULTS
# ==========================================

# Save extracted Chinese text
text_file = OUTPUT_DIR / "extracted_text.txt"

text_file.write_text(
    text,
    encoding="utf-8"
)

# Save image with detected text boxes
result_file = OUTPUT_DIR / "ocr_result.jpg"

cv2.imwrite(
    str(result_file),
    result_image
)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n========== CHINESE OCR RESULTS ==========")

print("\nRecognized text:")
print(text)

print(f"\nAverage confidence: {average_confidence:.2f}%")

print(f"\nText saved to: {text_file}")
print(f"Image saved to: {result_file}")


# ==========================================
# 9. DISPLAY OCR RESULT
# ==========================================

cv2.imshow("Chinese OCR Result", result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
