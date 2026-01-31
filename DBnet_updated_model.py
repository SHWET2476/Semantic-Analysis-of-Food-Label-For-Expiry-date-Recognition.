import torch
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import re
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from paddleocr import PaddleOCR

# BERT MODEL
tokenizer = BertTokenizerFast.from_pretrained("dslim/bert-base-NER")
model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# DATE REGEX PATTERNS
DATE_PATTERNS = [
    r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b",  # YYYY-MM-DD or YYYY.MM.DD or YYYY/MM/DD
    r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b",  # DD-MM-YYYY or similar
    r"\b\d{2}[-/.]\d{2}[-/.]\d{2}\b"  # YY-MM-DD or similar
]



# TEXT DETECTION USING DBNet (PaddleOCR)
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image path does not exist: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"[ERROR] Could not read image at: {image_path}")

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image_path, cls=True)

    all_text = []
    for line in result[0]:
        text = line[1][0]
        all_text.append(text)

    extracted_text = " ".join(all_text)
    return image, extracted_text


# EXPIRY DATE DETECTION
def detect_expiry_date(text):
    ner_results = ner_pipeline(text)
    found_dates = []

    # First try regex
    for pattern in DATE_PATTERNS:
        found = re.findall(pattern, text)
        if found:
            found_dates.extend(found)

    # Fallback to BERT NER
    if not found_dates:
        for ent in ner_results:
            if ent['entity_group'] == 'DATE':
                found_dates.append(ent['word'])

    return list(set(found_dates))  # Remove duplicates


# ANNOTATE IMAGE
def annotate_image(image, detected_dates):
    annotated = image.copy()
    if detected_dates:
        text = f"Expiry Date: {', '.join(detected_dates)}"
        cv2.putText(annotated, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return annotated



# MAIN PROCESSING PIPELINE
def process_image(image_path):
    print(f"[INFO] Processing image: {image_path}")
    image, extracted_text = extract_text_from_image(image_path)
    print(f"[INFO] Extracted Text: {extracted_text}")

    predicted_dates = detect_expiry_date(extracted_text)
    print(f"[INFO] Detected Expiry Date(s): {predicted_dates}")

    annotated_image = annotate_image(image, predicted_dates)
    show_image(annotated_image)

    return predicted_dates


#SHOW ANNOTATED IMAGE
def show_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Annotated Image")
    plt.axis("off")
    plt.show()


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect expiry dates using DBNet + BERT from image")
    parser.add_argument("image_path", type=str, help="Path to the input image (e.g., /path/to/image.jpg)")

    args = parser.parse_args()
    try:
        detected_expiry = process_image(args.image_path)
        if detected_expiry:
            print(f"\nFinal Detected Expiry Date(s): {detected_expiry}")
        else:
            print("\n[WARNING] No expiry date detected.")
    except Exception as e:
        print(f"[ERROR] {str(e)}")
