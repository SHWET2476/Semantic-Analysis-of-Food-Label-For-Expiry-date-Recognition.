# import torch
# from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
# import re
# import cv2
# import matplotlib.pyplot as plt
# import os
# import argparse
# import easyocr

# #BERT_MODEL_LOADING
# tokenizer = BertTokenizerFast.from_pretrained("dslim/bert-base-NER")
# model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# #DATA_REGEX_PATTERN
# DATE_PATTERNS = [
#     r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b",              # 2024-04-22 / 2024.04.22 / 2024/04/22
#     r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b",              # 22-04-2024 / 22.04.2024
#     r"\b\d{2}[-/.]\d{2}[-/.]\d{2}\b",              # 22-04-24
#     r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",      # 22 April 2024 or 22 Apr 24
#     r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",       # April 22, 2024
#     r"\b\d{1,2}[A-Za-z]{2}\s+[A-Za-z]{3,9}\s+\d{4}\b",  # 22nd April 2024
#     r"\b[A-Za-z]{3,9}\s+\d{1,2}[,]?\s+\d{2,4}\b",  # Apr 22 2024 or April 22 2024
#     r"\b\d{2}/\d{2}/\d{4}\b",                      # 22/04/2024
#     r"\b\d{2}/\d{2}/\d{2}\b",                      # 22/04/24
# ]

# #EASYOCR_DATA_EXTRACTION
# def extract_text_from_image(image_path):
#     #TEXT EXTRACTION USING EASYOCR
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"[ERROR] Image path does not exist: {image_path}")

#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"[ERROR] Could not read image at: {image_path}")

#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_path, detail=0)  # Only get the text content
#     extracted_text = " ".join(result)

#     return image, extracted_text

# #DATA_DETECTION
# def detect_expiry_date(text):
#     #EXPIRT DATE DETECTION using a regex and NER pipeline
#     ner_results = ner_pipeline(text)
#     found_dates = []

#     # First use regex
#     for pattern in DATE_PATTERNS:
#         found = re.findall(pattern, text)
#         if found:
#             found_dates.extend(found)

#     # Fallback to NER
#     if not found_dates:
#         for ent in ner_results:
#             if ent['entity_group'] in ['DATE']:
#                 found_dates.append(ent['word'])

#     return list(set(found_dates))  # Remove duplicates

# # IMAGE_ANNOTATION
# def annotate_image(image, detected_dates):
#     #Detected image drawing on picture
#     annotated = image.copy()
#     if detected_dates:
#         text = f"Expiry Date: {', '.join(detected_dates)}"
#         cv2.putText(annotated, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#     return annotated

# # PIPELINE
# def process_image(image_path):
#     #MAIN FUNCTION
#     print(f"[INFO] Processing image: {image_path}")
#     image, extracted_text = extract_text_from_image(image_path)
#     print(f"[INFO] Extracted Text: {extracted_text}")

#     predicted_dates = detect_expiry_date(extracted_text)
#     print(f"[INFO] Detected Expiry Date(s): {predicted_dates}")

#     annotated_image = annotate_image(image, predicted_dates)
#     show_image(annotated_image)

#     return predicted_dates

# #IMAGE_DISPLAY
# def show_image(image):
#     #IMAGE DISPLAY
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(image_rgb)
#     plt.title("Annotated Image")
#     plt.axis("off")
#     plt.show()

# # CLI_EXECUTION
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Detect expiry dates using NER from image path")
#     parser.add_argument("image_path", type=str, help="Path to the input image (any location)")

#     args = parser.parse_args()
#     try:
#         detected_expiry = process_image(args.image_path)
#         if detected_expiry:
#             print(f"\nFinal Detected Expiry Date(s): {detected_expiry}")
#         else:
#             print("\n[WARNING] No expiry date detected.")
#     except Exception as e:
#         print(f"[ERROR] {str(e)}")





import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import re
import cv2
import matplotlib.pyplot as plt
import os
import easyocr
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import csv

# Load BERT tokenizer and model for NER
tokenizer = BertTokenizerFast.from_pretrained("dslim/bert-base-NER")
bert_model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=bert_model, tokenizer=tokenizer, aggregation_strategy="simple")

# CNN Feature extractor class
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC layer

    def forward(self, x):
        with torch.no_grad():
            return self.features(x).squeeze()

cnn_model = CNNFeatureExtractor()
cnn_model.eval()

DATE_PATTERNS = [
    r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b",
    r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b",
    r"\b\d{2}[-/.]\d{2}[-/.]\d{2}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
    r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",
    r"\b\d{1,2}[A-Za-z]{2}\s+[A-Za-z]{3,9}\s+\d{4}\b",
    r"\b[A-Za-z]{3,9}\s+\d{1,2}[,]?\s+\d{2,4}\b",
    r"\b\d{2}/\d{2}/\d{4}\b",
    r"\b\d{2}/\d{2}/\d{2}\b",
]

def extract_text(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path, detail=0)
    return " ".join(result)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Invalid image: {image_path}")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0), image


def detect_expiry_dates(text):
    found_dates = []

    # Regex detection
    for pattern in DATE_PATTERNS:
        found = re.findall(pattern, text)
        if found:
            found_dates.extend(found)

    # If regex finds nothing, try BERT NER
    if not found_dates:
        ner_results = ner_pipeline(text)
        for ent in ner_results:
            if ent['entity_group'] == 'DATE':
                found_dates.append(ent['word'])

    # Remove duplicates, clean whitespace
    found_dates = list(set([d.strip() for d in found_dates if d.strip()]))
    return found_dates

def annotate_image(image, dates):
    annotated = image.copy()
    if dates:
        text = f"Expiry: {', '.join(dates)}"
        cv2.putText(annotated, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return annotated

def process_image(image_path, visualize=False):
    print(f"[INFO] Processing: {image_path}")

    # Extract text with OCR
    text = extract_text(image_path)
    print(f"[INFO] Extracted Text: {text}")

    # Preprocess image and extract CNN features (optional usage)
    img_tensor, raw_image = preprocess_image(image_path)
    image_features = cnn_model(img_tensor)

    # Detect dates using regex + BERT NER
    dates = detect_expiry_dates(text)
    print(f"[INFO] Detected Dates: {dates}")

    # Optional visualization
    if visualize:
        annotated = annotate_image(raw_image, dates)
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title("Detected Expiry Date(s)")
        plt.axis("off")
        plt.show()

    return {
        "cnn_features": image_features,
        "text": text,
        "dates": dates
    }


def load_ground_truth(csv_path):
    # CSV format: image_path, expiry_date_1|expiry_date_2|...
    ground_truth = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img_path = row[0].strip()
            dates = row[1].split('|') if len(row) > 1 else []
            ground_truth[img_path] = [d.strip() for d in dates if d.strip()]
    return ground_truth

def calculate_metrics(predicted_dates, true_dates):
    pred_set = set(predicted_dates)
    true_set = set(true_dates)
    true_positives = len(pred_set & true_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(true_set) if true_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def evaluate_model_accuracy(dataset_csv, base_image_folder=None, visualize=True):
    ground_truth = load_ground_truth(dataset_csv)
    total_correct = 0
    total_samples = len(ground_truth)

    for img_path, true_dates in ground_truth.items():
        full_path = os.path.join(base_image_folder, img_path) if base_image_folder else img_path
        print(f"\nEvaluating image: {full_path}")
        try:
            result = process_image(full_path, visualize=visualize)
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            continue

        pred_dates = result['dates']

        # Consider correct if any predicted date matches any true date
        is_correct = any(d in true_dates for d in pred_dates)
        total_correct += int(is_correct)

        print(f"True dates: {true_dates}")
        print(f"Predicted dates: {pred_dates}")
        print(f"Correct Prediction: {is_correct}")

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print("\n=== Overall Accuracy ===")
    print(f"Accuracy: {accuracy:.3f}")