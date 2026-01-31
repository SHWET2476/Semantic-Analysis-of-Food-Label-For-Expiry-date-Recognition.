import torch
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import re
import cv2
import matplotlib.pyplot as plt
import os
import argparse
import easyocr
import json  # Import the json module

# BERT_MODEL_LOADING
tokenizer = BertTokenizerFast.from_pretrained("dslim/bert-base-NER")
model = BertForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# DATA_REGEX_PATTERN
DATE_PATTERNS = [
    r"\b\d{4}[-/.]\d{2}[-/.]\d{2}\b",              # 2024-04-22 / 2024.04.22 / 2024/04/22
    r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b",              # 22-04-2024 / 22.04.2024
    r"\b\d{2}[-/.]\d{2}[-/.]\d{2}\b",              # 22-04-24
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",      # 22 April 2024 or 22 Apr 24
    r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}\b",      # April 22, 2024
    r"\b[A-Za-z]{3,9}\s+\d{1,2},\s+\d{2}\b",        # April 22, 24
    r"\b\d{1,2}-[A-Za-z]{3,9}-\d{2,4}\b",          # 22-April-2024
    r"\b\d{1,2}/[A-Za-z]{3,9}/\d{2,4}\b",          # 22/April/2024
    r"\b\d{1,2}[A-Za-z]{3,9}\d{2,4}\b",            # 22April2024
    r"^(?:(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[012])[-/.](19|20)\d\d)$", #dd-mm-yyyy
    r"^(?:(0[1-9]|[12][0-9]|3[01])(0[1-9]|1[012])(19|20)\d\d)$" #ddmmyyyy
]
def extract_text_from_image(image_path):
    """
    Extracts text from an image using EasyOCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the original image (as a cv2 object) and the extracted text (string).
               Returns (None, "") if the image cannot be read.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return None, ""
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image)
        extracted_text = " ".join([text for _, text in results])
        return image, extracted_text
    except Exception as e:
        print(f"[ERROR] Error extracting text from image: {e}")
        return None, ""

def detect_dates_with_bert(text):
    """
    Detects date entities in the given text using a pre-trained BERT model.

    Args:
        text (str): The input text to analyze.

    Returns:
        list: A list of date strings identified by the BERT model.  Returns an empty list on error.
    """
    try:
        ner_results = ner_pipeline(text)
        dates = [entity['word'] for entity in ner_results if entity['entity_group'] == 'DATE']
        return dates
    except Exception as e:
        print(f"[ERROR] Error during BERT date detection: {e}")
        return []

def filter_valid_dates(dates):
    """
    Filters a list of date strings, returning only those that match predefined date patterns.

    Args:
        dates (list): A list of strings, potentially containing dates.

    Returns:
        list: A list of strings that match the defined date patterns.
    """
    valid_dates = []
    for date_str in dates:
        for pattern in DATE_PATTERNS:
            if re.search(pattern, date_str):
                valid_dates.append(date_str)
                break  # Important: Stop after finding one matching pattern
    return valid_dates

def process_image(image_path, annotations_data):
    """
    Processes an image to extract text, detect dates, and compare with annotations.

    Args:
        image_path (str): Path to the image file.
        annotations_data (dict):  Dictionary containing the date annotations from the JSON file.

    Returns:
        tuple:
            - annotated_image (cv2.Mat): The original image with detected dates annotated.
            - accuracy (float): The accuracy of the date detection (0 to 1). Returns None on error.
            - extracted_text (str): the text extracted from the image.
    """
    print(f"[INFO] Processing image: {image_path}")
    image, extracted_text = extract_text_from_image(image_path)
    if image is None:
        return None, None, "" # Error occurred in image loading

    print(f"[INFO] Extracted Text: {extracted_text}")

    predicted_dates = detect_dates_with_bert(extracted_text)
    predicted_dates = filter_valid_dates(predicted_dates)
    print(f"[INFO] Detected Dates: {predicted_dates}")

    # Get the filename without the directory
    image_name = os.path.basename(image_path)

    # Get expected dates from the JSON data.
    expected_dates = []
    if image_name in annotations_data:
        for annotation in annotations_data[image_name]['ann']:
            expected_dates.append(annotation['transcription'])

    print(f"[INFO] Expected Dates: {expected_dates}")

    # Calculate accuracy
    if not expected_dates:
        accuracy = 0.0 if not predicted_dates else 0.0 # No expected dates, 100% if no predictions
    else:
        correct_predictions = sum(1 for date in predicted_dates if date in expected_dates)
        accuracy = correct_predictions / len(expected_dates) if expected_dates else 0.0

    print(f"[INFO] Accuracy: {accuracy:.2f}")

    annotated_image = annotate_image(image, predicted_dates)
    return annotated_image, accuracy, extracted_text

def annotate_image(image, detected_dates):
    """
    Annotates the image with the detected dates.

    Args:
        image (cv2.Mat): The image to annotate.
        detected_dates (list): A list of date strings to annotate on the image.

    Returns:
        cv2.Mat: The annotated image.
    """
    annotated = image.copy()
    if detected_dates:
        text = f"Detected Dates: {', '.join(detected_dates)}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return annotated

def show_image(image, title="Annotated Image"):
    """
    Displays the image using matplotlib.

    Args:
        image (cv2.Mat): The image to display.
        title (str, optional): The title of the image window. Defaults to "Annotated Image".
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

def load_annotations(json_file_path):
    """Loads the date annotations from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary containing the annotations.  Returns an empty dict on error.
    """
    try:
        with open(json_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load annotations from {json_file_path}: {e}")
        return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect dates in images and compare with JSON annotations.")
    parser.add_argument("image_path", type=str, help="Path to the image file or directory.")
    parser.add_argument("json_path", type=str, help="Path to the JSON annotation file.")
    args = parser.parse_args()

    annotations_data = load_annotations(args.json_path)
    if not annotations_data:
        print("[ERROR] No annotation data loaded. Exiting.")
        exit(1)

    image_path = args.image_path
    if os.path.isdir(image_path):
        # Process all images in the directory
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_accuracy = 0
        num_images = 0
        for img_file in image_files:
            annotated_image, accuracy, extracted_text = process_image(img_file, annotations_data)
            if annotated_image is not None and accuracy is not None: # Check for errors during processing
                show_image(annotated_image, title=f"Processed: {os.path.basename(img_file)}  Accuracy: {accuracy:.2f}")
                total_accuracy += accuracy
                num_images += 1
            elif annotated_image is None:
                print(f"[ERROR] Skipping image: {img_file}")

        if num_images > 0:
            average_accuracy = total_accuracy / num_images
            print(f"[INFO] Average Accuracy: {average_accuracy:.2f} over {num_images} images")
        else:
            print(f"[INFO] No images processed.")

    else:
        # Process a single image
        annotated_image, accuracy, extracted_text = process_image(image_path, annotations_data)
        if annotated_image is not None and accuracy is not None:
            show_image(annotated_image, title=f"Processed: {os.path.basename(image_path)}  Accuracy: {accuracy:.2f}")
        else:
            print(f"[ERROR] Could not process the image.  Check the path and image format.")
