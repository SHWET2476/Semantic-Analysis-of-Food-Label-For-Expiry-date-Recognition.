import cv2
import easyocr
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from datetime import datetime

# -------- Step 1: OCR using EasyOCR --------
reader = easyocr.Reader(['en'])  # English

def extract_text(image_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(results)
    return extracted_text, image

# -------- Step 2: Clean Text --------
def clean_text(text):
    return text.replace('\n', ' ').strip()

# -------- Step 3: Load BERT NER Model --------
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_expiry_date(text):
    ner_results = ner_pipeline(text)
    expiry_dates = [ent['word'] for ent in ner_results if ent['entity_group'] == 'DATE']
    return expiry_dates

# -------- Step 4: Calculate Time to Expiry --------
def days_until_expiry(date_str):
    date_formats = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"]
    for fmt in date_formats:
        try:
            expiry = datetime.strptime(date_str, fmt)
            return (expiry - datetime.now()).days
        except:
            continue
    return "Unknown or Invalid Format"

# -------- Step 5: Display Image and Results --------
def display_result(image, expiry_dates):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    title = " | ".join(expiry_dates) if expiry_dates else "No Expiry Date Found"
    plt.title(f"Detected Expiry Date(s): {title}", fontsize=12)
    plt.show()

# -------- Step 6: Main Function --------
def run_pipeline(image_name):  # image_name = "milk_label.jpg"
    text, image = extract_text(image_name)
    cleaned_text = clean_text(text)
    print("OCR Output:", cleaned_text)

    dates = extract_expiry_date(cleaned_text)
    print("Detected Dates:", dates)

    display_result(image, dates)

    for date in dates:
        print(f"> {date} → Days Until Expiry: {days_until_expiry(date)} days")

# -------- Run your model --------
# Just place your image in the same folder and run this:
run_pipeline("test_00001.jpg")
