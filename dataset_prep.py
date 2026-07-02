import os
import json
import pdfplumber
from PIL import Image

def normalize_bbox(bbox, width, height):
    """
    LayoutLMv3 requires bounding boxes to be normalized to a 0-1000 scale.
    """
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height))
    ]

def extract_pdf_features(pdf_path):
    """
    Extracts tokens, bboxes and page size from a PDF file.
    """
    data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            width = page.width
            height = page.height
            
            # Extract words along with their bounding boxes
            words = page.extract_words()
            
            tokens = []
            bboxes = []
            
            for word in words:
                tokens.append(word['text'])
                
                # pdfplumber format: (x0, top, x1, bottom)
                bbox = [word['x0'], word['top'], word['x1'], word['bottom']]
                normalized_box = normalize_bbox(bbox, width, height)
                bboxes.append(normalized_box)
            
            data.append({
                "page": page_idx,
                "tokens": tokens,
                "bboxes": bboxes,
                "width": width,
                "height": height
            })
            
    return data

if __name__ == "__main__":
    # Example usage:
    # pdf_path = "sample_quote.pdf"
    # if os.path.exists(pdf_path):
    #     features = extract_pdf_features(pdf_path)
    #     print(json.dumps(features[0], indent=2))
    print("dataset_prep.py: Template script for preprocessing PDFs for LayoutLMv3 loaded.")
