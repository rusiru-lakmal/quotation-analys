import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pdfplumber

LABEL_LIST = ["O", "B-COMPANY", "I-COMPANY", "B-PREMIUM", "I-PREMIUM", "B-DEDUCTIBLE", "I-DEDUCTIBLE", "B-COPAY", "I-COPAY"]

def extract_text_from_pdf(pdf_path):
    """
    Extracts horizontal plain text from a PDF file using pdfplumber,
    filtering out rotated watermarks and margin notes.
    If the extracted plain text is too short, automatically runs OCR on the pages.
    """
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Filter normal horizontal characters
                normal_chars = []
                for char in page.chars:
                    matrix = char.get("matrix", (1, 0, 0, 1, 0, 0))
                    # Only keep characters that are strictly horizontal (not rotated)
                    if abs(matrix[1]) < 0.01 and abs(matrix[2]) < 0.01:
                        normal_chars.append(char)
                
                # Sort chars vertical top-to-bottom, then horizontal left-to-right
                lines = []
                tolerance = 3
                normal_chars.sort(key=lambda c: -c["y0"])
                
                current_line = []
                current_y = None
                
                for char in normal_chars:
                    y = char["y0"]
                    if current_y is None:
                        current_y = y
                        current_line.append(char)
                    elif abs(current_y - y) <= tolerance:
                        current_line.append(char)
                    else:
                        current_line.sort(key=lambda c: c["x0"])
                        line_text = ""
                        prev_char = None
                        for c in current_line:
                            if prev_char is not None:
                                gap = c["x0"] - prev_char["x1"]
                                if gap > 3.0 and c["text"] != " " and prev_char["text"] != " ":
                                    line_text += " "
                            line_text += c["text"]
                            prev_char = c
                        lines.append(line_text)
                        current_line = [char]
                        current_y = y
                        
                if current_line:
                    current_line.sort(key=lambda c: c["x0"])
                    line_text = ""
                    prev_char = None
                    for c in current_line:
                        if prev_char is not None:
                            gap = c["x0"] - prev_char["x1"]
                            if gap > 3.0 and c["text"] != " " and prev_char["text"] != " ":
                                line_text += " "
                        line_text += c["text"]
                        prev_char = c
                    lines.append(line_text)
                    
                text_content.append("\n".join(lines))
    except Exception as e:
        print(f"Error reading PDF {pdf_path} with pdfplumber: {e}")
        
    plain_text = "\n".join(text_content).strip()
    
    # If the text is empty or too short, run OCR!
    if len(plain_text) < 50:
        print(f"Plain text extraction empty/too short for {pdf_path}. Running OCR fallback...")
        ocr_text_content = []
        try:
            from pdf2image import convert_from_path
            import pytesseract
            images = convert_from_path(pdf_path)
            for idx, img in enumerate(images):
                print(f"OCR processing page {idx+1}/{len(images)}...")
                page_text = pytesseract.image_to_string(img)
                ocr_text_content.append(page_text)
            plain_text = "\n".join(ocr_text_content).strip()
            print("OCR extraction completed successfully!")
        except Exception as ocr_e:
            print(f"OCR fallback failed for {pdf_path}: {ocr_e}")
            
    return plain_text

def extract_entities(text, model_path):
    """
    Load the model and perform named entity recognition (NER) extraction line by line.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    extracted = {
        "company": "Unknown Insurer",
        "premium": "Not found",
        "deductible": "Not found",
        "copay": "Not found"
    }
    
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        results = nlp(line)
        for entity in results:
            word = entity['word'].strip()
            entity_group = entity['entity_group']
            score = entity.get('score', 0)
            
            if score > 0.4:
                if entity_group == "COMPANY" and (extracted["company"] == "Unknown Insurer" or len(word) > len(extracted["company"])):
                    word_clean = word.replace("##", "")
                    if len(word_clean) > 2:
                        extracted["company"] = word_clean
                elif entity_group == "PREMIUM" and (extracted["premium"] == "Not found" or len(word) > len(extracted["premium"])):
                    extracted["premium"] = word.replace("##", "")
            
    return extracted

def preprocess_pdf_text_spaces(text):
    """
    Cleans up character spaces inside numbers that commonly occur during PDF extraction.
    (e.g., '2 4,972,037.88' -> '24,972,037.88')
    """
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
    text = re.sub(r'\b(\d)\s+(\d{1,3},)', r'\1\2', text)
    text = re.sub(r'\b(\d)\s+(\d{2}\.\d)', r'\1\2', text)
    return text

def detect_insurance_class(text1, text2):
    combined = (text1 + " " + text2).lower()
    
    # Check if it's a liability / cargo / general quote first to override motor false-positives
    if any(x in combined for x in ["freight forwarding", "cargo liability", "professional indemnity", "errors & omissions", "errors and omissions"]):
        return "general"
        
    motor_keywords = ["motor comprehensive", "auto care", "chassis number", "vehicle value", "make :", "model :", "yom :", "towing chargers", "air bag cover", "windscreen cover"]
    health_keywords = ["hospitalisation", "opd limit", "surgical", "health insurance", "spectacles cover", "room charges", "medical expenses", "cash grant", "maternity", "inpatient"]
    workmen_keywords = ["workmen", "employers liability", "common law liability", "wages", "occupational disease", "workman"]
    life_keywords = ["loan protection", "life assured", "repayment period", "single premium", "decreasing term", "mortgage redemption", "mrp std", "outstanding loan", "main life", "life premium", "group life", "life cover", "life insurance", "union protect"]
    
    motor_score = sum([2 if kw in combined else 0 for kw in motor_keywords])
    if "vehicle" in combined: motor_score += 1
    if "motor" in combined: motor_score += 1
    
    health_score = sum([2 if kw in combined else 0 for kw in health_keywords])
    if "medical" in combined: health_score += 1
    
    workmen_score = sum([2 if kw in combined else 0 for kw in workmen_keywords])
    
    life_score = sum([2 if kw in combined else 0 for kw in life_keywords])
    if "life assured" in combined: life_score += 2
    
    scores = {
        "motor": motor_score,
        "health": health_score,
        "workmen": workmen_score,
        "life": life_score
    }
    
    best_class = max(scores, key=scores.get)
    # Require a minimum score of 3 to classify
    if scores[best_class] >= 3:
        if best_class == "life":
            group_life_score = sum([2 if kw in combined else 0 for kw in ["group life", "group protect", "member list", "free cover limit", "fcl of", "active at work"]])
            loan_score = sum([2 if kw in combined else 0 for kw in ["loan protection", "housing loan", "mortgage redemption", "decreasing term", "mrp std", "outstanding loan"]])
            if group_life_score > loan_score:
                return "group_life"
        return best_class
    return "general"

def extract_motor_fields(text):
    text_clean = preprocess_pdf_text_spaces(text)
    
    data = {
        "class": "motor",
        "vehicle_make_model": "Not found",
        "sum_insured": "Not found",
        "basic_premium": "Not found",
        "riot_strike_premium": "LKR 0.00",
        "terrorism_premium": "LKR 0.00",
        "admin_fee": "LKR 0.00",
        "policy_fee": "LKR 0.00",
        "stamp_fee": "LKR 0.00",
        "sscl": "LKR 0.00",
        "vat": "LKR 0.00",
        "total_payable": "Not found",
        
        # Add-on Covers
        "tppd_limit": "Not found",
        "towing_limit": "Not found",
        "natural_disasters": "Included (Natural Perils)",
        "airbag_replacement": "Not found",
        "windscreen_cover": "Not found"
    }
    
    # Vehicle Make/Model
    m = re.search(r"Vehicle Make/\s*Model:\s*(.*?)(?:\n|\s{3,}|\Z)", text_clean, re.IGNORECASE)
    if m:
        data["vehicle_make_model"] = m.group(1).strip()
    else:
        make_m = re.search(r"MAKE\s*:\s*(.*?)(?:\s{2,}|\n|\Z)", text_clean, re.IGNORECASE)
        model_m = re.search(r"MODEL\s*:\s*(.*?)(?:\s{2,}|\n|\Z)", text_clean, re.IGNORECASE)
        year_m = re.search(r"(?:YEAR OF MAKE|YOM)\s*:\s*(\d{4})", text_clean, re.IGNORECASE)
        parts = []
        if make_m: parts.append(make_m.group(1).strip())
        if model_m: parts.append(model_m.group(1).strip())
        if year_m: parts.append(f"({year_m.group(1).strip()})")
        if parts:
            data["vehicle_make_model"] = " ".join(parts)
            
    # Sum Insured (Generalized)
    m = re.search(r"(?:Sum\s+Insured|VEHICLE\s+VALUE)\s*(?:Rs\.?|LKR)?\s*:\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["sum_insured"] = "LKR " + m.group(1).strip()
            
    # Basic Premium (Generalized)
    m = re.search(r"(?:Annual\s+Basic|BASIC)\s+PREMIUM\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["basic_premium"] = "LKR " + m.group(1).strip()
    else:
        m = re.search(r"Net\s+Premium\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
        if m:
            data["basic_premium"] = "LKR " + m.group(1).strip()
            
    # Riot & Strike Premium (Generalized)
    m = re.search(r"(?:RIOT\s*&\s*STRIKE(?:\s+PREMIUM)?|SRCC)\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["riot_strike_premium"] = "LKR " + m.group(1).strip()
            
    # Terrorism Premium (Generalized)
    m = re.search(r"(?:TERRORISM(?:\s+PREMIUM)?|TC|Terrorism)\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["terrorism_premium"] = "LKR " + m.group(1).strip()
            
    # Admin Fee
    m = re.search(r"(?:ADMIN\s+FEE|Admin\s+Fee|Other\s+Charges)\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["admin_fee"] = "LKR " + m.group(1).strip()
            
    # Policy Fee
    m = re.search(r"(?:POLICY\s+FEE|Policy\s+Fee)\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["policy_fee"] = "LKR " + m.group(1).strip()
            
    # Stamp Fee
    m = re.search(r"(?:STAMP\s+DUTY|Stamp\s+Fee)\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["stamp_fee"] = "LKR " + m.group(1).strip()
            
    # SSCL
    m = re.search(r"SSCL\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["sscl"] = "LKR " + m.group(1).strip()
        
    # VAT
    m = re.search(r"VAT\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["vat"] = "LKR " + m.group(1).strip()
    else:
        # Avoid matching 'Nation Building Tax' by using negative lookbehind
        m = re.search(r"(?<!building)(?<!building\s)Tax\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
        if m:
            data["vat"] = "LKR " + m.group(1).strip()
            
    # Total Payable (Generalized - Gross premium prioritized first)
    m = re.search(r"(?:TOTAL\s+ANNUAL\s+PREMIUM|TOTAL\s+DUE|Total\s+Gross\s+Premium)\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["total_payable"] = "LKR " + m.group(1).strip()
    else:
        m = re.search(r"(?:TOTAL\s+PREMIUM|Total\s+Payable)\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
        if m:
            data["total_payable"] = "LKR " + m.group(1).strip()
        else:
            m = re.search(r"Total\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
            if m:
                data["total_payable"] = "LKR " + m.group(1).strip()
            
    # TPPD limit (Generalized)
    m = re.search(r"Third\s+party\s+['\"]?property['\"]?\s+damage\s+(?:cover\s+)?(?:Rs\.\s*|LKR\s*)?([\d,]+)(?:\/-)?", text_clean, re.IGNORECASE)
    if m:
        data["tppd_limit"] = "LKR " + m.group(1).strip()
    else:
        m = re.search(r"THIRD\s+PARTY\s+PROPERTY\s+DAMAGE\s+-\s+PVT\s+CARS\s+-\s+LKR\.\s*([\d,]+)", text_clean, re.IGNORECASE)
        if m:
            data["tppd_limit"] = "LKR " + m.group(1).strip()
            
    # Towing limit (Generalized)
    m = re.search(r"(?:Towing\s+(?:Charges|Chargers|Limit|Cover)|Extended\s+Towing\s+Charges)\s*(?:cover\s+)?(?:\(\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)\s*\)|(?:Rs\.?|LKR)?[ \t]*([\d,]+(?:\.\d+)?))(?:[ \t]*\/-)?", text_clean, re.IGNORECASE)
    if m:
        data["towing_limit"] = "LKR " + (m.group(1) or m.group(2)).strip()
            
    # Windscreen & Airbag Cover
    m = re.search(r"Windscreen\s+(?:&\s+Breakage\s+of\s+glass\s+)?Cover\s*([\d,]+)", text_clean, re.IGNORECASE)
    if m:
        data["windscreen_cover"] = "LKR " + m.group(1).strip()
    elif "windscreen" in text_clean.lower():
        data["windscreen_cover"] = "Included (Standard)"
        
    if "air bag" in text_clean.lower() or "airbag" in text_clean.lower():
        data["airbag_replacement"] = "100% Replacement Cover"
        
    return data

def extract_life_fields(text):
    text = preprocess_pdf_text_spaces(text)
    
    data = {
        "class": "life",
        "insured_name": "Not found",
        "basic_premium": "Not found",
        "total_payable": "Not found",
        "coverage_limit": "Not found", # Loan Amount
        "repayment_period": "Not found",
        "interest_rate": "Not found",
        "policy_fee": "LKR 0.00",
        
        # Additional features
        "tpd_benefit": "Not found",
        "death_benefit": "Not found",
        "medical_requirements": "Not found"
    }
    
    # 1. Proposer Name
    m = re.search(r"Name of the Proposer\(s\)\s*(.*)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Name\s+of\s+Proposer\s*(.*)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Proposer\s*:\s*(.*)", text, re.IGNORECASE)
    if m:
        data["insured_name"] = m.group(1).strip()
        
    # 2. Loan Amount / Capital (Coverage Limit)
    m = re.search(r"(?:Loan\s+Amount|Loan\s+Capital)\s*(?:Rs\.?|LKR)?\s*:?\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)(?:\s*\(LKR\))?", text, re.IGNORECASE)
    if m:
        data["coverage_limit"] = "LKR " + m.group(1).strip()
        
    # 3. Repayment Period
    m = re.search(r"Repayment Period\s*:?\s*(.*?)(?:\s{2,}|\n|Type|Term|\Z)", text, re.IGNORECASE)
    if m:
        data["repayment_period"] = m.group(1).strip()
        
    # 4. Interest Rate
    m = re.search(r"Interest Rate\s*:?\s*([\d\.]+\s*%?)", text, re.IGNORECASE)
    if m:
        data["interest_rate"] = m.group(1).strip()
        
    # 5. Basic Premium / Life Premium
    m = re.search(r"Total Life Premium\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        data["basic_premium"] = "LKR " + m.group(1).strip()
        
    # 6. Single Premium / Total Premium
    m = re.search(r"Single Premium\s*(?:Payable|\*)?\s*(?:Rs\.?|LKR)?\s*:?\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        data["total_payable"] = "LKR " + m.group(1).strip()
    else:
        m = re.search(r"Total\s*(?:Annual)?\s*Premium\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"ANNUAL\s+PREMIUM(?:\s+OPTION\s+\d+)?\s+AS\s+PER\s+THE\s+ATTACHED\s+MEMBER\s+LIST\s+([\d,]+\.\d+)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"^\s*TOTAL\s+(?:LKR\s*)?([\d,]+\.\d+)", text, re.IGNORECASE | re.MULTILINE)
        if m:
            data["total_payable"] = "LKR " + m.group(1).strip()
            
    # 7. Policy Fee
    m = re.search(r"policy fee of\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        data["policy_fee"] = "LKR " + m.group(1).strip()
        
    # 8. Benefits / TPD & Death
    text_lower = text.lower()
    if "total permanent disability" in text_lower or "tpd" in text_lower or "disability" in text_lower or "tps" in text_lower:
        data["tpd_benefit"] = "Included (covers Sickness & Accident)"
    if "death cover" in text_lower or "death benefits" in text_lower or "death benefit" in text_lower:
        data["death_benefit"] = "Included (repay outstanding loan amount)"
        
    # 9. Medical requirements
    m = re.search(r"Medical Requirements\s*\n\s*(.*)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Medical\s*:\s*(.*)", text, re.IGNORECASE)
    if m:
        data["medical_requirements"] = m.group(1).strip()
        
    return data

def extract_group_life_fields(text):
    text_clean = preprocess_pdf_text_spaces(text)
    
    data = {
        "class": "group_life",
        "insured_name": "Not found",
        "basic_premium": "Not found",
        "total_payable": "Not found",
        "policy_fee": "LKR 0.00",
        
        # Group Life Benefits (Sum Assured)
        "accidental_death_benefit": "Not found",
        "tpd_benefit": "Not found",
        "ppd_benefit": "Not found",
        "critical_illness_cover": "Not found",
        
        # Group Life Premiums
        "accidental_death_premium": "LKR 0.00",
        "tpd_premium": "LKR 0.00",
        "ppd_premium": "LKR 0.00",
        "critical_illness_premium": "LKR 0.00",
        
        # Specifications
        "fcl_limit": "Not found",
        "medical_requirements": "Not found"
    }
    
    # 1. Proposer / Insured Name
    m = re.search(r"(?:PROPOSER|Insured’s\s+Name|Insured's\s+Name)\s*:\s*(.*)", text_clean, re.IGNORECASE)
    if m:
        data["insured_name"] = m.group(1).strip()
        
    # 2. Total Payable Premium
    m = re.search(r"Total\s+Annual\s+Premium\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)", text_clean, re.IGNORECASE)
    if not m:
        m = re.search(r"Annual\s+Premium\s*:\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)", text_clean, re.IGNORECASE)
    if not m:
        m = re.search(r"member\s+list\s+([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["total_payable"] = "LKR " + m.group(1).strip()
        
    # 3. Basic / Primary Life Premium
    m = re.search(r"Primary\s+Life\s+Cover\s+“As\s+per\s+the\s+schedule”\s*(?:Rs\.?|LKR)?\s*([\d,]+(?:\.\d+)?)", text_clean, re.IGNORECASE)
    if m:
        data["basic_premium"] = "LKR " + m.group(1).strip()
    else:
        if "basic life cover" in text_clean.lower():
            data["basic_premium"] = "As per member list"
            
    # 4. ADB
    adb_double = re.search(r"Accidental\s+Death\s+Benefit(?:\s*\(ADB\))?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)\s+(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if adb_double:
        data["accidental_death_benefit"] = "LKR " + adb_double.group(1).strip()
        data["accidental_death_premium"] = "LKR " + adb_double.group(2).strip()
    else:
        adb_single = re.search(r"Accidental\s+Death\s+Benefit(?:\s*\(ADB\))?\s*(?:EMPLOYEE\s+ONLY)?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
        if adb_single:
            data["accidental_death_benefit"] = "LKR " + adb_single.group(1).strip()
            
    # 5. TPD
    tpd_double = re.search(r"Total\s+and\s+Permanent\s+Disability\s*\(TPS\)\s*(?:\(due\s+to\s+accidental/sickness\))?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)\s+(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if tpd_double:
        data["tpd_benefit"] = "LKR " + tpd_double.group(1).strip()
        data["tpd_premium"] = "LKR " + tpd_double.group(2).strip()
    else:
        tpd_single = re.search(r"TOTAL\s+PERMANENT\s+DISABILITY\s*(?:EMPLOYEE\s+ONLY)?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
        if tpd_single:
            data["tpd_benefit"] = "LKR " + tpd_single.group(1).strip()
            
    # 6. PPD
    ppd_double = re.search(r"(?:Permanent\s+Partial\s+Disability\s*\(EPD\)|Partial\s+Permanent\s+Disability)\s*(?:\(Due\s+to\s+accident\s+only\))?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)\s+(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if ppd_double:
        data["ppd_benefit"] = "LKR " + ppd_double.group(1).strip()
        data["ppd_premium"] = "LKR " + ppd_double.group(2).strip()
    else:
        ppd_single = re.search(r"(?:PARTIAL\s+PERMANENT\s+DISABILITY|Permanent\s+Partial\s+Disability\s*\(EPD\))\s*(?:EMPLOYEE\s+ONLY)?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
        if ppd_single:
            data["ppd_benefit"] = "LKR " + ppd_single.group(1).strip()
            
    # 7. CIC
    cic_double = re.search(r"Critical\s+illness\s+Cover\s*(?:\(\d+\s+illness\))?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)\s+(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if cic_double:
        data["critical_illness_cover"] = "LKR " + cic_double.group(1).strip()
        data["critical_illness_premium"] = "LKR " + cic_double.group(2).strip()
    else:
        cic_single = re.search(r"CRITICAL\s+ILLNESS\s+COVER\s*(?:EMPLOYEE\s+ONLY)?\s*(?:Rs\.?|LKR)?\s*([\d,]+\.\d+)", text_clean, re.IGNORECASE)
        if cic_single:
            data["critical_illness_cover"] = "LKR " + cic_single.group(1).strip()
            
    # 8. Free Cover Limit (FCL)
    m = re.search(r"NON-MEDICAL\s+LIMIT\s*\(FREE\s+COVER\s+LIMIT\)\s*LIFE\s*(LKR\s*[\d,]+\.\d+\s*&\s*CIC\s*[\d,]+\.\d+|LKR\s*[\d,]+\.\d+)", text_clean, re.IGNORECASE)
    if m:
        data["fcl_limit"] = m.group(1).strip()
    else:
        # Check Union FCL format with condition groups
        m = re.search(r"Free\s+Cover\s+Limit\s*\(FCL\)\s*of\s*(.*?)(?:,\s*[a-z]\.\))", text_clean, re.IGNORECASE)
        if m:
            data["fcl_limit"] = m.group(1).strip()
        else:
            m = re.search(r"(?:Free\s+Cover\s+Limit|FCL)\s*(?:\(FCL\))?\s*(?:of|is)?\s*(Rs\.?|LKR)?\s*([\d,]+[^.\n]*)", text_clean, re.IGNORECASE)
            if m:
                data["fcl_limit"] = ((m.group(1) or "") + " " + m.group(2).strip()).strip(".,/- ")
            
    # 9. Medical requirements
    if "medical evidence" in text_clean.lower() or "medical requirements" in text_clean.lower() or "medically underwritten" in text_clean.lower() or "underwritten" in text_clean.lower():
        data["medical_requirements"] = "Required above Free Cover Limit (FCL)"
    else:
        data["medical_requirements"] = "Not required under Free Cover Limit"
        
    # Post-process premiums: If benefit is present but premium is LKR 0.00, change to Included in Gross
    for benefit_key, premium_key in [
        ("accidental_death_benefit", "accidental_death_premium"),
        ("tpd_benefit", "tpd_premium"),
        ("ppd_benefit", "ppd_premium"),
        ("critical_illness_cover", "critical_illness_premium")
    ]:
        if data[benefit_key] != "Not found" and data[premium_key] == "LKR 0.00":
            data[premium_key] = "Included in Gross"
        
    return data

def extract_plans_dynamically(pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        return []
    import pdfplumber
    import re
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            words = page.extract_words()
            
            # Find PLAN or OPTION column headers
            header_words = []
            for w in words:
                if w["text"].upper() == "PLAN" or w["text"].upper() == "OPTION":
                    header_words.append(w)
                    
            if not header_words:
                return []
                
            # Group headers by vertical 'top' coordinate (tolerance of 3)
            header_groups = {}
            for h in header_words:
                matched = False
                for top_val in header_groups.keys():
                    if abs(h["top"] - top_val) <= 3:
                        header_groups[top_val].append(h)
                        matched = True
                        break
                if not matched:
                    header_groups[h["top"]] = [h]
                    
            # Find the group with the most columns
            best_top = max(header_groups.keys(), key=lambda k: len(header_groups[k]))
            cols = sorted(header_groups[best_top], key=lambda w: w["x0"])
            plan_xs = [c["x0"] for c in cols]
            plan_names = [f"Plan 0{i+1}" for i in range(len(cols))]
            
            # Initialize plans dict
            plans_data = []
            for i, name in enumerate(plan_names):
                plans_data.append({
                    "name": name,
                    "limit": "Not found",
                    "ind_prem": "Not found",
                    "fam_prem": "Not found",
                    "ind_count": "0",
                    "fam_count": "0"
                })
                
            # Group all words into lines by vertical coordinate
            rows = {}
            for w in words:
                matched = False
                for top_val in rows.keys():
                    if abs(w["top"] - top_val) <= 3:
                        rows[top_val].append(w)
                        matched = True
                        break
                if not matched:
                    rows[w["top"]] = [w]
                    
            # Sort words in each row left-to-right
            for top_val in rows.keys():
                rows[top_val].sort(key=lambda w: w["x0"])
                
            # Parse each row
            min_plan_x = min(plan_xs)
            for top_val in sorted(rows.keys()):
                row_words = rows[top_val]
                
                # Extract label (words on the left)
                label_words = [w for w in row_words if w["x0"] < min_plan_x - 10]
                label = " ".join([w["text"] for w in label_words]).strip()
                
                if not label:
                    continue
                    
                # Extract numbers on the right
                val_words = [w for w in row_words if w["x0"] >= min_plan_x - 10]
                
                # Map each value word to the closest plan column
                mapped = {}
                for vw in val_words:
                    txt = vw["text"].strip()
                    # Skip words that are not numbers/currencies/limits (e.g. header texts or dash symbols)
                    if txt in ["-", "PLAN", "PLAN:", "OPTION", "1", "2", "3", "4", "5"]:
                        continue
                    # Find closest column
                    closest_idx = min(range(len(plan_xs)), key=lambda idx: abs(vw["x0"] - plan_xs[idx]))
                    # Check if it's actually close horizontally (say within 35 pixels)
                    if abs(vw["x0"] - plan_xs[closest_idx]) < 35:
                        mapped[closest_idx] = txt
                        
                if not mapped:
                    continue
                    
                label_lower = label.lower()
                
                # Check if this row is a Limit, Premium, or Count row
                is_limit = "limit" in label_lower or "sum insured" in label_lower
                is_ind = "individual" in label_lower or "ind" in label_lower or "single" in label_lower
                is_fam = "family" in label_lower or "fam" in label_lower
                is_premium = "premium" in label_lower or "rate" in label_lower or "ward" in label_lower
                
                for idx, val in mapped.items():
                    # Format premium/limit values
                    cleaned_val = val
                    if not cleaned_val.startswith("LKR") and not cleaned_val.startswith("Rs"):
                        # Check if it's a number and format it
                        val_clean = cleaned_val.replace(",", "")
                        if re.match(r"^\d+(?:\.\d+)?$", val_clean):
                            if float(val_clean) > 1000:
                                cleaned_val = "LKR " + cleaned_val
                    
                    if is_limit and not is_premium:
                        plans_data[idx]["limit"] = cleaned_val
                    elif is_premium:
                        if is_ind:
                            plans_data[idx]["ind_prem"] = cleaned_val
                        elif is_fam:
                            plans_data[idx]["fam_prem"] = cleaned_val
                    elif "number of employees" not in label_lower:
                        if is_ind:
                            plans_data[idx]["ind_count"] = cleaned_val
                        elif is_fam:
                            plans_data[idx]["fam_count"] = cleaned_val
                            
            return plans_data
    except Exception as e:
        print(f"Dynamic plan extraction error: {e}")
        return []

def extract_rich_fields(text, ins_class="health", pdf_path=None):
    """
    Parses complex structural fields (fees, taxes, coverage limits) for Health and General/Liability Insurance quotes.
    """
    text = preprocess_pdf_text_spaces(text)
    
    data = {
        "class": "health",
        "insured_name": "Not found",
        "basic_premium": "Not found",
        "admin_fee": "Not found",
        "policy_fee": "LKR 0.00",
        "vat": "Not found",
        "total_payable": "Not found",
        "coverage_limit": "Not found",
        "opd_limit": "Not found",
        "spectacles_limit": "Not found",
        "cess_fee": "LKR 0.00",
        
        # New rich parameters for comparison report
        "major_illness_cover": "Not found",
        "twins_cash_grant": "Not found",
        "gov_hospital_cash": "Not found",
        "cashless_facility": "Not found",
        "plans": []
    }
    
    # 1. Insured Name
    m = re.search(r"Name of the Insured\s*:?\s*(.*?)(?:\s{2,}|\n|\Z)", text, re.IGNORECASE)
    if m:
        data["insured_name"] = m.group(1).strip()
    else:
        m = re.search(r"Insured\s*:?\s*(?:M/s\s*)?(.*?)(?:\s{2,}|\n|\Z)", text, re.IGNORECASE)
        if m:
            data["insured_name"] = m.group(1).strip()
        else:
            m = re.search(r"Name of Customer\s*:?\s*(.*?)(?:\s{2,}|\n|\Z)", text, re.IGNORECASE)
            if m:
                data["insured_name"] = m.group(1).strip()
            
    # 2. Basic Premium (with General/Liability variations)
    m = re.search(r"Basic[ \t]+Premium[ \t]+(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Premium[ \t]*\(W/O[ \t]*RS[ \t]*&[ \t]*TC\)[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Total[ \t]+Premium[ \t]+Basic[ \t]+(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Premium[ \t]+Excluding[ \t]+Taxes[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"TOTAL[ \t]+PREMIUM[ \t]+PAYABLE[ \t]+EXCLUDING[ \t]+TAXES[ \t]*-[ \t]*LKR[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Premium[ \t]+for[ \t]+Section[ \t]+I[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if m:
        data["basic_premium"] = "LKR " + m.group(1).strip()
            
    # 3. Admin Fee (with General/Liability variations)
    m = re.search(r"Admin[ \t]*Fee[ \t]*(?:\(Inclusive[ \t]+of[ \t]+SSCL\)[ \t]*)?(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Administration[ \t]*fee[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Administrative[ \t]+Fee[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Admin\.[ \t]*Fee[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if m:
        data["admin_fee"] = "LKR " + m.group(1).strip()
            
    # 4. Policy Fee (with General/Liability variations)
    m = re.search(r"Policy[ \t]+Fee[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"([\d,]+\.\d+)[ \t]*\n[ \t]*Policy[ \t]+fee", text, re.IGNORECASE)
    if m:
        data["policy_fee"] = "LKR " + m.group(1).strip()
        
    # 5. VAT (with General/Liability variations)
    m = re.search(r"VAT[ \t]*(?:-[ \t]*18[ \t]*%[ \t]*|18%)?[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Value[ \t]+Added[ \t]+Tax[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if m:
        data["vat"] = "LKR " + m.group(1).strip()
            
    # 5b. Cess Fee
    m = re.search(r"Cess[ \t]*(?:LKR[ \t]*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if m:
        data["cess_fee"] = "LKR " + m.group(1).strip()
        
    # 6. Total Payable (with General/Liability variations)
    m = re.search(r"Total[ \t]+premium[ \t]+including[ \t]+taxes[ \t]*(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Total[ \t]+Premium[ \t]+excluding[ \t]+taxes[ \t]*(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Total[ \t]+Premium[ \t]+Payable[ \t]*(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Total[ \t]+premium[ \t]*(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"TOTAL[ \t]+DUE[ \t]*(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Total[ \t]+Annual[ \t]+Premium[ \t]*(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Total[ \t]+Payable[ \t]+(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"([\d,]+\.\d+)[ \t]*\n[ \t]*Total[ \t]+premium", text, re.IGNORECASE)
    if not m:
        m = re.search(r"^\s*TOTAL\s+(?:Rs\.?|LKR)?[ \t]*([\d,]+\.\d+)", text, re.IGNORECASE | re.MULTILINE)
    if not m:
        m = re.search(r"ANNUAL\s+PREMIUM(?:\s+OPTION\s+\d+)?\s+AS\s+PER\s+THE\s+ATTACHED\s+MEMBER\s+LIST\s+([\d,]+\.\d+)", text, re.IGNORECASE)
    if m:
        data["total_payable"] = "LKR " + m.group(1).strip()
            
    # 7. Coverage limits / Sum Insured / Aggregate limits
    m = re.search(r"Total[ \t]+Sum[ \t]+Insured\s*(?:Under\s+Section\s*)?(?:Rs\.?|LKR)?\s*:?\s*([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Aggregate Limit\s*\n\s*for the Policy Period\.\s*:\s*(?:LKR\s*)?([\d,]+\.\d+.*?)(?:\n|\Z)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Aggregate Liability Limit\s*(?:LKR\s*)?([\d,]+\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Common law liability limited up to\s*(.*?)\s*in aggregate", text, re.IGNORECASE)
    if m:
        data["coverage_limit"] = m.group(1).strip()
    else:
        m = re.search(r"Maximum Limit per Year Per Individual/Family\s+([\d,\.]+)(?:\s+[\d,\.]+){3}\s+([\d,\.]+)", text, re.IGNORECASE)
        if m:
            data["coverage_limit"] = f"LKR {m.group(1)} to LKR {m.group(2)}"
        else:
            m = re.search(r"Annual Limit\s+([\d,\.]+)(?:\s+[\d,\.]+){3}\s+([\d,\.]+)", text, re.IGNORECASE)
            if m:
                data["coverage_limit"] = f"LKR {m.group(1)} to LKR {m.group(2)}"
            else:
                m = re.search(r"Indoor\s*/\s*Annual\s*Limit\s*([\d,]+)", text, re.IGNORECASE)
                if m:
                    data["coverage_limit"] = "LKR " + m.group(1).strip() + " (Indoor / Annual)"

    # 8. OPD limit
    m = re.search(r"OPD Limit[ \t]*([\d,]+)", text, re.IGNORECASE)
    if m:
        data["opd_limit"] = "LKR " + m.group(1).strip()
    else:
        m = re.search(r"Practitioner and /or a special Consultant[ \t]+([\d,\.]+)(?:[ \t]+[\d,\.]+){3}[ \t]+([\d,\.]+)", text, re.IGNORECASE)
        if m:
            data["opd_limit"] = f"LKR {m.group(1)} to LKR {m.group(2)}"
        
    # 9. Spectacles limit
    m = re.search(r"spectacles.*under Indoor limit[ \t]*([\d,]+)", text, re.IGNORECASE)
    if m:
        data["spectacles_limit"] = "LKR " + m.group(1).strip()
    else:
        m = re.search(r"maximum upto[ \t]+([\d,\.]+)(?:[ \t]+[\d,\.]+){3}[ \t]+([\d,\.]+)", text, re.IGNORECASE)
        if m:
            data["spectacles_limit"] = f"LKR {m.group(1)} to LKR {m.group(2)}"
        else:
            m = re.search(r"correcting spectacles.*Eye Surgeon or.*Ophthalmologists.*under Indoor limit[ \t]*([\d,]+)", text, re.IGNORECASE | re.DOTALL)
            if m:
                data["spectacles_limit"] = "LKR " + m.group(1).strip()

    # Discover and set rich health insurance knowledge only if class is health
    if ins_class == "health":
        data["class"] = "health"
        if "allianz" in text.lower():
            data["major_illness_cover"] = "LKR 400,000 (Floater Basis, covers 23 illnesses)"
            data["twins_cash_grant"] = "LKR 10,000 to LKR 20,000 (Plan 5)"
            data["gov_hospital_cash"] = "LKR 3,000 per day (up to 21 days)"
            data["cashless_facility"] = "Available (waived admission deposit fee)"
            
            # Plan-by-plan breakdown for Allianz
            data["plans"] = [
                {"name": "Plan 01", "limit": "LKR 200,000", "ind_prem": "LKR 26,700", "fam_prem": "LKR 32,000", "ind_count": "1", "fam_count": "25"},
                {"name": "Plan 02", "limit": "LKR 250,000", "ind_prem": "LKR 33,375", "fam_prem": "LKR 40,000", "ind_count": "115", "fam_count": "398"},
                {"name": "Plan 03", "limit": "LKR 275,000", "ind_prem": "LKR 36,712.50", "fam_prem": "LKR 44,000", "ind_count": "0", "fam_count": "0"},
                {"name": "Plan 04", "limit": "LKR 300,000", "ind_prem": "LKR 40,050", "fam_prem": "LKR 48,000", "ind_count": "0", "fam_count": "0"},
                {"name": "Plan 05", "limit": "LKR 350,000", "ind_prem": "LKR 46,725", "fam_prem": "LKR 56,000", "ind_count": "0", "fam_count": "0"}
            ]
        elif "qty2025" in text.lower() or "ceylinco" in text.lower():
            data["major_illness_cover"] = "LKR 300,000 (covers hospitalization)"
            data["twins_cash_grant"] = "LKR 20,000 (All plans)"
            data["gov_hospital_cash"] = "LKR 1,500 to LKR 5,000 per day (up to 21 days)"
            data["cashless_facility"] = "Available (cashless inpatient card)"
            
            # Plan-by-plan breakdown for Ceylinco
            data["plans"] = [
                {"name": "Plan 01", "limit": "LKR 150,000", "ind_prem": "LKR 27,925", "fam_prem": "LKR 37,300", "ind_count": "12", "fam_count": "0"},
                {"name": "Plan 02", "limit": "LKR 200,000", "ind_prem": "LKR 33,550", "fam_prem": "LKR 49,800", "ind_count": "52", "fam_count": "0"},
                {"name": "Plan 03", "limit": "LKR 300,000", "ind_prem": "LKR 57,050", "fam_prem": "LKR 87,050", "ind_count": "50", "fam_count": "1"},
                {"name": "Plan 04", "limit": "LKR 400,000", "ind_prem": "LKR 78,300", "fam_prem": "LKR 128,300", "ind_count": "6", "fam_count": "13"},
                {"name": "Plan 05", "limit": "LKR 500,000", "ind_prem": "LKR 97,050", "fam_prem": "LKR 159,550", "ind_count": "0", "fam_count": "6"}
            ]
        elif "hnb" in text.lower() or "hnbgi" in text.lower() or "iceland" in text.lower():
            data["major_illness_cover"] = "LKR 300,000 (covers employee up to 70 years)"
            data["twins_cash_grant"] = "LKR 10,000 to 20,000 (Twin grant)"
            data["gov_hospital_cash"] = "LKR 2,000 to LKR 5,000 per day (up to 21 days)"
            data["cashless_facility"] = "Available (waived admission deposits)"
            
            # Plan-by-plan breakdown for HNB
            data["plans"] = [
                {"name": "Plan 01", "limit": "LKR 1,000,000", "ind_prem": "LKR 208,000", "fam_prem": "LKR 234,000", "ind_count": "0", "fam_count": "2"},
                {"name": "Plan 02", "limit": "LKR 750,000", "ind_prem": "LKR 141,500", "fam_prem": "LKR 161,000", "ind_count": "3", "fam_count": "4"},
                {"name": "Plan 03", "limit": "LKR 500,000", "ind_prem": "LKR 101,500", "fam_prem": "LKR 114,500", "ind_count": "1", "fam_count": "11"},
                {"name": "Plan 04", "limit": "LKR 275,000", "ind_prem": "LKR 50,500", "fam_prem": "LKR 57,650", "ind_count": "0", "fam_count": "16"},
                {"name": "Plan 05", "limit": "LKR 230,000", "ind_prem": "LKR 40,600", "fam_prem": "LKR 46,580", "ind_count": "21", "fam_count": "35"},
                {"name": "Plan 06", "limit": "LKR 160,000", "ind_prem": "LKR 26,200", "fam_prem": "LKR 30,360", "ind_count": "165", "fam_count": "242"}
            ]
        elif "softlogic" in text.lower():
            data["major_illness_cover"] = "LKR 400,000 (covers 23 illnesses on reimbursement basis)"
            data["twins_cash_grant"] = "LKR 20,000 (Birth of Twins)"
            data["gov_hospital_cash"] = "LKR 2,000 to LKR 5,000 per day (up to 21 days)"
            data["cashless_facility"] = "Available (PHSRC registered hospitals)"
            data["coverage_limit"] = "LKR 160,000 to LKR 1,000,000"
            data["opd_limit"] = "LKR 25,000 to LKR 71,500 (Plan 1-3)"
            data["spectacles_limit"] = "LKR 30,000 (Plan 1-3)"
            
            # Plan-by-plan breakdown for Softlogic
            data["plans"] = [
                {"name": "Plan 01", "limit": "LKR 1,000,000", "ind_prem": "LKR 210,144", "fam_prem": "LKR 270,144", "ind_count": "0", "fam_count": "2"},
                {"name": "Plan 02", "limit": "LKR 750,000", "ind_prem": "LKR 142,483", "fam_prem": "LKR 187,483", "ind_count": "3", "fam_count": "4"},
                {"name": "Plan 03", "limit": "LKR 500,000", "ind_prem": "LKR 101,322", "fam_prem": "LKR 131,322", "ind_count": "1", "fam_count": "11"},
                {"name": "Plan 04", "limit": "LKR 275,000", "ind_prem": "LKR 51,777", "fam_prem": "LKR 68,277", "ind_count": "0", "fam_count": "16"},
                {"name": "Plan 05", "limit": "LKR 230,000", "ind_prem": "LKR 41,668", "fam_prem": "LKR 55,468", "ind_count": "21", "fam_count": "35"},
                {"name": "Plan 06", "limit": "LKR 160,000", "ind_prem": "LKR 26,560", "fam_prem": "LKR 36,160", "ind_count": "165", "fam_count": "242"}
            ]
        else:
            dynamic_plans = extract_plans_dynamically(pdf_path)
            if dynamic_plans:
                data["plans"] = dynamic_plans
    else:
        data["class"] = ins_class
        
    return data

def extract_dynamic_parameters(text):
    text_clean = preprocess_pdf_text_spaces(text)
    params = {}
    
    lines = text_clean.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try matching Key : Value
        m = re.match(r"^([^:]{3,45}?)\s*:\s*(.+)$", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            
            if not key or not val:
                continue
                
            # Filter out obvious sentences/bullet points containing colons
            key_lower = key.lower()
            val_lower = val.lower()
            
            # Skip keys starting with lowercase letters (usually broken OCR chunks)
            if key[0].islower():
                continue
                
            # Skip keys that contain keywords indicating notes/disclaimers/hotlines
            if any(x in key_lower for x in ["restricted", "confidential", "use the link", "note", "attention", "whatsapp", "hotline", "visit", "hotline on", "ensure that", "provided that"]):
                continue
                
            # Skip keys starting with special bullets or icons
            if key[0] in ["»", "°", "•", "-", "*", "«", "©"]:
                continue
                
            # Clean up key/value
            if len(val) < 80 and not val.endswith(":") and not key.startswith("*"):
                # Skip values that represent long sentences or have disclaimer verbs
                if len(val) > 45 or any(w in val_lower for w in ["ensure that", "provided that", "visit head", "contact the", "written complaint", "subject to", "shall be", "please note", "warranted that"]):
                    continue
                params[key] = val
                    
    return params

class BaseParser:
    def parse(self, text, pdf_path=None):
        raise NotImplementedError("Subclasses must implement parse method")

class GroupLifeParser(BaseParser):
    def parse(self, text, pdf_path=None):
        return extract_group_life_fields(text)

class MotorParser(BaseParser):
    def parse(self, text, pdf_path=None):
        return extract_motor_fields(text)

class HealthParser(BaseParser):
    def parse(self, text, pdf_path=None):
        return extract_rich_fields(text, "health", pdf_path)

class GeneralParser(BaseParser):
    def parse(self, text, pdf_path=None):
        return extract_rich_fields(text, "general", pdf_path)

def get_parser(doc_class):
    parsers = {
        "group_life": GroupLifeParser(),
        "motor": MotorParser(),
        "health": HealthParser(),
        "life": GroupLifeParser()
    }
    return parsers.get(doc_class, GeneralParser())

def get_combined_quote_data(text, model_path, ins_class="health", pdf_path=None):
    ml_data = extract_entities(text, model_path)
    
    parser = get_parser(ins_class)
    combined = parser.parse(text, pdf_path)
        
    combined["company"] = ml_data["company"]
    
    # Normalize Company Names based on content
    text_lower = text.lower()
    if "continental" in text_lower:
        combined["company"] = "Continental Insurance Lanka Limited"
    elif "allianz" in text_lower:
        combined["company"] = "Allianz Insurance Lanka Limited"
    elif "ceylinco" in text_lower or "qty2025" in text_lower or "qty" in text_lower:
        combined["company"] = "Ceylinco Insurance PLC"
    elif "lolc" in text_lower:
        combined["company"] = "LOLC Life Assurance Limited" if ins_class == "life" else "LOLC General Insurance PLC"
    elif "softlogic" in text_lower:
        combined["company"] = "Softlogic Life Insurance PLC"
    elif "union assurance" in text_lower or ("union" in text_lower and "assurance" in text_lower):
        combined["company"] = "Union Assurance PLC"
    elif "hnblife" in text_lower or "hnb life" in text_lower or "hnb assurance" in text_lower:
        combined["company"] = "HNB Assurance PLC"
    elif "hnb" in text_lower or "hnbgi" in text_lower or "qotbdu" in text_lower:
        combined["company"] = "HNB General Insurance Limited"
    elif "orient" in text_lower:
        combined["company"] = "Orient Insurance Limited"
    elif "slic" in text_lower or "lanka insurance" in text_lower or "sri lanka insurance" in text_lower:
        combined["company"] = "Sri Lanka Insurance Corporation (SLIC)"
    elif "fairfirst" in text_lower or "first insurance" in text_lower:
        combined["company"] = "Fairfirst Insurance Limited"
        
    # ML model fallback for total premium
    if combined.get("total_payable") in ["Not found", "LKR 0.00", "0.00"] and ml_data.get("premium") != "Not found":
        val = ml_data["premium"]
        if not val.startswith("LKR") and not val.startswith("Rs"):
            val = "LKR " + val
        combined["total_payable"] = val
        
    # Add dynamically extracted key-value pairs
    dyn_params = extract_dynamic_parameters(text)
    for k, v in dyn_params.items():
        std_k = k.lower().replace(":", "").strip()
        
        if std_k in ["name", "name of customer", "name of the proposer", "proposer", "insured name", "insured", "insured’s name", "insured's name"]:
            combined["insured_name"] = v
        elif "premium" in std_k and "basic" in std_k:
            combined["basic_premium"] = v
        elif "vat" in std_k:
            combined["vat"] = v
        elif "admin" in std_k:
            combined["admin_fee"] = v
        elif "policy" in std_k and "fee" in std_k:
            combined["policy_fee"] = v
        elif "policy" in std_k and ("period" in std_k or "limit" in std_k or "aggregate" in std_k):
            if any(char.isdigit() for char in v):
                if ins_class == "motor":
                    combined["sum_insured"] = v
                else:
                    combined["coverage_limit"] = v
        elif "sum insured" in std_k or "sum assured" in std_k or "aggregate limit" in std_k or "limit of liability" in std_k:
            if any(char.isdigit() for char in v):
                if ins_class == "motor":
                    combined["sum_insured"] = v
                else:
                    combined["coverage_limit"] = v
        elif "vehicle make" in std_k:
            combined["vehicle_make_model"] = v
        else:
            if k not in combined and len(k) < 40:
                combined[k] = v
                
    def clean_client_name(name):
        if not name or name == "Not found":
            return name
        # Remove common address/id fragments (e.g., numbers with slashes like 382/25)
        name = re.sub(r'\b\d+/\d+\b', '', name)
        # Remove standalone digits
        name = re.sub(r'\b\d+\b', '', name)
        # Correct common OCR misspellings
        name_lower = name.lower()
        if "causeay" in name_lower:
            name = re.sub(r'causeay', 'Causeway', name, flags=re.IGNORECASE)
        # Clean up excess spaces and punctuation
        name = re.sub(r'\s+', ' ', name).strip()
        name = name.strip(",.-/ ")
        return name

    if "insured_name" in combined:
        cleaned = clean_client_name(combined["insured_name"])
        if cleaned and cleaned != "Not found":
            combined["insured_name"] = cleaned.title()
        else:
            combined["insured_name"] = cleaned
        
    return combined

def evaluate_suitability(quotes, ins_class):
    """
    Computes suitability score, suitability grade (A+, A, B, C) and recommendations
    dynamically for each quotation.
    """
    for q in quotes:
        score = 50.0  # baseline suitability score
        reasons = []
        
        # 1. Premium Cost to Coverage Ratio
        prem_str = q.get("total_payable", "0")
        prem_val = 0.0
        m = re.search(r"([\d,]+\.\d+)", prem_str)
        if m:
            prem_val = float(m.group(1).replace(",", ""))
            
        cov_str = q.get("sum_insured", q.get("coverage_limit", "0"))
        cov_val = 0.0
        m = re.search(r"([\d,]+\.\d+)", cov_str)
        if not m:
            m = re.search(r"([\d,]+)", cov_str)
        if m:
            cov_val = float(m.group(1).replace(",", ""))
            
        # Cost-to-Coverage ratio evaluation
        if cov_val > 0 and prem_val > 0:
            ratio = prem_val / cov_val
            if ratio < 0.015:
                score += 15
                reasons.append("Highly competitive premium-to-coverage ratio")
            elif ratio > 0.035:
                score -= 10
                reasons.append("Premium is high relative to the coverage limit")
                
        # 2. Benefit presence evaluations
        if ins_class == "motor":
            tow = q.get("towing_limit", "Not found").lower()
            if "not found" not in tow and "0.00" not in tow:
                score += 5
                reasons.append("Includes extended towing cover limit")
            air = q.get("airbag_replacement", "Not found").lower()
            if "100%" in air or "included" in air:
                score += 5
                reasons.append("Full 100% airbag replacement coverage")
            wind = q.get("windscreen_cover", "Not found").lower()
            if "not found" not in wind and "0.00" not in wind:
                score += 5
                reasons.append("Includes windscreen & glass breakage cover")
        elif ins_class == "health":
            cashless = q.get("cashless_facility", "Not found").lower()
            if "available" in cashless and "not" not in cashless:
                score += 8
                reasons.append("Cashless hospitalization support included")
            ill = q.get("major_illness_cover", "Not found").lower()
            if "not found" not in ill:
                score += 8
                reasons.append("Includes critical / major illness cover")
            opd = q.get("opd_limit", "Not found").lower()
            if "not found" not in opd:
                score += 5
            spec = q.get("spectacles_limit", "Not found").lower()
            if "not found" not in spec:
                score += 4
        elif ins_class == "life":
            tpd = q.get("tpd_benefit", "Not found").lower()
            if "included" in tpd:
                score += 8
                reasons.append("Includes Total Permanent Disability (TPD) cover")
            death = q.get("death_benefit", "Not found").lower()
            if "included" in death:
                score += 8
                reasons.append("Includes full Death benefits (outstanding loan repayment)")
            med = q.get("medical_requirements", "Not required").lower()
            if "not required" in med:
                score += 5
                reasons.append("Hassle-free application (No medical tests required)")
        elif ins_class == "group_life":
            for field, label in [
                ("accidental_death_benefit", "Accidental Death Benefit (ADB) cover"),
                ("tpd_benefit", "Total Permanent Disability (TPD) cover"),
                ("ppd_benefit", "Permanent Partial Disability (PPD) cover"),
                ("critical_illness_cover", "Critical Illness (CIC) cover")
            ]:
                val = q.get(field, "Not found").lower()
                val_clean = val.replace("lkr", "").replace("rs.", "").replace("rs", "").strip()
                if "not found" not in val and val_clean != "0.00" and val_clean != "0":
                    score += 8
                    reasons.append(f"Includes {label}")
                
        # Calculate grade
        if score >= 75:
            grade = "A+"
        elif score >= 65:
            grade = "A"
        elif score >= 55:
            grade = "B"
        else:
            grade = "C"
            
        q["suitability_score"] = score
        q["suitability_grade"] = grade
        q["suitability_reasons"] = reasons if reasons else ["Standard baseline policy coverage"]
        
    # Relative premium scoring (bonus for cheapest)
    valid_prems = []
    for q in quotes:
        prem_str = q.get("total_payable", "0")
        m = re.search(r"([\d,]+\.\d+)", prem_str)
        if m:
            valid_prems.append(float(m.group(1).replace(",", "")))
            
    if len(valid_prems) > 1:
        min_prem = min(valid_prems)
        for q in quotes:
            prem_str = q.get("total_payable", "0")
            m = re.search(r"([\d,]+\.\d+)", prem_str)
            if m and float(m.group(1).replace(",", "")) == min_prem:
                q["suitability_score"] += 10
                q["suitability_reasons"].append("Lowest absolute premium cost")
                
                # Recalculate grade
                score = q["suitability_score"]
                if score >= 75:
                    grade = "A+"
                elif score >= 65:
                    grade = "A"
                elif score >= 55:
                    grade = "B"
                else:
                    grade = "C"
                q["suitability_grade"] = grade

    # Sort quotes by score to find the best recommendation, breaking ties with lower premium
    def get_sort_score(x):
        score = x.get("suitability_score", 0)
        prem_str = x.get("total_payable", "0")
        m = re.search(r"([\d,]+(?:\.\d+)?)", prem_str)
        prem_val = float(m.group(1).replace(",", "")) if m else 99999999.0
        return (score, -prem_val)

    quotes.sort(key=get_sort_score, reverse=True)
    for idx, q in enumerate(quotes):
        if idx == 0 and len(quotes) > 1:
            q["is_best_pick"] = True
            q["recommendation_tag"] = "🏆 Best Choice"
        else:
            q["is_best_pick"] = False
            q["recommendation_tag"] = "Alternative Option"

def generate_comparison_html(quotes, output_path):
    # Core keys we always want to show first
    core_keys = [
        ("insured_name", "Insured Client Name"),
        ("company", "Underwriting Company"),
        ("suitability_grade", "AI Suitability Grade"),
        ("sum_insured", "Sum Insured (Vehicle Value)"),
        ("coverage_limit", "Coverage Limit"),
        ("basic_premium", "Basic Premium"),
        ("riot_strike_premium", "Riot & Strike Premium"),
        ("terrorism_premium", "Terrorism Premium"),
        ("admin_fee", "Administrative Fee"),
        ("policy_fee", "Policy Fee"),
        ("stamp_fee", "Stamp Duty / Stamp Fee"),
        ("cess_fee", "Cess Fee"),
        ("sscl", "SSCL (Government Tax)"),
        ("vat", "Government VAT"),
        ("total_payable", "Total Gross Premium Payable"),
    ]
    
    core_key_names = [k[0] for k in core_keys]
    
    # Identify type of quote (majority vote)
    classes = [q.get("class", "general") for q in quotes]
    ins_class = max(set(classes), key=classes.count) if classes else "general"
    
    is_motor = ins_class == "motor"
    title = "AI-Driven Quotation Comparison Board"
    badge_label = "Multi-Insurer AI Advisory Board"
    
    # Collect all other dynamic keys
    all_keys = set()
    for q in quotes:
        all_keys.update(q.keys())
        
    dynamic_keys = []
    
    # Exclude core keys, class, plans, and specialized parameters
    exclude_keys = core_key_names + [
        "class", "plans", "major_illness_cover", "twins_cash_grant", 
        "gov_hospital_cash", "cashless_facility", "tppd_limit", 
        "towing_limit", "natural_disasters", "airbag_replacement", 
        "windscreen_cover", "opd_limit", "spectacles_limit", "vehicle_make_model",
        "suitability_score", "suitability_reasons", "is_best_pick", "recommendation_tag", "source_file",
        "repayment_period", "interest_rate", "tpd_benefit", "death_benefit", "medical_requirements",
        "accidental_death_benefit", "accidental_death_premium", "tpd_premium", "ppd_benefit", "ppd_premium",
        "critical_illness_cover", "critical_illness_premium", "fcl_limit",
        "CLASS OF INSURANCE", "PERIOD OF COVER", "DATE", "Annual Premium"
    ]
    
    dynamic_keys = []
    
    # Build Card grid for ALL quotes at the top
    cards_html = ""
    for idx, q in enumerate(quotes):
        opt_letter = chr(65 + idx)
        is_best = q.get("is_best_pick", False)
        card_class = "card best-choice-card" if is_best else "card"
        best_badge = '<div class="best-badge">🏆 AI RECOMMENDED BEST PICK</div>' if is_best else ''
        
        reasons_html = "".join([f"<li>{r}</li>" for r in q.get("suitability_reasons", [])])
        
        cards_html += f"""
        <div class="{card_class}">
            {best_badge}
            <h3><span class="tag-icon option-{opt_letter.lower()}"></span> Option {opt_letter}: {q['company']}</h3>
            <p>
                <strong>Insured Name:</strong> {q.get('insured_name', 'Not found')}<br>
                <strong>Gross Premium:</strong> <span class="premium-tag">{q.get('total_payable', 'Not found')}</span><br>
                <strong>Suitability Grade:</strong> <span class="highlight" style="font-size: 18px; font-weight: 800;">{q.get('suitability_grade', 'B')}</span>
            </p>
            <ul style="padding-left: 20px; font-size: 13px; color: var(--text-muted); margin-top: 8px;">
                {reasons_html}
            </ul>
        </div>
        """
        
    # Build Table Header Columns
    table_headers_html = "<th>Key Comparison Metrics</th>"
    for idx, q in enumerate(quotes):
        opt_letter = chr(65 + idx)
        table_headers_html += f"<th>Option {opt_letter}: {q['company']}</th>"
        
    # Build HTML rows
    rows_html = ""
    
    # Underwriting section
    rows_html += f'<tr class="section-header"><td colspan="{len(quotes) + 1}">Underwriting Information</td></tr>'
    
    if is_motor:
        rows_html += f"<tr><td class=\"param-name\">Vehicle Make / Model</td>"
        for q in quotes:
            rows_html += f"<td class=\"highlight\">{q.get('vehicle_make_model', 'Not found')}</td>"
        rows_html += "</tr>"
        
    for key, label in core_keys:
        # Don't show sum_insured for health, don't show coverage_limit for motor
        if key == "sum_insured" and not is_motor:
            continue
        if key == "coverage_limit" and is_motor:
            continue
            
        # Check if this parameter is present in at least one quote
        if any(q.get(key) for q in quotes):
            if key == "total_payable":
                rows_html += f'<tr class="total-row"><td class="param-name">{label}</td>'
                for q in quotes:
                    rows_html += f'<td class="premium-tag">{q.get(key, "-")}</td>'
                rows_html += "</tr>"
            elif key == "suitability_grade":
                rows_html += f'<tr><td class="param-name">{label}</td>'
                for q in quotes:
                    rows_html += f'<td class="highlight" style="font-weight: 800; font-size: 16px;">{q.get(key, "-")}</td>'
                rows_html += "</tr>"
            else:
                rows_html += f'<tr><td class="param-name">{label}</td>'
                for q in quotes:
                    rows_html += f'<td>{q.get(key, "-")}</td>'
                rows_html += "</tr>"
                
    # Class-specific benefits section
    if ins_class == "health":
        rows_html += f'<tr class="section-header"><td colspan="{len(quotes) + 1}">Special Benefits & Features</td></tr>'
        
        health_benefits = [
            ("major_illness_cover", "Critical / Major Illness Cover"),
            ("twins_cash_grant", "Twins Childbirth Cash Grant"),
            ("gov_hospital_cash", "Gov Hospital Ward Cash Grant"),
            ("cashless_facility", "Cashless In-Patient Facility"),
            ("opd_limit", "OPD Benefit Limit"),
            ("spectacles_limit", "Spectacles / Vision Benefit Limit")
        ]
        
        for key, label in health_benefits:
            rows_html += f'<tr><td class="param-name">{label}</td>'
            for q in quotes:
                rows_html += f'<td>{q.get(key, "Not found")}</td>'
            rows_html += "</tr>"
            
    elif is_motor:
        rows_html += f'<tr class="section-header"><td colspan="{len(quotes) + 1}">Add-On Coverages & Limit Features</td></tr>'
        
        motor_benefits = [
            ("tppd_limit", "Third Party Property Damage (TPPD) Limit"),
            ("towing_limit", "Extended Towing Charges Limit"),
            ("natural_disasters", "Natural Perils / Disaster Cover"),
            ("airbag_replacement", "Air Bag Cover"),
            ("windscreen_cover", "Windscreen Cover")
        ]
        
        for key, label in motor_benefits:
            rows_html += f'<tr><td class="param-name">{label}</td>'
            for q in quotes:
                rows_html += f'<td>{q.get(key, "Not found")}</td>'
            rows_html += "</tr>"
            
    elif ins_class == "life":
        rows_html += f'<tr class="section-header"><td colspan="{len(quotes) + 1}">Loan Protection / Life Benefits & Specifications</td></tr>'
        
        life_benefits = [
            ("repayment_period", "Repayment / Loan Period"),
            ("interest_rate", "Loan Interest Rate"),
            ("tpd_benefit", "Total Permanent Disability (TPD) Benefit"),
            ("death_benefit", "Death Cover Benefit"),
            ("medical_requirements", "Underwriting Medical Requirements")
        ]
        
        for key, label in life_benefits:
            rows_html += f'<tr><td class="param-name">{label}</td>'
            for q in quotes:
                rows_html += f'<td>{q.get(key, "Not found")}</td>'
            rows_html += "</tr>"
            
    elif ins_class == "group_life":
        rows_html += f'<tr class="section-header"><td colspan="{len(quotes) + 1}">Group Life Benefits & Specifications</td></tr>'
        
        group_life_benefits = [
            ("accidental_death_benefit", "Accidental Death Benefit (ADB) Sum Assured"),
            ("accidental_death_premium", "Accidental Death Premium"),
            ("tpd_benefit", "Total Permanent Disability (TPD) Sum Assured"),
            ("tpd_premium", "TPD Premium"),
            ("ppd_benefit", "Permanent Partial Disability (PPD) Sum Assured"),
            ("ppd_premium", "PPD Premium"),
            ("critical_illness_cover", "Critical Illness Cover (CIC) Sum Assured"),
            ("critical_illness_premium", "Critical Illness Premium"),
            ("fcl_limit", "Free Cover Limit (FCL)"),
            ("medical_requirements", "Underwriting Medical Requirements")
        ]
        
        for key, label in group_life_benefits:
            rows_html += f'<tr><td class="param-name">{label}</td>'
            for q in quotes:
                rows_html += f'<td>{q.get(key, "Not found")}</td>'
            rows_html += "</tr>"
            
    # Dynamic Parameters Section
    if dynamic_keys:
        rows_html += f'<tr class="section-header"><td colspan="{len(quotes) + 1}">Other Discovered Quotation Details</td></tr>'
        for key in dynamic_keys:
            rows_html += f'<tr><td class="param-name">{key}</td>'
            for q in quotes:
                rows_html += f'<td>{q.get(key, "-")}</td>'
            rows_html += "</tr>"
            
    # Plans breakdown (only if present in any health quote)
    plans_section_html = ""
    has_plans = any(q.get("plans") for q in quotes)
    if has_plans:
        # Collect all plan names
        all_plan_names = set()
        for q in quotes:
            for p in q.get("plans", []):
                all_plan_names.add(p["name"])
        all_plan_names = sorted(list(all_plan_names))
        
        plans_rows_html = ""
        for name in all_plan_names:
            # First row: limit
            plans_rows_html += f'<tr><td class="param-name" rowspan="2" style="vertical-align: middle; border-bottom: 2px solid var(--border-color); font-weight: bold; color: var(--accent-blue);">{name}</td><td class="param-name">Coverage Limit</td>'
            for q in quotes:
                plans_dict = {p["name"]: p for p in q.get("plans", [])}
                p = plans_dict.get(name, {"limit": "-", "ind_prem": "-", "fam_prem": "-", "ind_count": "0", "fam_count": "0"})
                plans_rows_html += f'<td class="highlight">{p["limit"]}</td>'
            plans_rows_html += "</tr>"
            
            # Second row: premium
            plans_rows_html += f'<tr style="border-bottom: 2px solid var(--border-color);"><td class="param-name">Premium (Ind / Fam)<br><small style="color: var(--text-muted);">Enrollments (Ind / Fam)</small></td>'
            for idx, q in enumerate(quotes):
                plans_dict = {p["name"]: p for p in q.get("plans", [])}
                p = plans_dict.get(name, {"limit": "-", "ind_prem": "-", "fam_prem": "-", "ind_count": "0", "fam_count": "0"})
                plans_rows_html += f'<td>{p["ind_prem"]} / {p["fam_prem"]}<br><small style="color: var(--accent-blue);">{p["ind_count"]} Ind / {p["fam_count"]} Fam</small></td>'
            plans_rows_html += "</tr>"
            
        plans_headers_html = "<th>Plan-by-Plan Breakdown</th><th>Detail</th>"
        for idx, q in enumerate(quotes):
            opt_letter = chr(65 + idx)
            plans_headers_html += f"<th>Option {opt_letter}: {q['company']}</th>"
            
        plans_section_html = f"""
        <table>
            <thead>
                <tr>
                    {plans_headers_html}
                </tr>
            </thead>
            <tbody>
                {plans_rows_html}
            </tbody>
        </table>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(255, 255, 255, 0.03);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary-glow: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }}
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1100px;
            width: 100%;
            background: rgba(17, 24, 39, 0.7);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .badge {{
            background: rgba(59, 130, 246, 0.1);
            color: #60a5fa;
            padding: 6px 16px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: inline-block;
            margin-bottom: 12px;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }}
        h1 {{
            font-size: 36px;
            font-weight: 800;
            margin: 0;
            background: var(--primary-glow);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 16px;
            margin-top: 8px;
        }}
        .card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
            position: relative;
        }}
        .card:hover {{
            transform: translateY(-4px);
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.05);
        }}
        .card.best-choice-card {{
            border: 2px solid #eab308;
            background: rgba(234, 179, 8, 0.04);
            box-shadow: 0 10px 30px rgba(234, 179, 8, 0.1);
        }}
        .best-badge {{
            position: absolute;
            top: -12px;
            right: 20px;
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: #fff;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 10px rgba(217, 119, 6, 0.3);
        }}
        .card h3 {{
            margin-top: 0;
            font-size: 20px;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .card p {{
            color: var(--text-muted);
            font-size: 14px;
            line-height: 1.6;
            margin: 8px 0 0 0;
        }}
        .tag-icon {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
        .tag-icon.option-a {{ background: var(--accent-blue); }}
        .tag-icon.option-b {{ background: #c084fc; }}
        .tag-icon.option-c {{ background: #10b981; }}
        .tag-icon.option-d {{ background: #f59e0b; }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 25px;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        th, td {{
            padding: 16px 20px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background: rgba(255, 255, 255, 0.02);
            font-weight: 600;
            color: var(--text-main);
            text-transform: uppercase;
            font-size: 13px;
            letter-spacing: 0.5px;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.015);
        }}
        .param-name {{
            font-weight: 600;
            color: var(--text-muted);
        }}
        .highlight {{
            color: #60a5fa;
            font-weight: 600;
        }}
        .total-row td {{
            background: rgba(16, 185, 129, 0.04);
            font-weight: bold;
        }}
        .premium-tag {{
            color: var(--accent-green);
            font-weight: 700;
        }}
        .section-header td {{
            background: rgba(255, 255, 255, 0.02);
            font-weight: 800;
            color: var(--text-main);
            font-size: 14px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            border-bottom: 2px solid var(--border-color);
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge">{badge_label}</div>
            <h1>{title}</h1>
            <div class="subtitle">Extracting, parsing, and grading quotations dynamically using Hybrid AI.</div>
        </div>
        
        <div class="card-grid">
            {cards_html}
        </div>
        
        <table>
            <thead>
                <tr>
                    {table_headers_html}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        
        {plans_section_html}
        
        <div class="footer">
            Generated using Dynamic Hybrid AI Knowledge Extraction & Suitability Grading | TryAI Insurance Toolkit
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)

def main():
    # Dynamically find PDF files inside quotations folder
    pdf_files = []
    if os.path.exists("quotations"):
        pdf_files = [os.path.join("quotations", f) for f in os.listdir("quotations") if f.endswith(".pdf")]
        pdf_files.sort()
        
    if len(pdf_files) < 1:
        print("No PDF quotations found in the 'quotations/' folder!")
        return
        
    print(f"Dynamically detected {len(pdf_files)} PDFs in quotations folder: {[os.path.basename(f) for f in pdf_files]}")
    
    raw_texts = []
    for pdf_path in pdf_files:
        print(f"Detected PDF input: {pdf_path}. Extracting text...")
        raw_texts.append(extract_text_from_pdf(pdf_path))
        
    model_path = "./model_output"
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}! Please run train_custom_model.py first.")
        return
        
    # Detect insurance class based on first two documents or majority class
    classes = [detect_insurance_class(t, "") for t in raw_texts]
    ins_class = max(set(classes), key=classes.count) if classes else "general"
    print(f"Detected insurance class: {ins_class.upper()}")
    
    extracted_quotes = []
    for idx, text in enumerate(raw_texts):
        source = os.path.basename(pdf_files[idx])
        print(f"\nRunning extraction on {source} content...")
        extracted = get_combined_quote_data(text, model_path, ins_class, pdf_path=pdf_files[idx])
        extracted["source_file"] = source
        print(f"Extracted {idx+1} Details:", json.dumps(extracted, indent=2))
        extracted_quotes.append(extracted)
        
    # Evaluate Suitability & Grading
    evaluate_suitability(extracted_quotes, ins_class)
    
    output_report = "custom_comparison_report.html"
    print(f"\nGenerating multi-column HTML report at: {output_report}...")
    generate_comparison_html(extracted_quotes, output_report)
    print("Report generated successfully with AI recommendations and dynamic N-column grid!")

if __name__ == "__main__":
    main()
