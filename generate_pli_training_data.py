"""
generate_pli_training_data.py
Generates PLI NER training samples and appends to train_data.json / val_data.json.
Run:  python3 generate_pli_training_data.py
"""
import json, os, random

TRAIN_FILE = "train_data.json"
VAL_FILE   = "val_data.json"
VAL_RATIO  = 0.15

PLI_TEMPLATES = [
    # Company
    ("Fairfirst Insurance Limited hereby offers Public Liability Insurance coverage.",
     {"Fairfirst Insurance Limited": "COMPANY"}),
    ("HNB General Insurance Limited Quotation No QOTBDUMIS202629819 for Public Liability.",
     {"HNB General Insurance Limited": "COMPANY"}),
    ("Sri Lanka Insurance Corporation offers PLI at competitive rates.",
     {"Sri Lanka Insurance Corporation": "COMPANY"}),
    ("Ceylinco Insurance PLC Public Liability Quotation for United Tractors.",
     {"Ceylinco Insurance PLC": "COMPANY"}),
    ("Allianz Insurance Lanka Limited issues this Public Liability quotation.",
     {"Allianz Insurance Lanka Limited": "COMPANY"}),
    # Gross Premium / Basic Premium
    ("The Gross Premium for Public Liability Insurance is LKR 130,000.00 per annum.",
     {"LKR 130,000.00": "PREMIUM"}),
    ("Annual Gross Premium LKR 202,629.00 excluding VAT and government levies.",
     {"LKR 202,629.00": "PREMIUM"}),
    ("Basic Premium LKR 85,500.00 for the policy period of one year.",
     {"LKR 85,500.00": "PREMIUM"}),
    ("Net Premium payable LKR 161,286.54 inclusive of all applicable taxes.",
     {"LKR 161,286.54": "PREMIUM"}),
    ("Total Annual Premium: LKR 75,000.00 subject to the schedule of benefits.",
     {"LKR 75,000.00": "PREMIUM"}),
    ("Public Liability Gross Premium LKR 45,000.00 per year for two locations.",
     {"LKR 45,000.00": "PREMIUM"}),
    ("The quoted annual premium is LKR 55,000.00 for this public liability policy.",
     {"LKR 55,000.00": "PREMIUM"}),
    # Deductible / Excess
    ("A compulsory excess of LKR 25,000 applies per claim under this policy.",
     {"LKR 25,000": "DEDUCTIBLE"}),
    ("Deductible LKR 10,000 per occurrence applicable to all third party claims.",
     {"LKR 10,000": "DEDUCTIBLE"}),
    ("The policy carries a loss retention of LKR 50,000 each and every event.",
     {"LKR 50,000": "DEDUCTIBLE"}),
    ("Excess LKR 5,000 per claim is applicable to property damage claims only.",
     {"LKR 5,000": "DEDUCTIBLE"}),
    ("Compulsory excess of LKR 15,000 per accident applies for bodily injury claims.",
     {"LKR 15,000": "DEDUCTIBLE"}),
    ("A deductible of LKR 20,000 per claim will be borne by the insured.",
     {"LKR 20,000": "DEDUCTIBLE"}),
]

def tokenize_and_tag(sentence, entities):
    tokens = sentence.split()
    tags   = ["O"] * len(tokens)
    for phrase, label in entities.items():
        pt = phrase.split()
        n  = len(pt)
        for i in range(len(tokens) - n + 1):
            win = tokens[i:i+n]
            if [t.strip(".,;:").lower() for t in win] == [p.strip(".,;:").lower() for p in pt]:
                tags[i] = f"B-{label}"
                for j in range(1, n):
                    tags[i+j] = f"I-{label}"
                break
    return {"tokens": tokens, "ner_tags": tags}

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except: return []
    return []

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

if __name__ == "__main__":
    samples = [tokenize_and_tag(s, e) for s, e in PLI_TEMPLATES]
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * VAL_RATIO))
    val_s, train_s = samples[:n_val], samples[n_val:]
    train_data = load_json(TRAIN_FILE); train_data.extend(train_s); save_json(TRAIN_FILE, train_data)
    val_data   = load_json(VAL_FILE);   val_data.extend(val_s);     save_json(VAL_FILE,   val_data)
    print(f"Added {len(train_s)} train + {len(val_s)} val PLI samples.")
    print(f"Total: {len(train_data)} train, {len(val_data)} val")
    print("Next: python3 train_custom_model.py")
