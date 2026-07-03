import json, os, random

TRAIN_FILE = "train_data.json"
VAL_FILE = "val_data.json"

def make_sample(sentence, annotations):
    tokens = sentence.split()
    ner_tags = ["O"] * len(tokens)
    annotations = sorted(annotations, key=lambda x: len(x[0]), reverse=True)
    for phrase, label in annotations:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        for i in range(len(tokens) - n + 1):
            window = tokens[i:i+n]
            window_clean = [t.strip(".,;:()").lower() for t in window]
            phrase_clean = [p.strip(".,;:()").lower() for p in phrase_tokens]
            if window_clean == phrase_clean:
                ner_tags[i] = f"B-{label}"
                for j in range(1, n):
                    ner_tags[i+j] = f"I-{label}"
                break
    return {"tokens": tokens, "ner_tags": ner_tags}

REAL_SAMPLES = [
    # Amana Takaful
    make_sample(
        "Amana Takaful PLC (PQ23) No. 660-1/1, Galle Road, Colombo 03.",
        [("Amana Takaful PLC", "COMPANY")]
    ),
    make_sample(
        "Proposer's Name : Choice Park Pvt Ltd",
        []
    ),
    make_sample(
        "Basic Contribution : 37,382.80 Strike & Riot Premium : 48,046.86",
        [("37,382.80", "PREMIUM"), ("48,046.86", "PREMIUM")]
    ),
    make_sample(
        "Terrorism Premium : 36,035.15 Administrative Charges : 6,149.16 VAT : 22,970.51",
        [("36,035.15", "PREMIUM")]
    ),
    make_sample(
        "Total : 150,584.49 Without TC Total : 105,910.36",
        [("150,584.49", "PREMIUM"), ("105,910.36", "PREMIUM")]
    ),
    make_sample(
        "Deductible / Excess : Riot,Strike : 10% on each and every loss",
        []
    ),
    # People's Insurance
    make_sample(
        "PEOPLE’S INSURANCE PLC (Company No. PB3754PQ) Havelock Road",
        [("PEOPLE’S INSURANCE PLC", "COMPANY")]
    ),
    make_sample(
        "Insured: Choice Park Pvt Ltd",
        []
    ),
    make_sample(
        "Basic Premium LKR 61,957.97 Riot &Strike LKR 29,391.39",
        [("LKR 61,957.97", "PREMIUM"), ("LKR 29,391.39", "PREMIUM")]
    ),
    make_sample(
        "Terrorism LKR 36,739.24 Policy Fees LKR 500.00 Admin Fee LKR 3,685.97",
        [("LKR 36,739.24", "PREMIUM")]
    ),
    make_sample(
        "Total Amount Payable LKR 156,216.99",
        [("LKR 156,216.99", "PREMIUM")]
    ),
    make_sample(
        "All Other Claims - 10% with a minimum of LKR 10,000.00 deductible.",
        [("LKR 10,000.00", "DEDUCTIBLE")]
    ),
    make_sample(
        "subject to an excess of 15% with a minimum of LKR 25,000.00 on each and every claim.",
        [("LKR 25,000.00", "DEDUCTIBLE")]
    )
]

def main():
    print("Loading datasets...")
    with open(TRAIN_FILE, "r") as f:
        train_data = json.load(f)
    with open(VAL_FILE, "r") as f:
        val_data = json.load(f)
        
    print(f"Original - Train: {len(train_data)}, Val: {len(val_data)}")
    
    augmented = []
    for s in REAL_SAMPLES:
        for _ in range(15):
            augmented.append(s)
            
    random.shuffle(augmented)
    split_idx = int(len(augmented) * 0.8)
    train_add = augmented[:split_idx]
    val_add = augmented[split_idx:]
    
    train_data.extend(train_add)
    val_data.extend(val_add)
    
    with open(TRAIN_FILE, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(VAL_FILE, "w") as f:
        json.dump(val_data, f, indent=2)
    print("Added new real Fire quote samples successfully!")

if __name__ == "__main__":
    main()
