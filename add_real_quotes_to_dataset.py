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
    make_sample(
        "FAIRFIRST INSURANCE LIMITED General Underwriting",
        [("FAIRFIRST INSURANCE LIMITED", "COMPANY")]
    ),
    make_sample(
        "Name of the Proposer : United Tractors & Equipments Private Limited",
        []
    ),
    make_sample(
        "Basic LKR 130,250.00 Cess LKR 390.75 Stamp LKR 131.00",
        [("LKR 130,250.00", "PREMIUM")]
    ),
    make_sample(
        "Total Premium LKR 161,286.54",
        [("LKR 161,286.54", "PREMIUM")]
    ),
    make_sample(
        "Excesses on each and every loss : 10% with a minimum of LKR. 25,000/- in respect of third party property damage only",
        [("LKR. 25,000/-", "DEDUCTIBLE")]
    ),
    make_sample(
        "Subject to a compulsory excess of LKR 25,000 per claim.",
        [("LKR 25,000", "DEDUCTIBLE")]
    ),
    make_sample(
        "HNB General Insurance Limited (Reg. # PB 5167)",
        [("HNB General Insurance Limited", "COMPANY")]
    ),
    make_sample(
        "QUOTATION NO : QOTBDUMIS202629819",
        []
    ),
    make_sample(
        "NAME OF THE INSURED : UNITED TRACTORS & EQUIPMENT'S PRIVATE LIMITED",
        []
    ),
    make_sample(
        "Annual Premium : As per the attached annexure 01",
        []
    ),
    make_sample(
        "Deductible : All claims: 10% or LKR 35,000/- whichever is higher.",
        [("LKR 35,000/-", "DEDUCTIBLE")]
    ),
    make_sample(
        "All other terms & conditions as per the standard HNB General Insurance Policy.",
        [("HNB General Insurance", "COMPANY")]
    ),
    make_sample(
        "Please click the link below to access the HNB General Insurance Limited (HNBGI) Insurance Product Information Document",
        [("HNB General Insurance Limited", "COMPANY")]
    ),
    make_sample(
        "Thank you for considering HNB General Insurance Limited.",
        [("HNB General Insurance Limited", "COMPANY")]
    )
]

def main():
    print("Loading datasets...")
    with open(TRAIN_FILE, "r") as f:
        train_data = json.load(f)
    with open(VAL_FILE, "r") as f:
        val_data = json.load(f)
        
    print(f"Original - Train: {len(train_data)}, Val: {len(val_data)}")
    
    augmented_samples = []
    for s in REAL_SAMPLES:
        for _ in range(15):
            augmented_samples.append(s)
            
    random.shuffle(augmented_samples)
    split_idx = int(len(augmented_samples) * 0.8)
    train_add = augmented_samples[:split_idx]
    val_add = augmented_samples[split_idx:]
    
    train_data.extend(train_add)
    val_data.extend(val_add)
    
    with open(TRAIN_FILE, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(VAL_FILE, "w") as f:
        json.dump(val_data, f, indent=2)
    print("Added new real quote samples successfully!")

if __name__ == "__main__":
    main()
