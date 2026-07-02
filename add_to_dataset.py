import json
import os

DATASET_FILE = "train_data.json"

def load_dataset():
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_dataset(data):
    with open(DATASET_FILE, "w") as f:
        json.dump(data, f, indent=2)

def clean_token(token):
    return token.strip(".,;:?!()\"'")

def match_phrase(tokens, phrase):
    """
    Finds the start and end index of a phrase in a list of tokens.
    """
    if not phrase:
        return None
    phrase_tokens = [clean_token(t.lower()) for t in phrase.split()]
    n_phrase = len(phrase_tokens)
    
    for i in range(len(tokens) - n_phrase + 1):
        sub_window = [clean_token(tokens[i+j].lower()) for j in range(n_phrase)]
        if sub_window == phrase_tokens:
            return list(range(i, i + n_phrase))
    return None

def main():
    print("=== Custom Model Annotation Helper ===")
    print("Use this tool to add real sentences from your PDFs to your training dataset.\n")
    
    dataset = load_dataset()
    print(f"Current training dataset size: {len(dataset)} samples.")
    
    while True:
        sentence = input("\nPaste a sentence/line containing quote details (or type 'exit' to quit):\n> ").strip()
        if not sentence:
            continue
        if sentence.lower() == 'exit':
            break
            
        tokens = sentence.split()
        tags = ["O"] * len(tokens)
        
        print(f"\nTokens detected: {tokens}")
        
        # 1. Company
        company = input("Enter the Company Name exact phrase (e.g. 'Blue Cross'): ").strip()
        comp_idx = match_phrase(tokens, company)
        if comp_idx:
            for idx, pos in enumerate(comp_idx):
                tags[pos] = "B-COMPANY" if idx == 0 else "I-COMPANY"
        elif company:
            print(f"Warning: Phrase '{company}' not found in tokens.")
            
        # 2. Premium
        premium = input("Enter the Premium exact phrase (e.g. '$150/month'): ").strip()
        prem_idx = match_phrase(tokens, premium)
        if prem_idx:
            for idx, pos in enumerate(prem_idx):
                tags[pos] = "B-PREMIUM" if idx == 0 else "I-PREMIUM"
        elif premium:
            print(f"Warning: Phrase '{premium}' not found in tokens.")
            
        # 3. Deductible
        deductible = input("Enter the Deductible exact phrase (e.g. '$500'): ").strip()
        ded_idx = match_phrase(tokens, deductible)
        if ded_idx:
            for idx, pos in enumerate(ded_idx):
                tags[pos] = "B-DEDUCTIBLE" if idx == 0 else "I-DEDUCTIBLE"
        elif deductible:
            print(f"Warning: Phrase '{deductible}' not found in tokens.")
            
        # 4. Copay
        copay = input("Enter the Copay exact phrase (e.g. '20% copay'): ").strip()
        copay_idx = match_phrase(tokens, copay)
        if copay_idx:
            for idx, pos in enumerate(copay_idx):
                tags[pos] = "B-COPAY" if idx == 0 else "I-COPAY"
        elif copay:
            print(f"Warning: Phrase '{copay}' not found in tokens.")
            
        # Review and Save
        print("\nReviewing Labels:")
        labeled_tokens = []
        for t, tag in zip(tokens, tags):
            if tag != "O":
                labeled_tokens.append(f"{t}[{tag}]")
            else:
                labeled_tokens.append(t)
        print(" ".join(labeled_tokens))
        
        confirm = input("\nSave this sample to dataset? (y/n): ").strip().lower()
        if confirm == 'y':
            dataset.append({
                "tokens": tokens,
                "ner_tags": tags
            })
            save_dataset(dataset)
            print(f"Saved! Total dataset size: {len(dataset)}")
        else:
            print("Discarded.")

if __name__ == "__main__":
    main()
