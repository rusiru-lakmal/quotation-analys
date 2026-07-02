import json
import random

# We will generate synthetic insurance quote sentences/paragraphs for training
# and validation of our custom NER/Token Classification model.

COMPANIES = ["Aetna Health", "Blue Cross", "Cigna Corp", "UnitedHealthcare", "Humana Inc", "Kaiser Permanente", "Geico Insurance", "State Farm", "Progressive Corp", "Allstate Corp"]
PREMIUMS = ["$120 per month", "$150/month", "$95/mo", "$230 monthly", "$1400 annually", "$1800 per year", "$85 per month", "$300/mo"]
DEDUCTIBLES = ["$500 deductible", "$1,000 deductible", "$250 deductible", "$2000 deductible", "$0 deductible", "$1500 deductible"]
COPAYS = ["20% copay", "10% copay", "$20 co-pay", "15% coinsurance", "30% copayment", "$35 copay"]

# Sentence templates with entity markers
TEMPLATES = [
     "This policy is provided by {company} with a monthly premium of {premium}.",
     "Under {company}, the deductible is {deductible} and the premium is {premium}.",
     "With {company}, you have a {copay} and a {deductible}.",
     "The {company} plan features a {premium} rate and {deductible} limit.",
     "Your monthly payment to {company} is {premium} with a {copay} per visit.",
     "For {company}, the annual premium is {premium} and the deductible is {deductible} with a {copay}.",
     "A quote from {company} offers {deductible} and {copay} for doctor visits at {premium}."
]

def generate_sample():
    company = random.choice(COMPANIES)
    premium = random.choice(PREMIUMS)
    deductible = random.choice(DEDUCTIBLES)
    copay = random.choice(COPAYS)
    
    template = random.choice(TEMPLATES)
    
    # We need to construct the sentence and keep track of character offsets/labels
    # or token-level labels directly. To be simple and robust, let's build the words
    # list and tag list word-by-word.
    
    words = []
    tags = []
    
    # We can split the template and insert words and their tags
    parts = template.split()
    for part in parts:
        clean_part = part.strip(".,;:?!")
        # Find ending punctuation
        punct = part[len(clean_part):] if len(clean_part) < len(part) else ""
        
        if clean_part == "{company}":
            comp_words = company.split()
            for idx, w in enumerate(comp_words):
                words.append(w)
                tags.append("B-COMPANY" if idx == 0 else "I-COMPANY")
        elif clean_part == "{premium}":
            prem_words = premium.split()
            for idx, w in enumerate(prem_words):
                words.append(w)
                tags.append("B-PREMIUM" if idx == 0 else "I-PREMIUM")
        elif clean_part == "{deductible}":
            ded_words = deductible.split()
            for idx, w in enumerate(ded_words):
                words.append(w)
                tags.append("B-DEDUCTIBLE" if idx == 0 else "I-DEDUCTIBLE")
        elif clean_part == "{copay}":
            copay_words = copay.split()
            for idx, w in enumerate(copay_words):
                words.append(w)
                tags.append("B-COPAY" if idx == 0 else "I-COPAY")
        else:
            words.append(clean_part)
            tags.append("O")
            
        if punct:
            words.append(punct)
            tags.append("O")
            
    return {"tokens": words, "ner_tags": tags}

def main():
    # Generate 100 training samples and 20 validation samples
    train_data = [generate_sample() for _ in range(100)]
    val_data = [generate_sample() for _ in range(20)]
    
    with open("train_data.json", "w") as f:
        json.dump(train_data, f, indent=2)
        
    with open("val_data.json", "w") as f:
        json.dump(val_data, f, indent=2)
        
    print(f"Generated 100 training samples to train_data.json and 20 samples to val_data.json.")

if __name__ == "__main__":
    main()
