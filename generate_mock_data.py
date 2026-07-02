import json
import random

# Insurance Domain Vocabularies
companies = ["Allianz Insurance", "Ceylinco General", "HNB Assurance", "LOLC Life", "Fairfirst Insurance", "Sri Lanka Insurance", "Union Assurance", "Softlogic Life", "Amana Takaful", "Janashakthi"]
categories = ["Motor Plus", "Health Cover", "Life Protector", "DTA Housing Loan", "Surgical Plan", "Third Party", "Comprehensive"]
currency_symbols = ["Rs.", "LKR", "Rupees", "Rs"]
taxes = ["VAT", "SSCL", "Cess", "Government Taxes"]

# Define complex sentence structures
templates = [
    "The total gross premium for the {category} policy by {company} is {currency} {premium} inclusive of {tax}.",
    "{company} Quotation Summary: {category} - Net Premium: {premium} {currency} (Deductible: {currency} {deductible})",
    "Subject to a {copay}% copayment, the {company} {category} requires a contribution of {currency} {premium}.",
    "Single Premium Payable: {premium} {currency} | Company: {company} | Excess: {deductible}"
]

def generate_bboxes(tokens):
    """
    Heuristic logic to generate normalized bounding boxes (0-1000 scale) for LayoutLM.
    """
    bboxes = []
    left = 50
    top = 100
    line_height = 30
    word_gap = 10
    char_width = 8
    
    for token in tokens:
        width = len(token) * char_width
        right = left + width
        
        # Word wrapping if it exceeds the right margin (950)
        if right > 950:
            left = 50
            top += line_height
            right = left + width
            
        bottom = top + line_height
        
        # Ensure values don't exceed the 1000x1000 bounds
        bboxes.append([
            min(max(left, 0), 1000), 
            min(max(top, 0), 1000), 
            min(max(right, 0), 1000), 
            min(max(bottom, 0), 1000)
        ])
        
        # Move 'left' cursor for the next word
        left = right + word_gap
        
    return bboxes

def generate_sample():
    template = random.choice(templates)
    company = random.choice(companies)
    category = random.choice(categories)
    currency = random.choice(currency_symbols)
    tax = random.choice(taxes)
    
    # Generate random numbers for financial values
    premium = f"{random.randint(15, 300)},{random.choice(['000', '500', '250', '750'])}.00"
    deductible = f"{random.randint(2, 25)},000"
    copay = str(random.choice([10, 15, 20, 25]))
    
    words = []
    tags = []
    
    parts = template.split()
    for part in parts:
        strip_chars = ".,;:?!()[]%|"
        
        # Strip prefix characters
        prefix = ""
        while len(part) > 0 and part[0] in strip_chars:
            prefix += part[0]
            part = part[1:]
            
        # Strip suffix characters
        suffix_list = []
        while len(part) > 0 and part[-1] in strip_chars:
            suffix_list.append(part[-1])
            part = part[:-1]
        suffix = "".join(reversed(suffix_list))
        
        clean_part = part
        
        # Add prefix punctuation to tokens
        if prefix:
            for char in prefix:
                words.append(char)
                tags.append("O")
                
        # Process clean_part and map entity tags dynamically
        if clean_part == "{company}":
            val_tokens = company.replace(",", " , ").replace(".", " . ").split()
            for idx, w in enumerate(val_tokens):
                words.append(w)
                tags.append("B-COMPANY" if idx == 0 else "I-COMPANY")
        elif clean_part == "{premium}":
            val_tokens = premium.replace(",", " , ").replace(".", " . ").split()
            for idx, w in enumerate(val_tokens):
                words.append(w)
                tags.append("B-PREMIUM" if idx == 0 else "I-PREMIUM")
        elif clean_part == "{deductible}":
            val_tokens = deductible.replace(",", " , ").replace(".", " . ").split()
            for idx, w in enumerate(val_tokens):
                words.append(w)
                tags.append("B-DEDUCTIBLE" if idx == 0 else "I-DEDUCTIBLE")
        elif clean_part == "{copay}":
            val_tokens = copay.replace(",", " , ").replace(".", " . ").split()
            for idx, w in enumerate(val_tokens):
                words.append(w)
                tags.append("B-COPAY" if idx == 0 else "I-COPAY")
        elif clean_part == "{category}":
            val_tokens = category.replace(",", " , ").replace(".", " . ").split()
            for w in val_tokens:
                words.append(w)
                tags.append("O")
        elif clean_part == "{currency}":
            val_tokens = currency.replace(",", " , ").replace(".", " . ").split()
            for w in val_tokens:
                words.append(w)
                tags.append("O")
        elif clean_part == "{tax}":
            val_tokens = tax.replace(",", " , ").replace(".", " . ").split()
            for w in val_tokens:
                words.append(w)
                tags.append("O")
        else:
            val_tokens = clean_part.replace(",", " , ").replace(".", " . ").split()
            for w in val_tokens:
                words.append(w)
                tags.append("O")
                
        # Add suffix punctuation to tokens
        if suffix:
            for char in suffix:
                words.append(char)
                tags.append("O")
                
    return words, tags

dataset = []

# Generate 500 unique synthetic samples
for _ in range(500):
    tokens, tags = generate_sample()
    
    # Generate heuristic bounding boxes
    bboxes = generate_bboxes(tokens)
    
    dataset.append({
        "tokens": tokens,
        "ner_tags": tags,
        "bboxes": bboxes
    })

# Save to your training file
with open('train_data_augmented.json', 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Successfully generated {len(dataset)} insurance training samples with dynamic token-tag matching and heuristic bboxes!")
