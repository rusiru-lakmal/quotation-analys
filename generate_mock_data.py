import json
import random

# Generic Tags: B-ATTRIBUTE_NAME, B-ATTRIBUTE_LIMIT, B-ATTRIBUTE_PREMIUM
templates = [
    # Group Life Example
    ("Accidental Death Benefit cover is {limit} and the premium is {premium}", 
     ["B-ATTRIBUTE_NAME", "I-ATTRIBUTE_NAME", "I-ATTRIBUTE_NAME", "O", "O", "B-ATTRIBUTE_LIMIT", "O", "O", "O", "O", "B-ATTRIBUTE_PREMIUM"]),
    
    # Motor Example
    ("Windscreen Cover limit {limit} subject to additional premium of {premium}",
     ["B-ATTRIBUTE_NAME", "I-ATTRIBUTE_NAME", "O", "B-ATTRIBUTE_LIMIT", "O", "O", "O", "O", "O", "B-ATTRIBUTE_PREMIUM"]),
    
    # Health Example
    ("Critical Illness Cover up to {limit} for an annual cost of {premium}",
     ["B-ATTRIBUTE_NAME", "I-ATTRIBUTE_NAME", "I-ATTRIBUTE_NAME", "O", "O", "B-ATTRIBUTE_LIMIT", "O", "O", "O", "O", "O", "B-ATTRIBUTE_PREMIUM"])
]

def generate_bboxes(tokens):
    bboxes, left, top = [], 50, 100
    for token in tokens:
        width = len(token) * 8
        right = left + width
        if right > 950:
            left, top = 50, top + 30
            right = left + width
        bboxes.append([min(max(left, 0), 1000), min(max(top, 0), 1000), min(max(right, 0), 1000), min(max(top + 30, 0), 1000)])
        left = right + 10
    return bboxes

def main():
    dataset = []
    for _ in range(1000):
        template, tags = random.choice(templates)
        limit = f"{random.randint(1, 5)},000,000.00"
        premium = f"{random.randint(10, 500)},000.00"
        
        sentence = template.format(limit=limit, premium=premium)
        tokens = sentence.split()
        
        dataset.append({
            "tokens": tokens,
            "ner_tags": tags[:len(tokens)],
            "bboxes": generate_bboxes(tokens)
        })

    with open('train_data_augmented.json', 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Generated {len(dataset)} Universal Generic Training Samples!")

if __name__ == "__main__":
    main()
