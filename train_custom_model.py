import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset

LABEL_LIST = ["O", "B-COMPANY", "I-COMPANY", "B-PREMIUM", "I-PREMIUM", "B-DEDUCTIBLE", "I-DEDUCTIBLE", "B-COPAY", "I-COPAY"]
id2label = {v: k for v, k in enumerate(LABEL_LIST)}
label2id = {k: v for v, k in enumerate(LABEL_LIST)}

def align_labels_with_tokens(labels, word_ids):
    """
    Subtoken alignment for wordpieces. If a word is split into multiple subtokens,
    we tag the first subtoken with the label and the rest with -100 (ignored in loss computation).
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            # First subtoken of a word
            current_word = word_id
            label_str = labels[word_id]
            new_labels.append(label2id[label_str])
        else:
            # Subsequent subtokens
            new_labels.append(-100)
    return new_labels

def tokenize_and_align(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = align_labels_with_tokens(labels, word_ids)
        all_labels.append(aligned_labels)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

def main():
    print("Loading datasets...")
    if not os.path.exists("train_data.json") or not os.path.exists("val_data.json"):
        print("Data files not found! Please run generate_mock_data.py first.")
        return
        
    with open("train_data.json", "r") as f:
        train_list = json.load(f)
    with open("val_data.json", "r") as f:
        val_list = json.load(f)
        
    train_dataset = Dataset.from_list(train_list)
    val_dataset = Dataset.from_list(val_list)
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Tokenizing and aligning labels...")
    tokenized_train = train_dataset.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: tokenize_and_align(x, tokenizer), batched=True)
    
    print("Initializing model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )
    
    # Fast training configs for CPU/local environment demo
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=5,
        use_cpu=False # Enable GPU (MPS) acceleration on MacBook Pro
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting custom model fine-tuning...")
    trainer.train()
    
    print("Saving trained model...")
    model.save_pretrained("./model_output")
    tokenizer.save_pretrained("./model_output")
    print("Model saved to './model_output'")

if __name__ == "__main__":
    main()
