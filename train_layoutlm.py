import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset as HFDataset

# 1. Define label mappings
LABEL_LIST = ["O", "B-ATTRIBUTE_NAME", "I-ATTRIBUTE_NAME", "B-ATTRIBUTE_LIMIT", "I-ATTRIBUTE_LIMIT", "B-ATTRIBUTE_PREMIUM", "I-ATTRIBUTE_PREMIUM"]
id2label = {v: k for v, k in enumerate(LABEL_LIST)}
label2id = {k: v for v, k in enumerate(LABEL_LIST)}

# 2. Define custom dataset encoder
def preprocess_data(examples, processor):
    """
    Encodes the text tokens, layout bounding boxes
    using LayoutLMv3Processor's tokenizer to prepare inputs for the model.
    """
    tokens = examples["tokens"]      # List of lists of strings
    bboxes = examples["bboxes"]      # List of lists of bounding boxes (normalized to 1000)
    ner_tags = examples["ner_tags"]  # List of lists of integer labels

    encoding = processor.tokenizer(
        text=tokens,
        boxes=bboxes,
        word_labels=ner_tags,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # Remove batch dimension if returned as tensor
    for k in encoding.keys():
        if isinstance(encoding[k], torch.Tensor) and encoding[k].shape[0] == 1:
            encoding[k] = encoding[k].squeeze(0)
            
    return encoding

def main():
    # Initialize Processor and Model
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id
    )
    
    print("Loading augmented training dataset...")
    if not os.path.exists("train_data_augmented.json"):
        print("Error: train_data_augmented.json not found! Run generate_mock_data.py first.")
        return
        
    with open("train_data_augmented.json", "r") as f:
        data_list = json.load(f)
        
    # Map string ner_tags to integer IDs
    processed_list = []
    for item in data_list:
        processed_list.append({
            "tokens": item["tokens"],
            "bboxes": item["bboxes"],
            "ner_tags": [label2id[tag] for tag in item["ner_tags"]]
        })
        
    # Split into 80% train and 20% validation
    dataset = HFDataset.from_list(processed_list)
    dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["test"]
    
    print(f"Total training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")
    
    # Process dataset
    print("Preprocessing datasets...")
    processed_train = train_dataset.map(
        lambda x: preprocess_data(x, processor),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    processed_val = val_dataset.map(
        lambda x: preprocess_data(x, processor),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 3. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./layoutlmv3-insurance-results",
        num_train_epochs=3,              # Train for 3 epochs
        per_device_train_batch_size=4,   # Adjust based on memory
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),  # Enable FP16 training on CUDA GPUs
        use_cpu=False                    # Use MPS on Apple Silicon if available
    )
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val
    )
    
    print("Starting LayoutLMv3 training...")
    trainer.train()
    
    # Save the model and processor
    print("Saving fine-tuned model...")
    model.save_pretrained("./model_output_layoutlm")
    processor.save_pretrained("./model_output_layoutlm")
    print("Model saved to './model_output_layoutlm'")

if __name__ == "__main__":
    main()
