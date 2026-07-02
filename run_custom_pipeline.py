import subprocess
import sys
import os

def run_cmd(args):
    print(f"Executing: {' '.join(args)}")
    result = subprocess.run(args, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(args)}")
        sys.exit(result.returncode)

def main():
    print("==================================================")
    # 1. Install dependencies
    print("Step 1: Installing/Checking dependencies...")
    run_cmd([sys.executable, "-m", "pip", "install", "transformers", "torch", "datasets", "accelerate", "pdfplumber"])
    
    print("\n==================================================")
    # 2. Generate data
    print("Step 2: Generating synthetic training data...")
    run_cmd([sys.executable, "generate_mock_data.py"])
    
    print("\n==================================================")
    # 3. Train model
    print("Step 3: Fine-tuning token classification model...")
    run_cmd([sys.executable, "train_custom_model.py"])
    
    print("\n==================================================")
    # 4. Infer and generate report
    print("Step 4: Running inference and compiling HTML comparison report...")
    run_cmd([sys.executable, "infer_and_compare.py"])
    
    print("\n==================================================")
    report_path = os.path.abspath("custom_comparison_report.html")
    print(f"Success! Custom Trained Model pipeline run completed.")
    print(f"Comparison report generated at: {report_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
