import os
import sys
import json
from infer_and_compare import extract_text_from_pdf, get_combined_quote_data, detect_insurance_class

# Regression Test Cases
EXPECTED_METRICS = {
    "SOFT LOGIC.pdf": {
        "company": "Softlogic Life Insurance PLC",
        "insured_name": "A",
        "total_payable": "LKR 3,344,094.00"
    },
    "UNION.pdf": {
        "company": "Union Assurance PLC",
        "insured_name": "Avery Dennison Lanka (Pvt) Ltd",
        "total_payable": "LKR 2,478,595.00"
    }
}

def run_tests():
    print("==================================================")
    print("      INSURANCE EXTRACTION REGRESSION SUITE       ")
    print("==================================================")
    
    quotations_dir = "quotations"
    model_path = "./model_output"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        sys.exit(1)
        
    failures = 0
    passed = 0
    
    for filename, expected in EXPECTED_METRICS.items():
        pdf_path = os.path.join(quotations_dir, filename)
        if not os.path.exists(pdf_path):
            print(f"Skipping {filename} (File not found in quotations/)")
            continue
            
        print(f"\nRunning extraction regression test on: {filename}...")
        try:
            # Extract text
            text = extract_text_from_pdf(pdf_path)
            # Run extraction pipeline
            ins_class = detect_insurance_class(text, "")
            actual = get_combined_quote_data(text, model_path, ins_class=ins_class)
            
            # Validate fields
            file_failed = False
            for field, expected_val in expected.items():
                actual_val = actual.get(field, "Not found")
                if actual_val != expected_val:
                    print(f"  ❌ FAILED: Field '{field}' | Expected: '{expected_val}' | Actual: '{actual_val}'")
                    file_failed = True
                else:
                    print(f"  ✅ PASSED: Field '{field}' matches '{expected_val}'")
                    
            if file_failed:
                failures += 1
            else:
                passed += 1
                
        except Exception as e:
            print(f"  ❌ ERROR: Failed to run extraction pipeline on {filename}: {e}")
            failures += 1
            
    print("\n==================================================")
    print(f"RESULTS: {passed} PASSED, {failures} FAILED")
    print("==================================================")
    
    if failures > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
