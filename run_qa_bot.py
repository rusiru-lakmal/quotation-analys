import os
import sys
import json
import time
from infer_and_compare import extract_text_from_pdf, get_combined_quote_data, detect_insurance_class

DATABASE_FILE = "regression_database.json"
REPORT_FILE = "qa_bot_report.html"
QUOTATIONS_DIR = "quotations"
MODEL_PATH = "./model_output"

def generate_html_report(results, total_docs, passed_docs, failed_docs, duration):
    # Determine overall status
    overall_status = "PASSED" if failed_docs == 0 and passed_docs > 0 else "FAILED"
    status_class = "status-pass" if overall_status == "PASSED" else "status-fail"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Automated QA Testing Bot Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151c2c;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --border-color: #243049;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .status-badge {{
            padding: 8px 16px;
            border-radius: 30px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .status-pass {{
            background-color: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
        }}
        .status-fail {{
            background-color: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
            border: 1px solid var(--accent-red);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .summary-card .number {{
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 5px;
        }}
        .summary-card .label {{
            color: var(--text-muted);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .doc-section {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 30px;
        }}
        .doc-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
            margin-bottom: 16px;
        }}
        .doc-title {{
            font-size: 18px;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .pass-row {{
            color: var(--accent-green);
        }}
        .fail-row {{
            color: var(--accent-red);
        }}
        .badge-inline {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .badge-pass {{
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--accent-green);
        }}
        .badge-fail {{
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--accent-red);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🤖 QA REGRESSION TESTING BOT</h1>
                <p style="color: var(--text-muted); margin: 5px 0 0 0;">Continuous Integration & Extraction Verification Report</p>
            </div>
            <div class="status-badge {status_class}">{overall_status}</div>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="number">{total_docs}</div>
                <div class="label">Total Checked</div>
            </div>
            <div class="summary-card" style="border-bottom: 3px solid var(--accent-green);">
                <div class="number" style="color: var(--accent-green);">{passed_docs}</div>
                <div class="label">Passed Docs</div>
            </div>
            <div class="summary-card" style="border-bottom: 3px solid var(--accent-red);">
                <div class="number" style="color: var(--accent-red);">{failed_docs}</div>
                <div class="label">Failed Docs</div>
            </div>
            <div class="summary-card">
                <div class="number" style="color: var(--accent-blue);">{duration:.2f}s</div>
                <div class="label">Total Duration</div>
            </div>
        </div>
        
        <h2 style="font-weight: 600; margin-bottom: 20px;">Detailed File Extractions</h2>
    """
    
    for filename, r in results.items():
        doc_status = "PASSED" if r["failed"] == 0 else "FAILED"
        doc_class = "badge-pass" if doc_status == "PASSED" else "badge-fail"
        
        html += f"""
        <div class="doc-section">
            <div class="doc-header">
                <div class="doc-title">📄 {filename}</div>
                <span class="badge-inline {doc_class}">{doc_status}</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Metric Field</th>
                        <th>Expected Reference Value</th>
                        <th>Actual Extracted Value</th>
                        <th>Result Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for field in r["fields"]:
            field_class = "pass-row" if field["status"] == "MATCH" else "fail-row"
            status_badge = f'<span class="badge-inline badge-pass">MATCH</span>' if field["status"] == "MATCH" else f'<span class="badge-inline badge-fail">MISMATCH</span>'
            
            html += f"""
                    <tr class="{field_class}">
                        <td><strong>{field['name']}</strong></td>
                        <td>{field['expected']}</td>
                        <td>{field['actual']}</td>
                        <td>{status_badge}</td>
                    </tr>
            """
            
        html += """
                </tbody>
            </table>
        </div>
        """
        
    html += """
    </div>
</body>
</html>
    """
    with open(REPORT_FILE, "w") as f:
        f.write(html)
    print(f"Generated dynamic QA Bot report at: {REPORT_FILE}")

def main():
    print("==================================================")
    print("        AUTOMATED QA TESTING BOT LAUNCHED        ")
    print("==================================================")
    
    if not os.path.exists(DATABASE_FILE):
        print(f"Error: Database file not found at {DATABASE_FILE}")
        sys.exit(1)
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    with open(DATABASE_FILE, "r") as f:
        database = json.load(f)
        
    results = {}
    total_docs = 0
    passed_docs = 0
    failed_docs = 0
    start_time = time.time()
    
    for filename, expected_fields in database.items():
        pdf_path = os.path.join(QUOTATIONS_DIR, filename)
        if not os.path.exists(pdf_path):
            continue
            
        total_docs += 1
        print(f"\n[Test #{total_docs}] Verifying: {filename}...")
        
        try:
            text = extract_text_from_pdf(pdf_path)
            ins_class = detect_insurance_class(text, "")
            actual_data = get_combined_quote_data(text, MODEL_PATH, ins_class)
            
            doc_result = {
                "failed": 0,
                "fields": []
            }
            
            for field, expected_val in expected_fields.items():
                actual_val = actual_data.get(field, "Not found")
                
                status = "MATCH" if actual_val == expected_val else "MISMATCH"
                if status == "MISMATCH":
                    doc_result["failed"] += 1
                    print(f"  ❌ FAILED: Field '{field}' | Expected: '{expected_val}' | Actual: '{actual_val}'")
                else:
                    print(f"  ✅ PASSED: Field '{field}' matches expected value.")
                    
                doc_result["fields"].append({
                    "name": field,
                    "expected": expected_val,
                    "actual": actual_val,
                    "status": status
                })
                
            results[filename] = doc_result
            if doc_result["failed"] > 0:
                failed_docs += 1
            else:
                passed_docs += 1
                
        except Exception as e:
            failed_docs += 1
            print(f"  💥 ERROR during testing: {e}")
            results[filename] = {
                "failed": 1,
                "fields": [{
                    "name": "Exception",
                    "expected": "No exception",
                    "actual": str(e),
                    "status": "MISMATCH"
                }]
            }
            
    duration = time.time() - start_time
    print("\n==================================================")
    print("                QA BOT RUN SUMMARY                ")
    print("==================================================")
    print(f"Total files checked: {total_docs}")
    print(f"Passed: {passed_docs}")
    print(f"Failed: {failed_docs}")
    print(f"Duration: {duration:.2f} seconds")
    print("==================================================")
    
    if total_docs > 0:
        generate_html_report(results, total_docs, passed_docs, failed_docs, duration)
        
    if failed_docs > 0:
        print("❌ Build Status: RED (Failing)")
        sys.exit(1)
    else:
        print("💚 Build Status: GREEN (All Passed)")
        sys.exit(0)

if __name__ == "__main__":
    main()
