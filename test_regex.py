from infer_and_compare import extract_rich_fields, detect_insurance_class
import re

with open("extracted_text1.txt", "r") as f:
    t1 = f.read()

with open("extracted_text2.txt", "r") as f:
    t2 = f.read()

print("Detecting class 1:", detect_insurance_class(t1, ""))
print("Detecting class 2:", detect_insurance_class(t2, ""))

print("\n--- Testing Quote 1 (Allianz) ---")
res1 = extract_rich_fields(t1, "general")
for k, v in res1.items():
    if v != "Not found" and k != "plans":
        print(f"{k}: {v}")

print("\n--- Testing Quote 2 (LOLC General) ---")
res2 = extract_rich_fields(t2, "general")
for k, v in res2.items():
    if v != "Not found" and k != "plans":
        print(f"{k}: {v}")
