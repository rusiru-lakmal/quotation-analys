import pdfplumber

def get_clean_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = []
        for page in pdf.pages:
            # Filter normal horizontal characters
            normal_chars = []
            for char in page.chars:
                matrix = char.get("matrix", (1, 0, 0, 1, 0, 0))
                # Only keep characters that are strictly horizontal (not rotated)
                if abs(matrix[1]) < 0.01 and abs(matrix[2]) < 0.01:
                    normal_chars.append(char)
            
            # Sort chars by vertical position (top to bottom), then horizontal position (left to right)
            # y0 coordinates in PDF run bottom-to-top, so we group by line first.
            # A simple line grouping algorithm:
            lines = []
            tolerance = 3 # y-axis tolerance for a single line
            
            # Sort chars by y0 descending (top of the page first)
            normal_chars.sort(key=lambda c: -c["y0"])
            
            current_line = []
            current_y = None
            
            for char in normal_chars:
                y = char["y0"]
                if current_y is None:
                    current_y = y
                    current_line.append(char)
                elif abs(current_y - y) <= tolerance:
                    current_line.append(char)
                else:
                    # Sort the completed line left-to-right (x0 ascending)
                    current_line.sort(key=lambda c: c["x0"])
                    lines.append("".join([c["text"] for c in current_line]))
                    current_line = [char]
                    current_y = y
                    
            if current_line:
                current_line.sort(key=lambda c: c["x0"])
                lines.append("".join([c["text"] for c in current_line]))
                
            page_text = "\n".join(lines)
            all_text.append(page_text)
            
    return "\n".join(all_text)

with pdfplumber.open("quotations/download (3).pdf") as pdf:
    print("Number of pages:", len(pdf.pages))
    for i, page in enumerate(pdf.pages):
        print(f"Page {i}: Chars={len(page.chars)}, Images={len(page.images)}, Rects={len(page.rects)}")
        
clean_text = get_clean_text("quotations/download (3).pdf")
print("--- Clean Filtered Text ---")
print(repr(clean_text))
