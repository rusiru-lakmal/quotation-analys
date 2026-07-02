import pdfplumber

with pdfplumber.open("quotations/SLIC Car Quotation.pdf") as pdf:
    page = pdf.pages[0]
    for char in page.chars[:100]:
        print(f"Char: '{char['text']}' x0: {char['x0']:.1f} x1: {char['x1']:.1f} y0: {char['y0']:.1f}")
