import fitz  # PyMuPDF
import json

doc = fitz.open('Liam_McGivney_CV.pdf')
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    text = page.get_text()
    print(text)

'''
import PyPDF2

with open('cv.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        print(page.extract_text())

'''

