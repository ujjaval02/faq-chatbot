import PyPDF2
with open('faqs.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text[:500])  # Print first 500 characters