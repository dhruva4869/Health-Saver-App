import os
import PyPDF2
from transformers import T5ForConditionalGeneration,T5Tokenizer

pdf_directory = 'storage/pdf_files'
text_directory = 'storage/extracted_text'

resume_files = []
for file_name in os.listdir(pdf_directory):
    if file_name.endswith('.pdf'):
        resume_files.append(os.path.join(pdf_directory, file_name))

resume_summaries = []  # To store the generated summaries

# Loop through each resume file
for resume_file in resume_files:
    with open(resume_file, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)

        # Extract text from each page
        text = ''
        
        for page in reader.pages:
            text += page.extract_text()
        
            model = T5ForConditionalGeneration.from_pretrained("t5-base")
            tokenizer = T5Tokenizer.from_pretrained("t5-base")

            # first tokenizing will be performed
            inputs = tokenizer.encode("summarize: " + text, 
            return_tensors="pt", max_length=1000, 
            truncation=True)

            # model if ForConditionalGeneration will do the summarizing
            outputs = model.generate(inputs, max_length=1000, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(outputs[0])

            resume_summaries.append(summary)


text_file_name = file_name.replace('.pdf', '.txt')
text_file_path = os.path.join(text_directory, text_file_name)
for i, summary in enumerate(resume_summaries):
    print(f"Summary for Resume {i+1}:")
    print(summary)
    with open(text_file_path, 'w') as text_file:
        text_file.write(summary)
    print()