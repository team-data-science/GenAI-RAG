# creates a chat promt from the local llm

import json, os  # Importing JSON for handling JSON data and os for interacting with the operating system
import fitz  # PyMuPDF
from ollama import chat
from ollama import ChatResponse

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
    print(text)
    
    return text

def prepare_text_to_json(text_to_summarize):
    # Define the template with a placeholder
    instruction_template = "Here's a CV please create a JSON that separates each section. Just two attributes section name and content. Example sections are: Person details, Experience, Education, Certification, skills. Do not leave out or rephrase text: {}"
    
    response: ChatResponse = chat(model='mistral', messages=[
        {
            'role': 'user',
            'content': 'why is the sky blue?',
        },
    ])
    #print(response['message']['content'])
    # or access fields directly from the response object
    #print(response.message.content)

    return response['message']['content']

def main():

    extracted = extract_text_from_pdf('Liam_McGivney_CV.pdf')
    prepped_json = prepare_text_to_json(extracted)

    print(prepped_json)

# Entry point of the script
if __name__ == "__main__":
    main()  # Call the main function