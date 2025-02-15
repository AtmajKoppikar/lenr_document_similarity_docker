from openai import OpenAI
import pdfplumber
import os

client = OpenAI(
    api_key = 'sk-H9W8WafE2ckLkPlhvTYgxq1uJbrs_rFyAmnnF5yW8lT3BlbkFJBeGg9cFeH9-JkNUFv8OUUUGISKk2kbETflv0J3ttAA'
)
# Function to extract text from a PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    all_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + '\n'
    return all_text

# Function to query OpenAI API for extracting the main body of text
def get_main_body_of_text(pdf_text):
    # prompt = (
    #     f"Extract the main body of the text from the following document. "
    #     f"Ignore titles, references, acknowledgements, and figures: \n\n{pdf_text}"
    # )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", 
             "content": "Extract the main body of the text from the following document. Ignore titles, references, acknowledgements, and figures"},
            {"role": "user", 
             "content": pdf_text}
        ],
        model="gpt-4o-mini"
    )
    
    return response.choices[0].message.content

# Function to split text into smaller chunks if the document is too large
def split_text_into_chunks(text, chunk_size=4000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Main function that processes the PDF, splits into chunks, and gets the main body
def process_pdf_and_extract_main_body(pdf_path):
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks if necessary (each chunk of size ~4000 tokens)
    chunks = split_text_into_chunks(pdf_text)
    
    # Initialize OpenAI API key

    # Query OpenAI API for each chunk and collect results
    main_body_text = ''
    for chunk in chunks:
        main_body_text += get_main_body_of_text(chunk)
    
    return main_body_text

# Save the main body to a file
def save_to_file(text, file_name):
    with open(file_name, 'w') as file:
        file.write(text)

# Example usage
if __name__ == "__main__":
    # Replace with your OpenAI API key
    
    # Path to your PDF file
    pdf_path = "/Users/atmajkoppikar/Academics/LENR/test/4.pdf"
    
    # Process PDF and get the main body of the text
    main_body = process_pdf_and_extract_main_body(pdf_path)
    
    # Print the main body of the text
    print(main_body)
    
    # Optionally, save the output to a file
    # save_to_file(main_body, "main_body_output.txt")
