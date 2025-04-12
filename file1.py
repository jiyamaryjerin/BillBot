'''
from paddleocr import PaddleOCR
import os

# 'en', 'ta', 'te', 'ka', 'devanagari'
lang_code = 'en'  
image_path = 'C:\\Users\\Sonal\\OneDrive\\Desktop\\VS PROJS\\hackinghard\\BillBot\\bmtc1.jpg'  


ocr = PaddleOCR(use_angle_cls=True, lang=lang_code)

result = ocr.ocr(image_path, cls=True)

output_file = f'output_{lang_code}.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    if result:
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                confidence = word_info[1][1]
                f.write(f"{text} - Confidence: {confidence:.2f}\n")
        print(f"\nOCR complete. Output saved to: {os.path.abspath(output_file)}")
    else:
        f.write("No text detected.\n")
        print("\nNo text detected. Empty result written to file.")
'''



import streamlit as st
import os
from dotenv import load_dotenv
import base64
import json
import time
from mistralai import Mistral
import pandas as pd 

st.title("BillBot")

# 1. API Key Input

load_dotenv()
api_key = os.getenv("MISTRAL_TOKEN")
if(api_key is None ):
    print( "API key not found.")

# Initialize session state variables for persistence
if "ocr_result" not in st.session_state:
    st.session_state["ocr_result"] = []
if "preview_src" not in st.session_state:
    st.session_state["preview_src"] = []
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = []

# 2. Choose file type: PDF or Image
file_type = st.radio("Select file type", ("PDF", "Image"))

# 3. Select source type: URL or Local Upload
source_type = st.radio("Select source type", ("URL", "Local Upload"))

input_url = ""
uploaded_files = []

if source_type == "URL":
    input_url = st.text_area("Enter one or multiple URLs (separate with new lines)")
else:
    uploaded_files = st.file_uploader("Upload one or more files", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

# Generate CSV output 



# 4. Process Button & OCR Handling
if st.button("Process"):
    if source_type == "URL" and not input_url.strip():
        st.error("Please enter at least one valid URL.")
    elif source_type == "Local Upload" and not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        #MISTRAL AI IS HERE
        client = Mistral(api_key=api_key)
        st.session_state["ocr_result"] = []
        st.session_state["preview_src"] = []
        st.session_state["image_bytes"] = []
        
        sources = input_url.split("\n") if source_type == "URL" else uploaded_files
        
        for idx, source in enumerate(sources):
            if file_type == "PDF":
                if source_type == "URL":
                    document = {"type": "document_url", "document_url": source.strip()}
                    preview_src = source.strip()
                else:
                    file_bytes = source.read()
                    encoded_pdf = base64.b64encode(file_bytes).decode("utf-8")
                    document = {"type": "document_url", "document_url": f"data:application/pdf;base64,{encoded_pdf}"}
                    preview_src = f"data:application/pdf;base64,{encoded_pdf}"
            else:
                if source_type == "URL":
                    document = {"type": "image_url", "image_url": source.strip()}
                    preview_src = source.strip()
                else:
                    file_bytes = source.read()
                    mime_type = source.type
                    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
                    document = {"type": "image_url", "image_url": f"data:{mime_type};base64,{encoded_image}"}
                    preview_src = f"data:{mime_type};base64,{encoded_image}"
                    st.session_state["image_bytes"].append(file_bytes)
            
            with st.spinner(f"Processing {source if source_type == 'URL' else source.name}..."):
                try:
                    #CHANGE TO PADDLEOCR 
                    ocr_response = client.ocr.process(model="mistral-ocr-latest", document=document, include_image_base64=True )
                    time.sleep(1)  # wait 1 second between request to prevent rate limit exceeding
                    
                    pages = ocr_response.pages if hasattr(ocr_response, "pages") else (ocr_response if isinstance(ocr_response, list) else [])
                    result_text = "\n\n".join(page.markdown for page in pages) or "No result found."
                except Exception as e:
                    result_text = f"Error extracting result: {e}"
                
                st.session_state["ocr_result"].append(result_text)
                st.session_state["preview_src"].append(preview_src)
                #print("output: \n",result_text)
                #print(type(result_text))
        
                # providing prompt to mistral ai 

                prompt = f"""
                            You are a bills and invoice parser. Understand raw and unstructured text and extract structured fields from the following bill text. map the fields to the following JSON format.
                            if there are fields you can identify but not in the JSON format, add them to the JSON as well. add only relevant fields to the JSON.
                            If you cannot identify any fields, remove that JSON field from the answer. do not generate own content.
                            Return the output as JSON with fields like this and all the fields should be seperately mapped (no nested json). Stick to the order of the fields mentioned below.:
                            - vendor_name
                            - invoice_number
                            - date
                            - due_date
                            - total_amount
                            - description
                            - quantity
                            - unit_price
                            - tax
                            - subtotal
                            - currency

                            give output only in json format. do not add any other text.
                            Bill Text:
                            \"\"\"
                            {result_text}
                            \"\"\"
                            """

                model = "mistral-large-latest"
                client = Mistral(api_key=api_key)

                chat_response = client.chat.complete(
                    model = model,
                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                )

                response = chat_response.choices[0].message.content
                import json
                with open('data.json', 'w') as f:
                    json.dump(response, f)
                print(response)
                
                lines = response.strip().splitlines()
                # Remove the first and last training lines
                clean_json_str = "\n".join(lines[1:-1])

                # Parse the cleaned JSON string
                response = json.loads(clean_json_str)
                print(response)
                '''
                df = pd.json_normalize(response)
        
                # df.to_excel("bill_template.xlsx", index=False)

                with pd.ExcelWriter('bill_template.xlsx', engine='openpyxl', mode='a') as writer:
                try:
                    new_df.to_excel(writer, sheet_name='Sheet 1', index=False, header=None)
                except PermissionError:
                    print("Close the file in Excel and try again.")

                # print this response (it is in json) if facing errors during converson to excel
                '''


