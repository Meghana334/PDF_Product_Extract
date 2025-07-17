from mistralai import Mistral
from pathlib import Path
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import json
import json
import base64
import os

from logger import setup_logger, log_info

logger = setup_logger()

# NEW: Import the organizer function
from ocr_organizer import process_ocr_response

from dotenv import load_dotenv
load_dotenv()


# Your existing OCR processing code remains the same
api_key = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)
from groq import Groq


groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

pdf_file = Path("data/test_2.pdf")
# assert pdf_file.is_file()

uploaded_file = client.files.upload(
    file={
        "file_name": pdf_file.stem,
        "content": pdf_file.read_bytes(),
    },
    purpose="ocr",
)

signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

pdf_response = client.ocr.process(
    document=DocumentURLChunk(document_url=signed_url.url),
    model="mistral-ocr-latest",
    include_image_base64=True
)
log_info(logger,"pdf_response")
log_info(logger,pdf_response)

# ✅ Use safe native Python dict
response_dict = pdf_response.model_dump()

# Continue to process
organized_data = process_ocr_response(response_dict, str(pdf_file))




# print("Processing completed! Check the generated files.")

# pprint.pprint(response_dict)

def generate_product_desc(product_input: str) -> str:
    """
    Generate an enhanced product description using Groq AI.
    
    Args:
        product_input (str): Product description and features (as a combined text str format)
        groq_api_key (str): Your Groq API key
    
    Returns:
        str: Generated product description
    """

    
    # Create a comprehensive prompt for product description generation
    prompt = f"""
    You are an expert product description writer. Create a compelling, professional product description based on the following product information:

    Product Information: {product_input}

    Please generate a well-structured product description that includes :
    - An engaging headline/title
    - Key features and benefits
    - Clear value proposition
    - Compelling call-to-action language
    - SEO-friendly content
    
    Make it persuasive, informative, and customer-focused. Keep the tone professional yet engaging.
    Note: Don't Generate more than 3 lines (FOLLOW THIS STRICTLY)
    NOTE: don't format it just raw text (FOLLOW THIS STRICTLY)
    """
    
    try:
        # Create completion
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,  # Slightly lower for more consistent output
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        # Collect the streamed response
        generated_description = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                generated_description += chunk.choices[0].delta.content
        
        return generated_description.strip()
    
    except Exception as e:
        return f"Error generating product description: {str(e)}"
    
    
def convert_json_format(input_file, output_file):
    # Read the input JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Helper function to read image and convert to base64
    def image_to_base64(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except:
            return ""
    
    # Helper function to extract specifications from tables
    def extract_specifications(tables):
        specs = []
        for table in tables:
            if 'rows' in table:
                for row in table['rows']:
                    if len(row) >= 2:
                        label_value = row[1].split(' | ')
                        if len(label_value) == 2:
                            specs.append({
                                "label": label_value[0].strip(),
                                "value": label_value[1].strip()
                            })
            # Handle single row tables from headers
            if 'headers' in table and len(table['headers']) >= 2:
                header_parts = table['headers'][1].split(' | ')
                if len(header_parts) == 2:
                    specs.append({
                        "label": header_parts[0].strip(),
                        "value": header_parts[1].strip()
                    })
        return specs
    
    # Convert each product
    converted_products = []
    for product in data['products']:
        # Get base64 data from all_page_images
        main_image_base64 = ""
        thumbnails = []
        count =0
        if 'all_page_images' in product and product['all_page_images']:
            for img_data in product['all_page_images']:
                if 'base64_data' in img_data and img_data["id"]!= "img-0.jpeg":
                    print(img_data["id"])
                    base64_str = img_data['base64_data']
                    count += 1 
                    thumbnails.append(base64_str)
        main_image_base64 = product["all_page_images"][0]["base64_data"]
        print(product["all_page_images"][0]["id"])
        
        # If all_page_images is empty, try to read from local paths
        count =0
        if not main_image_base64 and 'product_images' in product:
            for img_path in product['product_images']:
                base64_data = image_to_base64(img_path)
                if base64_data:
                    full_base64 = f"data:image/jpeg;base64,{base64_data}"
                    if main_image_base64 == "":
                        main_image_base64 = full_base64
                    count += 1
                    print(f"Count of Image {count}")    
                    thumbnails.append(full_base64)
        
        converted_product = {
            "product_name": product.get('product_name', ''),
            "product_description": generate_product_desc(f"{product.get('product_description', '')}"),
            "category": "Blowers",  # Default category
            "rating": "4.5",  # Default rating
            "reviewCount": "128",  # Default review count
            "detailedDescription": f"This is a {product.get('product_description', 'product')} designed for professional use.",
            "features": product.get('features', []),
            "specifications": extract_specifications(product.get('tables', [])),
            "mainImage": main_image_base64,
            "thumbnails": thumbnails
        }
        
        converted_products.append(converted_product)
    
    # Create final output structure
    output_data = {
        "products": converted_products
    }
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

# Usage
convert_json_format('test_2_organized_data.json', 'data.json')




