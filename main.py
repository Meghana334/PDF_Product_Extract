from mistralai import Mistral, DocumentURLChunk
from pathlib import Path
import json
import base64
import os
import ast
import re

from logger import setup_logger, log_info
from ocr_convertor5 import process_ocr_response
from groq import Groq

logger = setup_logger()

# Setup API keys
api_key = ""
os.environ["GROQ_API_KEY"] = ''

client = Mistral(api_key=api_key)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

pdf_file = Path(r"C:\Users\MEGHA\OneDrive\Desktop\AIProj\pdf_to_html\integrated_pdf\test1.pdf")

# Upload the PDF to Mistral OCR
uploaded_file = client.files.upload(
    file={
        "file_name": pdf_file.stem,
        "content": pdf_file.read_bytes(),
    },
    purpose="ocr",
)

signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

# Run OCR processing
pdf_response = client.ocr.process(
    document=DocumentURLChunk(document_url=signed_url.url),
    model="mistral-ocr-latest",
    include_image_base64=True
)

response_dict = json.loads(pdf_response.model_dump_json())
organized_data = process_ocr_response(response_dict, str(pdf_file))


# ✅ Accurate LLM-based product name + description extraction
def extract_product_metadata(text: str) -> dict:
    prompt = f"""
You are an intelligent assistant. From the following product-related OCR text, extract the **product_name** and **product_description**.

- product_name: Use the actual title (4–10 words max, no model/brand codes)
- product_description: What it is, what it does, and who it's for in clean English.

Return JSON only like:
{{
  "product_name": "Electric Industrial Blower",
  "product_description": "A high-speed blower designed for industrial cooling and ventilation systems."
}}

Text:
{text}
""".strip()

    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
            top_p=1,
        )
        output = completion.choices[0].message.content.strip()

        # Try to find complete or partial JSON
        json_match = re.search(r'\{.*?\}', output, re.DOTALL)

        if not json_match:
            start = output.find("{")
            if start != -1:
                partial = output[start:]
                if partial.count("{") > partial.count("}"):
                    partial += "}" * (partial.count("{") - partial.count("}"))
                json_match = re.search(r'\{.*\}', partial, re.DOTALL)

        if json_match:
            json_str = json_match.group()

            # Fix single quotes
            if json_str.startswith("{'") and not json_str.startswith('{"'):
                json_str = json_str.replace("'", '"')

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return ast.literal_eval(json_str)

        return {"error": "No valid JSON found in model response."}

    except Exception as e:
        return {"error": str(e)}


# 🔁 Update each product with improved metadata
for product in organized_data["products"]:
    raw_text = product.get("raw_text", "")
    metadata = extract_product_metadata(raw_text)


    if metadata:
        product["product_name"] = metadata.get("product_name", product.get("product_name", ""))
        product["product_description"] = metadata.get("product_description", product.get("product_description", ""))


# 📂 Save the organized product data
with open("test4_organized_data.json", "w", encoding="utf-8") as f:
    json.dump(organized_data, f, indent=2, ensure_ascii=False)


# 🔁 Convert JSON for frontend consumption
def convert_json_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def image_to_base64(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except:
            return ""

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
            if 'headers' in table and len(table['headers']) >= 2:
                header_parts = table['headers'][1].split(' | ')
                if len(header_parts) == 2:
                    specs.append({
                        "label": header_parts[0].strip(),
                        "value": header_parts[1].strip()
                    })
        return specs

    converted_products = []
    for product in data['products']:
        main_image_base64 = ""
        thumbnails = []
        if 'all_page_images' in product and product['all_page_images']:
            for img_data in product['all_page_images']:
                if 'base64_data' in img_data and img_data["id"] != "img-0.jpeg":
                    thumbnails.append(img_data['base64_data'])
        if product["all_page_images"]:
            main_image_base64 = product["all_page_images"][0]["base64_data"]

        converted_product = {
            "product_name": product.get('product_name', ''),
            "product_description": product.get('product_description', ''),
            "category": "Blowers",
            "rating": "4.5",
            "reviewCount": "128",
            "detailedDescription": f"This is a {product.get('product_description', 'product')} designed for professional use.",
            "features": product.get('features', []),
            "specifications": extract_specifications(product.get('tables', [])),
            "mainImage": main_image_base64,
            "thumbnails": thumbnails,
            "groq_chunks": product.get("groq_chunks", [])
        }

        converted_products.append(converted_product)

    output_data = {"products": converted_products}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


# ✅ Create UI-ready JSON
convert_json_format('test4_organized_data.json', 'data.json')