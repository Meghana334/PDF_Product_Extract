
import json
import base64
import os
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from logger import setup_logger, log_info

logger = setup_logger()


def save_base64_image(base64_str: str, filename: str, output_dir: str = "extracted_images") -> str:
    os.makedirs(output_dir, exist_ok=True)
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    image_data = base64.b64decode(base64_str)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path


import re
from typing import Dict, Any

def extract_product_info_from_text(text: str) -> Dict[str, Any]:
    # ✅ Remove markdown image tags
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    log_info(logger,text)

    product_info = {
        "product_name": "",
        "product_description": "",
        "specifications": {},
        "features": [],
        "model_number": "",
        "brand": ""
    }

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # ✅ Skip markdown table lines
        if re.match(r'^\|.*\|$', line):
            i += 1
            continue

        # ✅ Product name from markdown style **ProductName**
        if not product_info["product_name"]:
            bold_match = re.search(r'\*\*([^\*\s]+)(?=\s)', line)
            if bold_match:
                product_info["product_name"] = bold_match.group(1)
                i += 1
                continue

        # ✅ Fallback product name from first token
        if not product_info["product_name"]:
            tok_match = re.match(r'^([A-Za-z0-9\-]+)', line)
            if tok_match:
                product_info["product_name"] = tok_match.group(1)

        # ✅ Model number
        model_match = re.search(r'Model[:\s]+([A-Z0-9\-]+)', line, re.IGNORECASE)
        if model_match:
            product_info["model_number"] = model_match.group(1)

        # ✅ Brand
        brand_match = re.search(r'Brand[:\s]+(\w+)', line, re.IGNORECASE)
        if brand_match:
            product_info["brand"] = brand_match.group(1)

        # ✅ Specifications
        spec_match = re.match(r'^([A-Za-z\s]+):\s*(.+)$', line)
        if spec_match:
            key = spec_match.group(1).strip()
            value = spec_match.group(2).strip()
            product_info["specifications"][key] = value

        # ✅ Enhanced feature extraction (first 50 lines)
        if i < 50:
            # **KEY** Value
            feature_md = re.match(r'\*\*([^\*]+)\*\*\s*(.*)', line)
            if feature_md:
                key = feature_md.group(1).strip()
                value = feature_md.group(2).strip()
                if not value and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        value = next_line
                        i += 1
                product_info["features"].append({key: value})
                i += 1
                continue

            # KEY: Value or KEY - Value (dot and number allowed)
            feature_kv = re.match(r'^([A-Za-z0-9.\s]+?)\s*[:\-–]\s*(.*)$', line)
            log_info(logger,feature_kv)
            if feature_kv:
                key = feature_kv.group(1).strip()
                value = feature_kv.group(2).strip()
                if not value and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not any(x in next_line for x in [":", "**", "|"]):
                        value = next_line
                        i += 1
                product_info["features"].append({key: value})
                i += 1
                continue

        # ✅ Always apply fallback: Line followed by another line
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not any(x in next_line for x in [":", "**", "|"]):
                product_info["features"].append({line: next_line})
                i += 2
                continue

        # ✅ Product description from markdown headers
        if not product_info["product_description"]:
            if '#' in line and '**' in line:
                desc_match = re.search(r'\*\*(.+?)\*\*', line)
                if desc_match:
                    product_info["product_description"] = desc_match.group(1).strip()

        i += 1

    # ✅ Fallback product name
    if not product_info["product_name"]:
        for line in lines:
            if len(line.strip()) > 10 and not line.strip().isdigit():
                product_info["product_name"] = line.strip()
                break

    # ✅ Fallback product description
    if not product_info["product_description"]:
        description_lines = []
        for line in lines:
            if len(line.strip()) > 20:
                description_lines.append(line.strip())
            if len(description_lines) >= 3:
                break
        product_info["product_description"] = " ".join(description_lines)

    return product_info



def extract_tables_from_text(text: str) -> List[Dict[str, Any]]:
    tables = []
    lines = text.split('\n')
    current_table = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if '\t' in line or '  ' in line:
            columns = re.split(r'\s{2,}|\t', line)
            columns = [col.strip() for col in columns if col.strip()]

            if len(columns) >= 2:
                if current_table is None:
                    current_table = {
                        "headers": columns,
                        "rows": []
                    }
                else:
                    current_table["rows"].append(columns)
        else:
            if current_table is not None:
                tables.append(current_table)
                current_table = None

    if current_table is not None:
        tables.append(current_table)

    return tables


def organize_ocr_response(ocr_response_dict: Dict[str, Any], pdf_filename: str) -> Dict[str, Any]:
    organized_data = {
        "source_pdf": pdf_filename,
        "extraction_timestamp": datetime.now().isoformat(),
        "products": [],
        "all_extracted_images": [],
        "metadata": {
            "total_pages": len(ocr_response_dict.get("pages", [])),
            "total_images": 0,
            "total_text_length": 0
        }
    }

    image_counter = 1
    all_text = ""

    for page_idx, page in enumerate(ocr_response_dict.get("pages", [])):
        page_text = page.get("markdown", "")
        all_text += page_text + "\n"

    for image in page.get("images", []):
        image_id = logger_info(logger_info)
    
        
        image_id = image.get("id", f"img_{image_counter}")
        # Skip specific IDs
        if image_id in ["img-1.jpeg", "img-6.jpeg"]:
            continue

        image_filename = f"page_{page_idx + 1}_image_{image_counter}.jpg"
        try:
            saved_path = save_base64_image(
                image.get("image_base64", ""),
                image_filename
            )
            organized_data["all_extracted_images"].append({
                "id": image_id,
                "filename": image_filename,
                "local_path": saved_path,
                "base64_data": image.get("image_base64", ""),
                "page_number": page_idx + 1,
                "size_estimate": len(image.get("image_base64", "")) * 3 // 4
            })
            image_counter += 1
        except Exception as e:
            print(f"Error saving image {image_filename}: {e}")

    product_info = extract_product_info_from_text(all_text)
    log_info(logger, product_info)

    tables = extract_tables_from_text(all_text)
    log_info(logger, tables)

    product_data = {
        "product_name": product_info["product_name"],
        "product_description": product_info["product_description"],
        "model_number": product_info["model_number"],
        "brand": product_info["brand"],
        "specifications": product_info["specifications"],
        "features": product_info["features"],
        "tables": tables,
        "product_images": [],
        "thumbnail_image": "",
        "all_page_images": organized_data["all_extracted_images"],
        "raw_text": all_text
    }

    if organized_data["all_extracted_images"]:
        product_data["thumbnail_image"] = organized_data["all_extracted_images"][0]["local_path"]
        product_data["product_images"] = [img["local_path"] for img in organized_data["all_extracted_images"]]

    organized_data["products"].append(product_data)
    organized_data["metadata"]["total_images"] = len(organized_data["all_extracted_images"])
    organized_data["metadata"]["total_text_length"] = len(all_text)

    return organized_data


def save_organized_data(organized_data: Dict[str, Any], output_filename: str = "organized_product_data.json"):
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, indent=4, ensure_ascii=False)
        #print(f"Organized data saved to {output_filename}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def process_ocr_response(ocr_response_dict: Dict[str, Any], pdf_filename: str):
    #print("Processing OCR response...")
    
    organized_data = organize_ocr_response(ocr_response_dict, pdf_filename)
    log_info(logger, organized_data)

    output_filename = f"{Path(pdf_filename).stem}_organized_data.json"
    save_organized_data(organized_data, output_filename)

    # #print(f"\n=== Processing Summary ===")
    # #print(f"Source PDF: {pdf_filename}")
    # #print(f"Total pages processed: {organized_data['metadata']['total_pages']}")
    # #print(f"Total images extracted: {organized_data['metadata']['total_images']}")
    # #print(f"Images saved to: ./extracted_images/")
    # #print(f"Organized data saved to: {output_filename}")

    if organized_data["products"]:
        product = organized_data["products"][0]
        # #print(f"\nProduct Information:")
        # #print(f"- Name: {product['product_name']}")
        # #print(f"- Model: {product['model_number']}")
        # #print(f"- Brand: {product['brand']}")
        # #print(f"- Specifications: {len(product['specifications'])} items")
        # #print(f"- Features: {len(product['features'])} items")
        # #print(f"- Tables: {len(product['tables'])} found")

    return organized_data


# if __name__ == "__main__":
#     print("Code ready! Use process_ocr_response(response_dict, pdf_filename) to organize your OCR data.")
