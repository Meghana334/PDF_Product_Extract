import json
import base64
import os
import re
import ast
import unicodedata
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from groq import Groq
from logger import setup_logger, log_info

logger = setup_logger()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

def save_base64_image(base64_str: str, filename: str, output_dir: str = "extracted_images") -> str:
    os.makedirs(output_dir, exist_ok=True)
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    image_data = base64.b64decode(base64_str)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^\w\s.,:;()/-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_first_word_as_product_name(text: str) -> str:
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    words = [word.strip() for word in cleaned_text.split() if word.strip()]
    return words[0] if words else ""

def extract_product_info_from_text(text: str) -> Dict[str, Any]:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    log_info(logger, text)

    product_info = {
        "product_name": "",
        "product_description": "",
        "specifications": {},
        "features": [],
        "model_number": "",
        "brand": ""
    }

    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line or re.match(r'^\|.*\|$', line):
            continue
        first_word = extract_first_word_as_product_name(line)
        if first_word and len(first_word) > 2:
            product_info["product_name"] = first_word
            break

    for line in lines:
        line = line.strip()
        if not line or re.match(r'^\|.*\|$', line):
            continue

        model_match = re.search(r'Model[:\s]+([A-Z0-9\-]+)', line, re.IGNORECASE)
        if model_match:
            product_info["model_number"] = model_match.group(1)

        brand_match = re.search(r'Brand[:\s]+(\w+)', line, re.IGNORECASE)
        if brand_match:
            product_info["brand"] = brand_match.group(1)

        spec_match = re.match(r'^([A-Za-z\s]+):\s*(.+)$', line)
        if spec_match:
            key = spec_match.group(1).strip()
            value = spec_match.group(2).strip()
            product_info["specifications"][key] = value

        if not product_info["product_description"]:
            if '#' in line and '**' in line:
                desc_match = re.search(r'\*\*(.+?)\*\*', line)
                if desc_match:
                    product_info["product_description"] = desc_match.group(1).strip()

    if not product_info["product_name"]:
        for line in lines:
            line = line.strip()
            if len(line) > 2 and not line.isdigit():
                first_word = extract_first_word_as_product_name(line)
                if first_word:
                    product_info["product_name"] = first_word
                    break

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
                    current_table = {"headers": columns, "rows": []}
                else:
                    current_table["rows"].append(columns)
        else:
            if current_table is not None:
                tables.append(current_table)
                current_table = None

    if current_table is not None:
        tables.append(current_table)

    return tables

def generate_product_desc(product_input: str) -> dict:
    """
    Extract topic and description from a product input string.
    Returns a dictionary with format: {"topic": "description"}
    """
    prompt = f"""You are a text parser. Your job is to split a product input into topic and description.

Rules:
1. The topic is usually the first few words (product name/title)
2. The description is the rest of the text
3. Return ONLY a valid JSON dictionary in this exact format: {{"topic": "description"}}
4. Do not add, remove, or modify any words from the original input
5. Preserve all spacing and punctuation exactly as given
6. Do not include any explanation or additional text

Examples:
Input: "Treaded Clutch Shaft The solid steel inner drive-shaft is threaded at the clutch end, which eliminates vibration."
Output: {{"Treaded Clutch Shaft": "The solid steel inner drive-shaft is threaded at the clutch end, which eliminates vibration."}}

Input: "H.E.R.E Technology High Efficiency Recirculatory Engine A unique MARUYAMA engineered system that is low emission, high power and highly fuel efficient. EU stage 2 compliant."
Output: {{"H.E.R.E Technology High Efficiency Recirculatory Engine": "A unique MARUYAMA engineered system that is low emission, high power and highly fuel efficient. EU stage 2 compliant."}}

Now process this input:
Input: "{product_input}"
Output:"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated to use the better model
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        output_text = completion.choices[0].message.content.strip()
        log_info(logger, f"Raw LLM output: {output_text}")  # Debug log
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            
            # Handle single quotes in JSON
            if json_str.startswith("{'") and not json_str.startswith('{"'):
                json_str = json_str.replace("'", '"')
            
            try:
                # Try JSON first (more robust)
                result = json.loads(json_str)
                # Ensure result is a dictionary
                if isinstance(result, dict):
                    return result
                else:
                    log_info(logger, f"Result is not a dict: {type(result)}")
                    return fallback_parse(product_input)
            except json.JSONDecodeError:
                try:
                    # Fallback to ast.literal_eval for single-quoted strings
                    result = ast.literal_eval(json_match.group())
                    # Ensure result is a dictionary
                    if isinstance(result, dict):
                        return result
                    else:
                        log_info(logger, f"ast.literal_eval result is not a dict: {type(result)}")
                        return fallback_parse(product_input)
                except (ValueError, SyntaxError) as e:
                    log_info(logger, f"Failed to parse JSON/dict: {e}")
                    return fallback_parse(product_input)
        else:
            log_info(logger, "No valid JSON found in response")
            return fallback_parse(product_input)
            
    except Exception as e:
        log_info(logger, f"API Error: {e}")
        return fallback_parse(product_input)

def fallback_parse(product_input: str) -> dict:
    """
    Fallback method to parse product input using regex patterns.
    This runs if LLM fails to provide proper output.
    """
    # Common patterns for product titles
    patterns = [
        # Pattern 1: Title followed by description (multiple words in title)
        r'^([A-Z][A-Za-z\s&\.]+(?:Technology|Engine|System|Shaft|Component|Tool|Device|Machine))\s+(.+)$',
        # Pattern 2: Acronym/Technical term followed by description
        r'^([A-Z\.]+(?:\s+[A-Z][A-Za-z]+)*)\s+(.+)$',
        # Pattern 3: First sentence as title, rest as description
        r'^([^.]+\.)\s*(.+)$',
        # Pattern 4: First few capitalized words as title
        r'^((?:[A-Z][A-Za-z]*\s*){2,4})\s*(.+)$'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, product_input.strip())
        if match:
            topic = match.group(1).strip()
            description = match.group(2).strip()
            if len(topic) > 5 and len(description) > 10:  # Reasonable lengths
                return {topic: description}
    
    # If no pattern matches, try simple split on first sentence
    sentences = product_input.split('.')
    if len(sentences) >= 2:
        topic = sentences[0].strip()
        description = '.'.join(sentences[1:]).strip()
        if description:
            return {topic: description}
    
    # Last resort: split roughly in the middle
    words = product_input.split()
    if len(words) > 6:
        mid_point = min(6, len(words) // 3)  # Take first 3-6 words as topic
        topic = ' '.join(words[:mid_point])
        description = ' '.join(words[mid_point:])
        return {topic: description}
    
    # If all else fails, return the original text with a generic key
    return {"Feature": product_input}

def extract_features_from_image_sections(text: str) -> List[Dict[str, str]]:
    features = []
    
    # Split text by image markers to get all sections
    image_pattern = r"!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)"
    sections = re.split(image_pattern, text)
    
    # Process each section that has meaningful content
    feature_count = 0
    for idx, section in enumerate(sections):
        input_text = clean_text(section.strip())
        
        # Skip empty sections or very short content
        if not input_text or len(input_text) < 10:
            continue
            
        # Skip the first section (usually header/title content before any images)
        if idx == 0:
            continue
        
        # Skip the first feature (increment counter but don't add to features)
        if feature_count == 0:
            feature_count += 1
            log_info(logger, f"Skipping first feature in section {idx}")
            continue
            
        log_info(logger, f"Processing section {idx}: {input_text[:100]}...")
        result = generate_product_desc(input_text)
        
        # Ensure result is a dictionary and handle all cases
        if isinstance(result, dict) and result and not result.get("error"):
            # Clean the result keys and values
            clean_result = {clean_text(k): clean_text(v) for k, v in result.items()}
            features.append(clean_result)
            log_info(logger, f"Successfully extracted feature: {clean_result}")
        else:
            # Fallback: create a generic feature entry
            fallback_feature = {"Feature": input_text[:200]}
            features.append(fallback_feature)
            log_info(logger, f"Used fallback for feature (result type: {type(result)}): {fallback_feature}")
        
        # Stop after collecting 4 features (excluding the skipped first one)
        if len(features) >= 4:
            break
    
    # If we still don't have enough features, try alternative extraction
    if len(features) < 4:
        # Look for additional patterns that might indicate features
        additional_patterns = [
            r"!\[.*?\]\(.*?\)(.*?)(?=!\[.*?\]\(.*?\)|$)",  # Any image followed by text
            r"\*\*([^*]+)\*\*(.*?)(?=\*\*|$)",  # Bold text followed by description
            r"([A-Z][A-Za-z\s]+):\s*([^.]+\.)",  # Title: Description pattern
        ]
        
        skip_first_additional = True  # Skip first feature from additional patterns too
        for pattern in additional_patterns:
            if len(features) >= 4:
                break
                
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if len(features) >= 4:
                    break
                
                if skip_first_additional:
                    skip_first_additional = False
                    continue
                    
                if isinstance(match, tuple) and len(match) == 2:
                    title, desc = match
                    title = clean_text(title.strip())
                    desc = clean_text(desc.strip())
                    
                    if len(title) > 3 and len(desc) > 10:
                        feature_dict = {title: desc}
                        # Avoid duplicates
                        if not any(list(existing.keys())[0] == title for existing in features):
                            features.append(feature_dict)
                            log_info(logger, f"Added additional feature: {feature_dict}")

    return features[:4]  # Return exactly 4 features or less if not available

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
            image_id = image.get("id", f"img_{image_counter}")
            if image_id in ["img-1.jpeg", "img-6.jpeg"]:
                continue

            image_filename = f"page_{page_idx + 1}image{image_counter}.jpg"
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
    features = extract_features_from_image_sections(all_text)
    log_info(logger, f"Extracted features: {features}")

    product_data = {
        "product_name": product_info["product_name"],
        "product_description": product_info["product_description"],
        "model_number": product_info["model_number"],
        "brand": product_info["brand"],
        "specifications": product_info["specifications"],
        "features": features,
        "tables": tables,
        "product_images": [],
        "thumbnail_image": "",
        "all_page_images": organized_data["all_extracted_images"],
        "raw_text": all_text,
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
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def process_ocr_response(ocr_response_dict: Dict[str, Any], pdf_filename: str):
    organized_data = organize_ocr_response(ocr_response_dict, pdf_filename)
    log_info(logger, organized_data)
    output_filename = f"{Path(pdf_filename).stem}_organized_data.json"
    save_organized_data(organized_data, output_filename)
    return organized_data