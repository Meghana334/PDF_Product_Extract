import json
import os
from pathlib import Path
from typing import Dict, List, Any
from groq import Groq
from logger import setup_logger

logger = setup_logger("text_converter")

class TextToFrontendConverter:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)

    def structure_text_to_frontend_format(self, raw_text: str, pdf_name: str = "unknown") -> Dict[str, Any]:
        """Convert raw extracted text to frontend-ready JSON format"""
        logger.info("Converting raw text to frontend format...")
        
        prompt = f"""
        You are an expert at structuring product information from raw text into a clean JSON format for frontend applications.

        Convert the following raw text into a structured JSON format with these fields:
        - product_name: The main product name/title
        - product_description: A comprehensive description
        - category: Product category
        - brand: Brand name
        - model_number: Product model/SKU
        - features: Array of {{name, description}} objects
        - specifications: Array of {{label, value, unit}} objects
        - rating: Default "4.5"
        - reviewCount: Default "128"

        Rules:
        1. Extract only information that's clearly present in the text
        2. Use empty strings for missing fields
        3. For specifications, look for label-value pairs in tables or lists
        4. For features, extract feature names and descriptions
        5. Return valid JSON only

        Raw text:
        \"\"\"
        {raw_text}
        \"\"\"

        Output JSON:
        """

        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=2048,
            )
            
            content = completion.choices[0].message.content
            logger.debug(f"LLM response: {content}")

            if not content or content.strip() == "":
                raise ValueError("LLM returned empty response")

            # Extract JSON from response
            response_text = content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            structured_data = json.loads(response_text)
            
            # Create frontend format
            frontend_data = {
                "products": [{
                    "product_name": structured_data.get("product_name", ""),
                    "product_description": structured_data.get("product_description", ""),
                    "category": structured_data.get("category", ""),
                    "brand": structured_data.get("brand", ""),
                    "model_number": structured_data.get("model_number", ""),
                    "rating": structured_data.get("rating", "4.5"),
                    "reviewCount": structured_data.get("reviewCount", "128"),
                    "detailedDescription": structured_data.get("product_description", ""),
                    "features": structured_data.get("features", []),
                    "specifications": structured_data.get("specifications", []),
                    "mainImage": "",  # No images in text-only mode
                    "thumbnails": [],  # No images in text-only mode
                    "raw_text": raw_text,
                    "extraction_type": "text_only",
                    "source_pdf": pdf_name
                }]
            }
            
            logger.info("Successfully converted text to frontend format")
            return frontend_data

        except Exception as e:
            logger.exception("Failed to convert text to frontend format")
            # Return basic structure with raw text
            return {
                "products": [{
                    "product_name": "",
                    "product_description": "",
                    "category": "",
                    "brand": "",
                    "model_number": "",
                    "rating": "4.5",
                    "reviewCount": "128",
                    "detailedDescription": "",
                    "features": [],
                    "specifications": [],
                    "mainImage": "",
                    "thumbnails": [],
                    "raw_text": raw_text,
                    "extraction_type": "text_only",
                    "source_pdf": pdf_name
                }]
            }

    def convert_text_file_to_frontend(self, text_file_path: str, output_path: str = None):
        """Convert a text file to frontend JSON format"""
        if not Path(text_file_path).exists():
            logger.error(f"Text file not found: {text_file_path}")
            return
        
        # Read the text file
        with open(text_file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        pdf_name = Path(text_file_path).stem.replace('_text', '').replace('_raw', '')
        
        # Convert to frontend format
        frontend_data = self.structure_text_to_frontend_format(raw_text, pdf_name)
        
        # Save to output file
        if output_path is None:
            output_path = text_file_path.replace('.txt', '_frontend.json')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Frontend JSON saved to: {output_path}")
        return frontend_data

def main():
    import sys
    
    # Default API key
    DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    
    if len(sys.argv) < 2:
        print("Usage: python text_to_frontend_converter.py <TEXT_FILE> [OUTPUT_PATH]")
        print("Example: python text_to_frontend_converter.py output/two_raw_text.txt output/frontend.json")
        return
    
    text_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert text to frontend format
    converter = TextToFrontendConverter(DEFAULT_GROQ_API_KEY)
    result = converter.convert_text_file_to_frontend(text_file, output_path)
    
    if result:
        print(f"\n✅ Successfully converted {text_file} to frontend format")
        print(f"📄 Product: {result['products'][0]['product_name']}")
        print(f"🏷️  Brand: {result['products'][0]['brand']}")
        print(f"📋 Features: {len(result['products'][0]['features'])}")
        print(f"⚙️  Specifications: {len(result['products'][0]['specifications'])}")

if __name__ == "__main__":
    main() 