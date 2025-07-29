import json
import os
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from groq import Groq
from logger import setup_logger

logger = setup_logger("enhanced_text_extractor")

class EnhancedTextExtractor:
    def __init__(self, mistral_api_key: str, groq_api_key: str):
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.groq_client = Groq(api_key=groq_api_key)

    def extract_raw_text(self, pdf_path: Path) -> str:
        """Extract only raw text from PDF using Mistral OCR"""
        logger.info(f"Extracting raw text from: {pdf_path}")
        
        # Upload PDF
        uploaded_file = self.mistral_client.files.upload(
            file={
                "file_name": pdf_path.stem,
                "content": pdf_path.read_bytes(),
            },
            purpose="ocr",
        )
        
        # Get signed URL
        signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
        # Process with OCR (text only)
        logger.info("Processing PDF with OCR for text extraction...")
        pdf_response = self.mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=False  # Only text, no images
        )
        
        # Convert response to dictionary format
        ocr_response = pdf_response.model_dump()
        
        # Combine all text from all pages
        all_text = "\n".join([page.get("markdown", "") for page in ocr_response.get("pages", [])])
        
        logger.info(f"Extracted {len(all_text)} characters of text from {len(ocr_response.get('pages', []))} pages")
        return all_text

    def structure_text_to_frontend_format(self, raw_text: str, pdf_name: str = "unknown") -> dict:
        """Convert raw text to frontend-ready JSON format"""
        logger.info("Converting text to frontend format...")
        
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
                    "rating": "4.5",
                    "reviewCount": "128",
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

    def extract_and_structure(self, pdf_path: str, output_dir: str = "output") -> dict:
        """Extract text and structure it into frontend format"""
        logger.info(f"Starting enhanced text extraction from: {pdf_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract raw text
        raw_text = self.extract_raw_text(Path(pdf_path))
        
        # Save raw text
        pdf_stem = Path(pdf_path).stem
        raw_text_path = os.path.join(output_dir, f"{pdf_stem}_raw_text.txt")
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        logger.info(f"✅ Saved raw text to: {raw_text_path}")
        
        # Convert to frontend format
        frontend_data = self.structure_text_to_frontend_format(raw_text, pdf_stem)
        
        # Save frontend JSON
        frontend_path = os.path.join(output_dir, f"{pdf_stem}_frontend.json")
        with open(frontend_path, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved frontend JSON to: {frontend_path}")
        
        # Print summary
        self.print_summary(frontend_data, raw_text)
        
        return frontend_data

    def print_summary(self, frontend_data: dict, raw_text: str):
        """Print extraction summary"""
        logger.info("\n" + "=" * 50)
        logger.info("ENHANCED TEXT EXTRACTION SUMMARY")
        logger.info("=" * 50)
        
        if frontend_data.get("products"):
            product = frontend_data["products"][0]
            logger.info(f"📄 Product: {product['product_name']}")
            logger.info(f"🏷️  Brand: {product['brand']}")
            logger.info(f"🔢 Model: {product['model_number']}")
            logger.info(f"📂 Category: {product['category']}")
            logger.info(f"📋 Features: {len(product['features'])}")
            logger.info(f"⚙️  Specifications: {len(product['specifications'])}")
            logger.info(f"📝 Text Length: {len(raw_text)} characters")
            
            # Show text preview
            text_preview = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
            logger.info(f"\n📖 Text Preview: {text_preview}")
        
        logger.info("=" * 50)

def main():
    import sys
    
    # Default API keys
    DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "j3QqwjYyXHa692fQKr3hawypMuYCdRmE")
    DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    
    if len(sys.argv) < 2:
        print("Enhanced Text Extractor")
        print("======================")
        print("Usage: python enhanced_text_extractor.py <PDF_PATH> [OUTPUT_DIR]")
        print("Example: python enhanced_text_extractor.py data/sample.pdf output/")
        print("\nThis tool extracts text and automatically creates frontend JSON format.")
        return
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    # Extract and structure
    extractor = EnhancedTextExtractor(DEFAULT_MISTRAL_API_KEY, DEFAULT_GROQ_API_KEY)
    result = extractor.extract_and_structure(pdf_path, output_dir)
    
    print(f"\n🎉 Extraction completed! Check the '{output_dir}' folder for results.")

if __name__ == "__main__":
    main() 