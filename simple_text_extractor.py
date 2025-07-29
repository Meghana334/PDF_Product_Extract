import os
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from logger import setup_logger

logger = setup_logger("simple_text_extractor")

class SimpleTextExtractor:
    def __init__(self, mistral_api_key: str):
        self.mistral_client = Mistral(api_key=mistral_api_key)

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

    def save_text_to_file(self, text: str, output_path: str):
        """Save extracted text to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Text saved to: {output_path}")

def main():
    import sys
    
    # Default API key
    DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "j3QqwjYyXHa692fQKr3hawypMuYCdRmE")
    
    if len(sys.argv) < 2:
        print("Usage: python simple_text_extractor.py <PDF_PATH> [OUTPUT_PATH]")
        print("Example: python simple_text_extractor.py data/sample.pdf output/text.txt")
        return
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"output/{Path(pdf_path).stem}_text.txt"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    # Extract text
    extractor = SimpleTextExtractor(DEFAULT_MISTRAL_API_KEY)
    text = extractor.extract_raw_text(Path(pdf_path))
    
    # Save text
    extractor.save_text_to_file(text, output_path)
    
    # Show preview
    preview = text[:500] + "..." if len(text) > 500 else text
    print(f"\nText Preview:\n{preview}")

if __name__ == "__main__":
    main() 