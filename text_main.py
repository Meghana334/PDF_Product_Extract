import json
import os
from pathlib import Path
from text_only_processor import TextOnlyPDFExtractor
from logger import setup_logger

logger = setup_logger("text_main")

# === Load API Keys ===
DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY",None)
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# === Load config.json if available ===
def load_config(config_path: str = "config.json"):
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None

# === Process Single PDF (Text Only) ===
def process_single_pdf_text_only(pdf_path: str, mistral_api_key: str, groq_api_key: str, output_dir: str = "output"):
    logger.info(f"Processing PDF for text extraction: {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)

    extractor = TextOnlyPDFExtractor(
        mistral_api_key=mistral_api_key,
        groq_api_key=groq_api_key
    )

    result = extractor.extract_from_pdf(Path(pdf_path))
    extractor.print_extraction_summary(result)

    # Save structured text data
    output_structured = os.path.join(output_dir, f"{Path(pdf_path).stem}_text_structured.json")
    extractor.save_structured_data(result, output_structured)

    # Save frontend format
    frontend_data = extractor.convert_to_frontend_format(result)
    output_frontend = os.path.join(output_dir, f"{Path(pdf_path).stem}_text_frontend.json")
    with open(output_frontend, "w", encoding='utf-8') as f:
        json.dump(frontend_data, f, indent=2, ensure_ascii=False)

    # Save raw text separately
    if result.products:
        raw_text_output = os.path.join(output_dir, f"{Path(pdf_path).stem}_raw_text.txt")
        with open(raw_text_output, "w", encoding='utf-8') as f:
            f.write(result.products[0].raw_text)
        logger.info(f"✅ Saved raw text to: {raw_text_output}")

    logger.info(f"✅ Saved structured text JSON to: {output_structured}")
    logger.info(f"✅ Saved frontend text JSON to: {output_frontend}")

# === Demo Function (Text Only) ===
def demonstrate_text_extraction(mistral_api_key: str, groq_api_key: str):
    logger.info("=== PDF Text Extraction Demo ===")
    pdf_path = "data/two.pdf"

    if Path(pdf_path).exists():
        process_single_pdf_text_only(pdf_path, mistral_api_key, groq_api_key)
    else:
        logger.error(f"❌ File not found: {pdf_path}")

# === Main Entry Point ===
def main():
    logger.info("🚀 Starting Text-Only PDF Extraction System")
    logger.info("=" * 50)

    config = load_config()

    if config:
        logger.info("✅ Loaded configuration from config.json")
        mistral_key = config["api_keys"].get("mistral_api_key", DEFAULT_MISTRAL_API_KEY)
        groq_key = config["api_keys"].get("groq_api_key", DEFAULT_GROQ_API_KEY)
    else:
        logger.warning("⚠️  No config file found, using default values")
        mistral_key = DEFAULT_MISTRAL_API_KEY
        groq_key = DEFAULT_GROQ_API_KEY

    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--demo":
            demonstrate_text_extraction(mistral_key, groq_key)
        elif arg == "--help":
            print_help()
        else:
            pdf_path = arg
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
            if Path(pdf_path).exists():
                process_single_pdf_text_only(pdf_path, mistral_key, groq_key, output_dir)
            else:
                logger.error(f"❌ File not found: {pdf_path}")
    else:
        demonstrate_text_extraction(mistral_key, groq_key)

def print_help():
    print("""
Text-Only PDF Extraction Tool
============================

Usage:
  python text_main.py [PDF_PATH] [OUTPUT_DIR]
  python text_main.py --demo
  python text_main.py --help

Examples:
  python text_main.py data/sample.pdf output/
  python text_main.py --demo

This tool extracts ONLY text content from PDFs using Mistral OCR.
No images are processed or saved.
    """)

if __name__ == "__main__":
    main() 