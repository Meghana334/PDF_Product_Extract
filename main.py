import json
import os
from pathlib import Path
from ocr_processor import StructuredPDFExtractor
from logger import setup_logger

logger = setup_logger("main")

# === Load API Keys ===
DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY",None)
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# === Load config.json if available ===
def load_config(config_path: str = "config.json"):
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None

# === Create Sample Config ===
def create_sample_config():
    config = {
        "api_keys": {
            "mistral_api_key": DEFAULT_MISTRAL_API_KEY,
            "groq_api_key": DEFAULT_GROQ_API_KEY
        },
        "settings": {
            "output_directory": "output",
            "image_output_directory": "extracted_images",
            "include_raw_text": True,
            "generate_thumbnails": True,
            "skip_image_ids": ["img-1.jpeg", "img-6.jpeg"]
        }
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("✅ Sample config.json created")

# === Process Single PDF ===
def process_single_pdf(pdf_path: str, mistral_api_key: str, groq_api_key: str, output_dir: str = "output"):
    logger.info(f"Processing PDF: {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)

    extractor = StructuredPDFExtractor(
        mistral_api_key=mistral_api_key,
        groq_api_key=groq_api_key
    )

    result = extractor.extract_from_pdf(Path(pdf_path))
    extractor.print_extraction_summary(result)

    output_structured = os.path.join(output_dir, f"{Path(pdf_path).stem}_structured.json")
    extractor.save_structured_data(result, output_structured)

    frontend_data = extractor.convert_to_frontend_format(result)
    output_frontend = os.path.join(output_dir, f"{Path(pdf_path).stem}_frontend.json")
    with open(output_frontend, "w", encoding='utf-8') as f:
        json.dump(frontend_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved structured JSON to: {output_structured}")
    logger.info(f"✅ Saved frontend JSON to: {output_frontend}")

# === Demo Function ===
def demonstrate_extraction(mistral_api_key: str, groq_api_key: str):
    logger.info("=== PDF Extraction Demo ===")
    pdf_path = "PDF_Product_Extract/data/test.pdf"

    if Path(pdf_path).exists():
        process_single_pdf(pdf_path, mistral_api_key, groq_api_key)
    else:
        logger.error(f"❌ File not found: {pdf_path}")

# === Main Entry Point ===
def main():
    logger.info("🚀 Starting Structured PDF Extraction System")
    logger.info("=" * 50)

    config = load_config()

    if config:
        logger.info("✅ Loaded configuration from config.json")
        mistral_key = config["api_keys"].get("mistral_api_key", DEFAULT_MISTRAL_API_KEY)
        groq_key = config["api_keys"].get("groq_api_key", DEFAULT_GROQ_API_KEY)
    else:
        logger.warning("⚠️  No config file found, using default values")
        logger.info("💡 Run with --create-config to generate one")
        mistral_key = DEFAULT_MISTRAL_API_KEY
        groq_key = DEFAULT_GROQ_API_KEY

    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--create-config":
            create_sample_config()
        # elif arg == "--help":
        #     print_help()
        elif arg == "--demo":
            demonstrate_extraction(mistral_key, groq_key)
        else:
            pdf_path = arg
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
            if Path(pdf_path).exists():
                process_single_pdf(pdf_path, mistral_key, groq_key, output_dir)
            else:
                logger.error(f"❌ File not found: {pdf_path}")
    else:
        demonstrate_extraction(mistral_key, groq_key)



if __name__ == "__main__":
    create_sample_config()
    config = load_config()

    if config:
        logger.info("✅ Loaded configuration from config.json")
        mistral_key = config["api_keys"].get("mistral_api_key", DEFAULT_MISTRAL_API_KEY)
        groq_key = config["api_keys"].get("groq_api_key", DEFAULT_GROQ_API_KEY)
    else:
        logger.warning("⚠️  No config file found, using default values")
        logger.info("💡 Run with --create-config to generate one")
        mistral_key = DEFAULT_MISTRAL_API_KEY
        groq_key = DEFAULT_GROQ_API_KEY

    demonstrate_extraction(mistral_api_key=mistral_key,groq_api_key=groq_key)





