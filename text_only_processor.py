import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from mistralai import Mistral, DocumentURLChunk
from groq import Groq
from logger import setup_logger

logger = setup_logger("text_processor")

# Pydantic Models for Text-Only Processing
class ProductSpecification(BaseModel):
    label: str
    value: str
    unit: Optional[str] = None

class ProductFeature(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None

class ProductTable(BaseModel):
    headers: List[str]
    rows: List[List[str]]
    table_type: Optional[str] = None

class ProductInfo(BaseModel):
    product_name: str
    product_description: Optional[str] = None
    model_number: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    specifications: List[ProductSpecification] = []
    features: List[ProductFeature] = []
    tables: List[ProductTable] = []
    raw_text: str

class TextExtractionResult(BaseModel):
    products: List[ProductInfo]
    metadata: Dict[str, Any]

class TextOnlyPDFExtractor:
    def __init__(self, mistral_api_key: str, groq_api_key: str):
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.groq_client = Groq(api_key=groq_api_key)

    def process_pdf_with_ocr(self, pdf_path: Path) -> Dict[str, Any]:
        """Process PDF with Mistral OCR to extract text only"""
        logger.info(f"Uploading PDF for text extraction: {pdf_path}")
        
        uploaded_file = self.mistral_client.files.upload(
            file={
                "file_name": pdf_path.stem,
                "content": pdf_path.read_bytes(),
            },
            purpose="ocr",
        )
        
        signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        logger.info("Processing PDF with OCR for text extraction...")
        
        pdf_response = self.mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=False  # Don't include images, only text
        )
        
        # Convert response to dictionary format
        ocr_response = pdf_response.model_dump()
        logger.debug(f"OCR Response received for {len(ocr_response.get('pages', []))} pages")
        return ocr_response
    
    def extract_structured_content(self, ocr_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured product information from OCR text using Groq LLM"""
        logger.info("Extracting structured content from OCR text...")
        
        # Combine all text from all pages
        all_text = "\n".join([page.get("markdown", "") for page in ocr_response.get("pages", [])])
        
        logger.info(f"Total text extracted: {len(all_text)} characters")
        logger.debug(f"Extracted text preview: {all_text[:500]}...")

        prompt = f"""
        You are an expert system for extracting structured product information from scanned product catalogs.

        Extract the following fields in JSON format from the provided text:
        - product_name: The main product name/title
        - product_description: A comprehensive description of the product
        - model_number: Product model number or SKU
        - brand: Brand name
        - category: Product category
        - specifications: List of {{label, value, unit}} objects for technical specifications
        - features: List of {{name, description}} objects for product features

        Rules:
        1. Use only information present in the input text
        2. If a field is not found, use empty string or empty array
        3. For specifications, extract label-value pairs from tables or lists
        4. For features, extract feature names and descriptions
        5. Return valid JSON only

        Input text:
        \"\"\"
        {all_text}
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
            logger.debug(f"LLM raw response: {content}")

            if not content or content.strip() == "":
                raise ValueError("LLM returned empty response")

            # Extract JSON from response
            response_text = content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            structured_data = json.loads(response_text)
            logger.info("Successfully extracted structured content")
            return structured_data

        except Exception as e:
            logger.exception("Failed to extract structured content from LLM")
            logger.debug(f"Prompt sent to LLM:\n{prompt}")
            # Return empty content to avoid crashing the pipeline
            return {
                "product_name": "",
                "product_description": "",
                "model_number": "",
                "brand": "",
                "category": "",
                "specifications": [],
                "features": []
            }

    def extract_tables_structured(self, ocr_response: Dict[str, Any]) -> List[ProductTable]:
        """Extract tables from OCR text"""
        logger.info("Extracting tables from OCR text...")
        tables = []
        
        for page in ocr_response.get("pages", []):
            lines = page.get("markdown", "").split('\n')
            current_table = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for tab-separated or space-separated columns
                if '\t' in line or '  ' in line:
                    columns = [col.strip() for col in line.split('\t') if col.strip()]
                    if not columns:
                        columns = [col.strip() for col in line.split('  ') if col.strip()]
                    
                    if len(columns) >= 2:
                        if current_table is None:
                            current_table = ProductTable(
                                headers=columns, 
                                rows=[],
                                table_type=self._classify_table_type(columns)
                            )
                        else:
                            current_table.rows.append(columns)
                else:
                    if current_table is not None:
                        tables.append(current_table)
                        current_table = None
                        
            if current_table is not None:
                tables.append(current_table)
                
        logger.info(f"Extracted {len(tables)} tables")
        return tables

    def _classify_table_type(self, headers: List[str]) -> str:
        """Classify table type based on headers"""
        header_text = " ".join(headers).lower()
        if any(word in header_text for word in ["spec", "specification", "technical", "dimension"]):
            return "specifications"
        elif any(word in header_text for word in ["feature", "benefit", "capability"]):
            return "features"
        elif any(word in header_text for word in ["compare", "comparison", "vs"]):
            return "comparison"
        return "general"

    def extract_from_pdf(self, pdf_path: Path) -> TextExtractionResult:
        """Main extraction method - text only"""
        logger.info(f"Starting text-only extraction from PDF: {pdf_path}")

        # Process PDF with OCR
        ocr_response = self.process_pdf_with_ocr(pdf_path)
        
        # Extract structured content
        structured_content = self.extract_structured_content(ocr_response)
        
        # Extract tables
        tables = self.extract_tables_structured(ocr_response)

        # Build specifications from structured content and tables
        specifications = [
            ProductSpecification(**s)
            for s in structured_content.get("specifications", [])
        ]
        
        # Add specifications from tables
        for table in tables:
            if table.table_type == "specifications":
                for row in table.rows:
                    if len(row) >= 2:
                        specifications.append(ProductSpecification(
                            label=row[0], 
                            value=row[1],
                            unit=row[2] if len(row) > 2 else None
                        ))

        # Build features
        features = [ProductFeature(**f) for f in structured_content.get("features", [])]
        
        # Combine all text
        all_text = "\n".join([page.get("markdown", "") for page in ocr_response.get("pages", [])])

        # Create product info
        product_info = ProductInfo(
            product_name=structured_content.get("product_name", ""),
            product_description=structured_content.get("product_description", ""),
            model_number=", ".join(structured_content.get("model_number")) if isinstance(structured_content.get("model_number"), list) else structured_content.get("model_number"),
            brand=structured_content.get("brand"),
            category=structured_content.get("category"),
            specifications=specifications,
            features=features,
            tables=tables,
            raw_text=all_text
        )

        # Create result
        result = TextExtractionResult(
            products=[product_info],
            metadata={
                "source_pdf": str(pdf_path),
                "extraction_timestamp": datetime.now().isoformat(),
                "total_pages": len(ocr_response.get("pages", [])),
                "total_text_length": len(all_text),
                "total_tables": len(tables),
                "extraction_type": "text_only"
            }
        )
        
        logger.info(f"Text extraction completed successfully")
        return result

    def save_structured_data(self, result: TextExtractionResult, output_path: str):
        """Save structured text data to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        result_dict = result.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Structured text data saved to: {output_path}")

    def convert_to_frontend_format(self, result: TextExtractionResult) -> Dict[str, Any]:
        """Convert to frontend format (text only)"""
        logger.info("Converting to frontend format...")
        products = []
        
        for product in result.products:
            specs = [
                {"label": s.label, "value": f"{s.value} {s.unit or ''}".strip()}
                for s in product.specifications
            ]
            features = [
                {"name": f.name, "description": f.description or ""}
                for f in product.features
            ]
            
            product_data = {
                "product_name": product.product_name,
                "product_description": product.product_description,
                "category": product.category,
                "brand": product.brand,
                "model_number": product.model_number,
                "features": features,
                "specifications": specs,
                "tables": [table.model_dump() for table in product.tables],
                "raw_text": product.raw_text,
                "extraction_type": "text_only"
            }
            
            products.append(product_data)
        
        return {"products": products}

    def print_extraction_summary(self, result: TextExtractionResult):
        """Print summary of text extraction results"""
        logger.info("\n" + "=" * 50)
        logger.info("TEXT EXTRACTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Source PDF: {result.metadata['source_pdf']}")
        logger.info(f"Total Pages: {result.metadata['total_pages']}")
        logger.info(f"Total Tables: {result.metadata['total_tables']}")
        logger.info(f"Text Length: {result.metadata['total_text_length']} characters")
        
        if result.products:
            p = result.products[0]
            logger.info(f"\nProduct: {p.product_name}")
            logger.info(f"Brand: {p.brand}")
            logger.info(f"Model: {p.model_number}")
            logger.info(f"Category: {p.category}")
            logger.info(f"Specifications: {len(p.specifications)}")
            logger.info(f"Features: {len(p.features)}")
            logger.info(f"Tables: {len(p.tables)}")
            
            # Show text preview
            text_preview = p.raw_text[:200] + "..." if len(p.raw_text) > 200 else p.raw_text
            logger.info(f"\nText Preview: {text_preview}")
        
        logger.info("=" * 50) 