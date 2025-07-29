import json
import base64
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from mistralai import Mistral, DocumentURLChunk
from groq import Groq
from logger import setup_logger

logger = setup_logger("ocr_processor")

# Pydantic Models
class ProductSpecification(BaseModel):
    label: str
    value: str
    unit: Optional[str] = None

class ProductFeature(BaseModel):
    name: str
    description: Optional[str] = None
    category: Optional[str] = None

class ProductImage(BaseModel):
    id: str
    filename: str
    local_path: Optional[str] = None
    base64_data: str
    page_number: int
    size_estimate: int
    image_type: Optional[str] = "product"

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
    images: List[ProductImage] = []
    thumbnail_image: Optional[str] = None
    rating: Optional[str] = "4.5"
    review_count: Optional[str] = "128"
    raw_text: str

class OCRExtractionResult(BaseModel):
    products: List[ProductInfo]
    metadata: Dict[str, Any]

class StructuredPDFExtractor:
    def __init__(self, mistral_api_key: str, groq_api_key: str):
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.groq_client = Groq(api_key=groq_api_key)
        self.current_run_folder = None

    def create_run_folder(self, pdf_filename: str) -> str:
        """Create a unique folder for this extraction run"""
    
        pdf_stem = Path(pdf_filename).stem
        run_folder = f"dam/{pdf_stem}"
        os.makedirs(run_folder, exist_ok=True)
        self.current_run_folder = run_folder
        logger.info(f"Created extraction folder: {run_folder}")
        return run_folder

    def save_base64_image(self, base64_str: str, filename: str, output_dir: Optional[str] = None) -> str:
        """Save base64 image to JPG format in the specified directory"""
        if output_dir is None:
            output_dir = self.current_run_folder or "dam"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean base64 string
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Ensure filename has .jpg extension
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            filename = f"{Path(filename).stem}.jpg"
        
        try:
            image_data = base64.b64decode(base64_str)
            file_path = os.path.join(output_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Saved image: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving image {filename}: {e}")
            raise

    def process_pdf_with_ocr(self, pdf_path: Path) -> Dict[str, Any]:
        logger.info(f"Uploading PDF: {pdf_path}")
        uploaded_file = self.mistral_client.files.upload(
            file={
                "file_name": pdf_path.stem,
                "content": pdf_path.read_bytes(),
            },
            purpose="ocr",
        )
        signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        logger.info("Processing PDF with OCR...")
        pdf_response = self.mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        logger.debug(f"OCR Response: {pdf_response}")
        return pdf_response.model_dump()
    
    def extract_structured_content(self, ocr_response: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Extracting structured content using Groq LLM...")
        all_text = "\n".join([page.get("markdown", "") for page in ocr_response.get("pages", [])])
        print(all_text)
        print("**************+++++++++++++++======================+++++++++***************")

        prompt = f"""
            You are an expert system for extracting structured product information from scanned product catalogs.

            Extract the following fields in JSON format:
            - product_name
            - product_description
            - model_number
            - brand
            - category
            - specifications: list of {{label, value, unit}}
            - features: list of {{name, description}}

            Use only values from the input text.

            Input:
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
            logger.debug(f"LLM raw content: {content}")

            if not content or content.strip() == "":
                raise ValueError("❌ LLM returned empty response")

            # Extract JSON from response
            response_text = content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text)

        except Exception as e:
            logger.exception("❌ Failed to extract structured content from LLM")
            logger.debug(f"Prompt sent to LLM:\n{prompt}")
            logger.debug(f"Response received:\n{locals().get('response_text', 'N/A')}")
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
        logger.info("Extracting tables...")
        tables = []
        for page in ocr_response.get("pages", []):
            lines = page.get("markdown", "").split('\n')
            current_table = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if '\t' in line or '  ' in line:
                    columns = [col.strip() for col in line.split('\t') if col.strip()]
                    if not columns:
                        columns = [col.strip() for col in line.split('  ') if col.strip()]
                    if len(columns) >= 2:
                        if current_table is None:
                            current_table = ProductTable(
                                headers=columns, rows=[],
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
        return tables

    def _classify_table_type(self, headers: List[str]) -> str:
        header_text = " ".join(headers).lower()
        if any(word in header_text for word in ["spec", "specification", "technical", "dimension"]):
            return "specifications"
        elif any(word in header_text for word in ["feature", "benefit", "capability"]):
            return "features"
        elif any(word in header_text for word in ["compare", "comparison", "vs"]):
            return "comparison"
        return "general"

    def process_images(self, ocr_response: Dict[str, Any], pdf_filename: str) -> List[ProductImage]:
        logger.info("Processing images...")
        images = []
        image_counter = 1
        
        for page_idx, page in enumerate(ocr_response.get("pages", [])):
            page_images = page.get("images", [])
            logger.info(f"Found {len(page_images)} images on page {page_idx + 1}")
            
            for image in page_images:
                image_id = image.get("id", f"img_{image_counter}")
                # Create descriptive filename with page and image number
                image_filename = f"page_{page_idx + 1:02d}_image_{image_counter:02d}.jpg"
                
                try:
                    base64_data = image.get("image_base64", "")
                    if not base64_data:
                        logger.warning(f"No base64 data found for image {image_filename}")
                        continue
                    
                    # Save image to the current run folder
                    saved_path = self.save_base64_image(base64_data, image_filename, self.current_run_folder)
                    
                    images.append(ProductImage(
                        id=image_id,
                        filename=image_filename,
                        local_path=saved_path,
                        base64_data=base64_data,
                        page_number=page_idx + 1,
                        size_estimate=len(base64_data) * 3 // 4,  # Approximate size
                        image_type=self._classify_image_type(image_id, page_idx + 1)
                    ))
                    
                    logger.info(f"Processed image {image_counter}: {image_filename}")
                    image_counter += 1
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_filename}: {e}")
                    continue
        
        logger.info(f"Total images processed: {len(images)}")
        return images

    def _classify_image_type(self, image_id: str, page_number: int) -> str:
        """Classify image type based on ID and page number"""
        image_id_lower = image_id.lower()
        
        if page_number == 1:
            return "product"
        elif any(word in image_id_lower for word in ["diagram", "schematic", "blueprint"]):
            return "diagram"
        elif any(word in image_id_lower for word in ["chart", "graph", "plot"]):
            return "chart"
        elif any(word in image_id_lower for word in ["logo", "brand"]):
            return "logo"
        elif any(word in image_id_lower for word in ["table", "spec"]):
            return "table"
        else:
            return "product"

    def extract_from_pdf(self, pdf_path: Path) -> OCRExtractionResult:
        logger.info(f"Starting extraction from PDF: {pdf_path}")
        
        # Create a unique folder for this run
        self.create_run_folder(str(pdf_path))

        ocr_response = self.process_pdf_with_ocr(pdf_path)
        structured_content = self.extract_structured_content(ocr_response)
        images = self.process_images(ocr_response, str(pdf_path))
        tables = self.extract_tables_structured(ocr_response)

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
                            label=row[0], value=row[1],
                            unit=row[2] if len(row) > 2 else None
                        ))

        features = [ProductFeature(**f) for f in structured_content.get("features", [])]
        all_text = "\n".join([page.get("markdown", "") for page in ocr_response.get("pages", [])])

        product_info = ProductInfo(
            product_name=structured_content.get("product_name", ""),
            product_description=structured_content.get("product_description", ""),
            model_number=", ".join(structured_content.get("model_number")) if isinstance(structured_content.get("model_number"), list) else structured_content.get("model_number"),
            brand=structured_content.get("brand"),
            category=structured_content.get("category"),
            specifications=specifications,
            features=features,
            tables=tables,
            images=images,
            thumbnail_image=images[0].local_path if images else None,
            raw_text=all_text
        )

        result = OCRExtractionResult(
            source_pdf=str(pdf_path),
            extraction_timestamp=datetime.now().isoformat(),
            products=[product_info],
            metadata={
                "total_pages": len(ocr_response.get("pages", [])),
                "total_images": len(images),
                "total_text_length": len(all_text),
                "total_tables": len(tables),
                "images_folder": self.current_run_folder
            }
        )
        
        logger.info(f"Extraction completed. Images saved in: {self.current_run_folder}")
        return result

    def save_structured_data(self, result: OCRExtractionResult, output_path: Optional[str] = None, include_base64: bool = False):
        """Save structured data with option to exclude base64 data"""
        if output_path is None:
            # Save in the same folder as images
            output_path = os.path.join(self.current_run_folder, "structured_data.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a copy of the result for saving
        result_dict = result.model_dump()
        
        # Remove base64 data if requested (saves space and focuses on paths)
        if not include_base64:
            for product in result_dict.get("products", []):
                for image in product.get("images", []):
                    if "base64_data" in image:
                        del image["base64_data"]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Structured data saved to: {output_path}")

    def convert_to_frontend_format(self, result: OCRExtractionResult, use_paths: bool = True) -> Dict[str, Any]:
        """Convert to frontend format with option to use image paths instead of base64"""
        logger.info("Converting to frontend format...")
        products = []
        
        for product in result.products:
            if use_paths:
                # Use local file paths instead of base64 data
                image_paths = [img.local_path for img in product.images if img.local_path]
                main_image = image_paths[0] if image_paths else ""
                thumbnails = image_paths
            else:
                # Use base64 data (original behavior)
                thumbnails = [img.base64_data for img in product.images]
                main_image = thumbnails[0] if thumbnails else ""
            
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
                "rating": product.rating,
                "reviewCount": product.review_count,
                "detailedDescription": product.product_description,
                "features": features,
                "specifications": specs,
                "mainImage": main_image,
                "thumbnails": thumbnails,
                "images_folder": self.current_run_folder
            }
            
            # Add image metadata if using paths
            if use_paths:
                product_data["image_details"] = [
                    {
                        "id": img.id,
                        "filename": img.filename,
                        "local_path": img.local_path,
                        "page_number": img.page_number,
                        "image_type": img.image_type,
                        "size_estimate": img.size_estimate
                    }
                    for img in product.images
                ]
            
            products.append(product_data)
        
        return {"products": products}

    def save_frontend_format(self, result: OCRExtractionResult, output_path: Optional[str] = None, use_paths: bool = True):
        """Save data in frontend format with image paths"""
        if output_path is None:
            output_path = os.path.join(self.current_run_folder, "frontend_data.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        frontend_data = self.convert_to_frontend_format(result, use_paths=use_paths)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(frontend_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Frontend format data saved to: {output_path}")

    def print_extraction_summary(self, result: OCRExtractionResult):
        logger.info("\n" + "=" * 50)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Images Folder: {result.metadata.get('images_folder', 'N/A')}")
        logger.info(f"Total Pages: {result.metadata['total_pages']}")
        logger.info(f"Total Images: {result.metadata['total_images']}")
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
            logger.info(f"Images: {len(p.images)}")
            
            # List all saved images
            if p.images:
                logger.info("\nSaved Images:")
                for i, img in enumerate(p.images, 1):
                    logger.info(f"  {i}. {img.filename} -> {img.local_path} (Page {img.page_number}, Type: {img.image_type})")
        
        logger.info("=" * 50)


















