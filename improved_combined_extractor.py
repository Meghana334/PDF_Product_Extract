import json
import os
import base64
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from PIL import Image
import io
import fitz
from mistralai import Mistral, DocumentURLChunk
from groq import Groq
from logger import setup_logger
from dotenv import load_dotenv
load_dotenv()

logger = setup_logger("improved_combined_extractor")

class ImprovedCombinedExtractor:
    def __init__(self, mistral_api_key: str, groq_api_key: str):
        self.mistral_client = Mistral(api_key=mistral_api_key)
        self.groq_client = Groq(api_key=groq_api_key)

    def extract_images_from_pdf(self, pdf_path: str, output_base: str = "dam") -> Dict[str, Any]:
        """Extract images using the existing image.py logic with improved naming"""
        logger.info(f"Extracting images from: {pdf_path}")
        
        name = Path(pdf_path).stem
        doc = fitz.open(pdf_path)
        image_index = 0
        feature_images = {}
        product_image = {}
        extracted_images = []

        for page_index, page in enumerate(doc):
            w, h = page.rect.width, page.rect.height
            logger.info(f"Page {page_index + 1}: Width = {w} pt, Height = {h} pt")

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                bbox = page.get_image_bbox(img)
                base_image = doc.extract_image(xref)

                width = base_image["width"]
                height = base_image["height"]
                image_bytes = base_image["image"]

                if height < 100 or width < 100:
                    logger.info(f"Skipping image {img_index+1} on page {page_index+1} due to small size: {width}x{height}")
                    continue

                # Determine image type based on position
                image_type = "product"  # Default to product
                if int(bbox.x0) > w/2 and int(bbox.y0) > (h/2):
                    feature_images[image_index] = {
                        "name": name,
                        "bbox": bbox,
                        "image_bytes": image_bytes,
                        "width": width,
                        "height": height,
                        "page_number": page_index + 1,
                        "image_type": "product"
                    }
                elif int(bbox.x0) < 10 or int(bbox.y0) < 10:
                    product_image[image_index] = {
                        "name": name,
                        "bbox": bbox,
                        "image_bytes": image_bytes,
                        "width": width,
                        "height": height,
                        "page_number": page_index + 1,
                        "image_type": "product"
                    }
                else:
                    feature_images[image_index] = {
                        "name": name,
                        "bbox": bbox,
                        "image_bytes": image_bytes,
                        "width": width,
                        "height": height,
                        "page_number": page_index + 1,
                        "image_type": "product"
                    }
                image_index += 1

        logger.info(f"✅ Found {image_index} images from PDF.")

        # Sort feature images
        import functools
        
        def compare_items(a, b):
            bbox_a = a[1]["bbox"]
            bbox_b = b[1]["bbox"]
            diff_x0 = abs(bbox_a.x0 - bbox_b.x0)
            if diff_x0 > 5:
                return -1 if bbox_a.x0 < bbox_b.x0 else 1
            else:
                return -1 if bbox_a.y0 < bbox_b.y0 else 1

        sorted_feature_images = sorted(
            feature_images.items(),
            key=functools.cmp_to_key(compare_items)
        )

        renumbered_feature_images = {
            new_idx: data for new_idx, (_, data) in enumerate(sorted_feature_images)
        }
        renumbered_feature_images.update(product_image)

        # Save images and collect metadata
        saved_images = self.save_images_with_metadata(renumbered_feature_images, output_base)
        
        return {
            "images": saved_images,
            "total_images": len(saved_images),
            "feature_images": len([img for img in saved_images if img["image_type"] == "feature"]),
            "product_images": len([img for img in saved_images if img["image_type"] == "product"])
        }

    def save_images_with_metadata(self, feature_images: Dict, output_base: str = "dam") -> List[Dict]:
        """Save images and return metadata with DAM URLs and base64 data"""
        saved_images = []
        
        if not feature_images:
            return saved_images
            
        # Get the product name from the first image
        product_name = list(feature_images.values())[0]["name"]
        
        for image_index, data in feature_images.items():
            name = data["name"]
            image_bytes = data["image_bytes"]
            image_type = data.get("image_type", "product")
            page_number = data.get("page_number", 1)

            # Create output folder
            output_dir = os.path.join(output_base, name)
            os.makedirs(output_dir, exist_ok=True)

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Create filename in the format: page_XX_image_XX.jpg
            filename = f"page_{page_number:02d}_image_{image_index + 1:02d}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Save as JPEG
            image.save(output_path, format="JPEG")

            # Calculate file size
            file_size = os.path.getsize(output_path)

            # Create base64 data
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            base64_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            # Create image metadata with DAM URL format and base64
            dam_url = f"{output_base}/{name}/{filename}"
            
            image_metadata = {
                "id": f"img-{image_index}.jpeg",
                "filename": filename,
                "local_path": dam_url,
                "base64_data": f"data:image/jpeg;base64,{base64_data}",
                "page_number": page_number,
                "image_type": image_type,
                "size_estimate": file_size
            }
            
            saved_images.append(image_metadata)
            logger.info(f"✅ Saved image: {output_path} (Type: {image_type})")

        return saved_images

    def extract_text_with_mistral(self, pdf_path: Path) -> str:
        """Extract text using Mistral OCR"""
        logger.info(f"Extracting text from: {pdf_path}")
        
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
        logger.info("Processing PDF with Mistral OCR...")
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

    def detect_category_from_text(self, raw_text: str) -> str:
        """Intelligently detect category from text content"""
        text_lower = raw_text.lower()
        
        # Common category patterns
        category_patterns = {
            "brush cutter": ["brush cutter", "brushcutter", "brush-cutter"],
            "chainsaw": ["chainsaw", "chain saw", "chain-saw"],
            "lawn mower": ["lawn mower", "lawnmower", "mower"],
            "trimmer": ["trimmer", "string trimmer", "line trimmer"],
            "blower": ["blower", "leaf blower", "garden blower"],
            "hedge trimmer": ["hedge trimmer", "hedge cutter"],
            "power tool": ["power tool", "power equipment"],
            "garden tool": ["garden tool", "garden equipment"],
            "outdoor equipment": ["outdoor equipment", "outdoor power equipment"],
            "construction tool": ["construction tool", "construction equipment"],
            "agricultural equipment": ["agricultural", "farm equipment", "farming"],
            "forestry equipment": ["forestry", "logging equipment"],
            "landscaping equipment": ["landscaping", "landscape equipment"]
        }
        
        # Check for category patterns
        for category, patterns in category_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return category.title()
        
        # Check for brand-specific categories
        if "maruyama" in text_lower:
            if "brush" in text_lower or "cutter" in text_lower:
                return "Brush Cutter"
            elif "chainsaw" in text_lower:
                return "Chainsaw"
            elif "blower" in text_lower:
                return "Blower"
        
        # Check for engine types that might indicate category
        if "cc" in text_lower or "engine" in text_lower:
            if "brush" in text_lower or "cutter" in text_lower:
                return "Brush Cutter"
            elif "chain" in text_lower:
                return "Chainsaw"
        
        return "Power Equipment"  # Default category

    def structure_text_to_json(self, raw_text: str) -> Dict[str, Any]:
        """Convert raw text to structured JSON using Groq LLM with improved category detection"""
        logger.info("Structuring text to JSON format...")
        
        # First, detect category from text
        detected_category = self.detect_category_from_text(raw_text)
        
        prompt = f"""
        You are an expert at structuring product information from raw text into a clean JSON format.

        Convert the following raw text into a structured JSON format with these fields:
        - product_name: The main product name/title (usually the model number)
        - product_description: A comprehensive description of the product
        - category: Product category (use: {detected_category})
        - brand: Brand name (extract from text)
        - model_number: Product model/SKU (usually same as product_name)
        - features: Array of {{name, description}} objects for product features
        - specifications: Array of {{label, value}} objects for technical specifications (NO unit field)

        Rules:
        1. Extract only information that's clearly present in the text
        2. Use the detected category: {detected_category}
        3. For specifications, look for label-value pairs in tables or lists
        4. For features, extract feature names and descriptions
        5. Brand is usually mentioned in the text (e.g., Maruyama, Honda, etc.)
        6. DO NOT include 'unit' field in specifications - only 'label' and 'value'
        7. Return valid JSON only

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
            
            # Ensure category is set (use detected if LLM didn't find one)
            if not structured_data.get("category"):
                structured_data["category"] = detected_category
            
            # Remove 'unit' from specifications if present
            specifications = structured_data.get("specifications", [])
            for spec in specifications:
                if "unit" in spec:
                    del spec["unit"]
            
            logger.info(f"Successfully structured text to JSON. Category: {structured_data.get('category')}")
            return structured_data

        except Exception as e:
            logger.exception("Failed to structure text to JSON")
            # Return basic structure with detected category
            return {
                "product_name": "",
                "product_description": "",
                "category": detected_category,
                "brand": "",
                "model_number": "",
                "features": [],
                "specifications": []
            }

    def extract_and_combine(self, pdf_path: str, output_dir: str = "output") -> Dict[str, Any]:
        """Main method: Extract images and text, then combine into structured JSON"""
        logger.info(f"Starting improved combined extraction from: {pdf_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        pdf_name = Path(pdf_path).stem
        
        # Extract images using image.py logic with DAM structure
        image_results = self.extract_images_from_pdf(pdf_path, "dam")
        
        # Extract text using Mistral OCR
        raw_text = self.extract_text_with_mistral(Path(pdf_path))
        
        # Structure text using Groq LLM with improved category detection
        structured_text = self.structure_text_to_json(raw_text)
        
        # Get image details
        image_details = image_results.get("images", [])
        
        # Create thumbnails and mainImage URLs (both DAM and base64)
        thumbnails_dam = [img["local_path"] for img in image_details]
        thumbnails_base64 = [img["base64_data"] for img in image_details]
        main_image_dam = thumbnails_dam[0] if thumbnails_dam else ""
        main_image_base64 = thumbnails_base64[0] if thumbnails_base64 else ""
        
        # Determine images folder
        images_folder = f"dam/{pdf_name}" if image_details else ""
        
        # Create the final JSON structure
        combined_data = {
            "category": [structured_text.get("category", "")],
            "products": [{
                "product_name": structured_text.get("product_name", ""),
                "product_description": structured_text.get("product_description", ""),
                "category": structured_text.get("category", ""),
                "rating": "4.5",
                "reviewCount": "128",
                "detailedDescription": "",
                "features": structured_text.get("features", []),
                "specifications": structured_text.get("specifications", []),
                "mainImage": main_image_dam,
                "mainImageBase64": main_image_base64,
                "thumbnails": thumbnails_dam,
                "thumbnailsBase64": thumbnails_base64,
                "images_folder": images_folder,
                "image_details": image_details
            }]
        }
        print(structured_text.get("product_name", "none"))

        
        
        # Save combined JSON
        output_path = os.path.join(output_dir, f"{pdf_name}_improved_combined.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        # Save raw text separately
        raw_text_path = os.path.join(output_dir, f"{pdf_name}_raw_text.txt")
        with open(raw_text_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        
        logger.info(f"✅ Saved improved combined JSON to: {output_path}")
        logger.info(f"✅ Saved raw text to: {raw_text_path}")
        
        # Print summary
        self.print_summary(combined_data, image_results, len(raw_text))
        
        return combined_data

    def print_summary(self, combined_data: Dict, image_results: Dict, text_length: int):
        """Print extraction summary"""
        logger.info("\n" + "=" * 50)
        logger.info("IMPROVED COMBINED EXTRACTION SUMMARY")
        logger.info("=" * 50)
        
        if combined_data.get("products"):
            product = combined_data["products"][0]
            logger.info(f"📄 Product: {product['product_name']}")
            logger.info(f"🏷️  Category: {product['category']}")
            logger.info(f"📂 Images Folder: {product['images_folder']}")
            logger.info(f"📋 Features: {len(product['features'])}")
            logger.info(f"⚙️  Specifications: {len(product['specifications'])}")
            logger.info(f"🖼️  Images: {len(product['image_details'])}")
            logger.info(f"📝 Text Length: {text_length} characters")
            
            # Show main image
            if product['mainImage']:
                logger.info(f"🖼️  Main Image: {product['mainImage']}")
        
        logger.info("=" * 50)

def main():
    import sys
    
    # Default API keys
    DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", None)
    DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    
    if len(sys.argv) < 2:
        print("Improved Combined PDF Extractor")
        print("=================================")
        print("Usage: python improved_combined_extractor.py <PDF_PATH> [OUTPUT_DIR]")
        print("Example: python improved_combined_extractor.py data/sample.pdf output/")
        print("\nThis tool combines image extraction with improved text extraction and category detection.")
        return
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    # Extract and combine
    extractor = ImprovedCombinedExtractor(DEFAULT_MISTRAL_API_KEY, DEFAULT_GROQ_API_KEY)
    result = extractor.extract_and_combine(pdf_path, output_dir)
    print(result)
    
    print(f"\n🎉 Improved combined extraction completed! Check the '{output_dir}' folder for results.")

if __name__ == "__main__":
    main()


# import json
# import os
# import base64
# import re
# from pathlib import Path
# from typing import Dict, List, Any
# from datetime import datetime
# from PIL import Image
# import io
# import fitz
# from mistralai import Mistral, DocumentURLChunk
# from groq import Groq
# from logger import setup_logger
# from dotenv import load_dotenv
# load_dotenv()

# logger = setup_logger("improved_combined_extractor")

# class ImprovedCombinedExtractor:
#     def __init__(self, mistral_api_key: str, groq_api_key: str):
#         self.mistral_client = Mistral(api_key=mistral_api_key)
#         self.groq_client = Groq(api_key=groq_api_key)

#     def extract_images_from_pdf(self, pdf_path: str, output_base: str = "dam") -> Dict[str, Any]:
#         """Extract images using the existing image.py logic"""
#         logger.info(f"Extracting images from: {pdf_path}")
        
#         name = Path(pdf_path).stem
#         doc = fitz.open(pdf_path)
#         image_index = 0
#         feature_images = {}
#         product_image = {}
#         extracted_images = []

#         for page_index, page in enumerate(doc):
#             w, h = page.rect.width, page.rect.height
#             logger.info(f"Page {page_index + 1}: Width = {w} pt, Height = {h} pt")

#             for img_index, img in enumerate(page.get_images(full=True)):
#                 xref = img[0]
#                 bbox = page.get_image_bbox(img)
#                 base_image = doc.extract_image(xref)

#                 width = base_image["width"]
#                 height = base_image["height"]
#                 image_bytes = base_image["image"]

#                 if height < 100 or width < 100:
#                     logger.info(f"Skipping image {img_index+1} on page {page_index+1} due to small size: {width}x{height}")
#                     continue

#                 # Determine image type based on position
#                 image_type = "feature"
#                 if int(bbox.x0) > w/2 and int(bbox.y0) > (h/2):
#                     feature_images[image_index] = {
#                         "name": name,
#                         "bbox": bbox,
#                         "image_bytes": image_bytes,
#                         "width": width,
#                         "height": height,
#                         "page_number": page_index + 1,
#                         "image_type": "feature"
#                     }
#                 elif int(bbox.x0) < 10 or int(bbox.y0) < 10:
#                     product_image[10] = {
#                         "name": name,
#                         "bbox": bbox,
#                         "image_bytes": image_bytes,
#                         "width": width,
#                         "height": height,
#                         "page_number": page_index + 1,
#                         "image_type": "product"
#                     }
#                 image_index += 1

#         logger.info(f"✅ Found {image_index} images from PDF.")

#         # Sort feature images
#         import functools
        
#         def compare_items(a, b):
#             bbox_a = a[1]["bbox"]
#             bbox_b = b[1]["bbox"]
#             diff_x0 = abs(bbox_a.x0 - bbox_b.x0)
#             if diff_x0 > 5:
#                 return -1 if bbox_a.x0 < bbox_b.x0 else 1
#             else:
#                 return -1 if bbox_a.y0 < bbox_b.y0 else 1

#         sorted_feature_images = sorted(
#             feature_images.items(),
#             key=functools.cmp_to_key(compare_items)
#         )

#         renumbered_feature_images = {
#             new_idx: data for new_idx, (_, data) in enumerate(sorted_feature_images)
#         }
#         renumbered_feature_images.update(product_image)

#         # Save images and collect metadata
#         saved_images = self.save_images_with_metadata(renumbered_feature_images, output_base)
        
#         return {
#             "images": saved_images,
#             "total_images": len(saved_images),
#             "feature_images": len([img for img in saved_images if img["image_type"] == "feature"]),
#             "product_images": len([img for img in saved_images if img["image_type"] == "product"])
#         }

#     def save_images_with_metadata(self, feature_images: Dict, output_base: str = "dam") -> List[Dict]:
#         """Save images and return metadata with base64 data"""
#         saved_images = []
        
#         for image_index, data in feature_images.items():
#             name = data["name"]
#             image_bytes = data["image_bytes"]
#             image_type = data.get("image_type", "feature")
#             page_number = data.get("page_number", 1)

#             # Create output folder
#             output_dir = os.path.join(output_base, name)
#             os.makedirs(output_dir, exist_ok=True)

#             # Convert bytes to PIL Image
#             image = Image.open(io.BytesIO(image_bytes))

#             # Save as JPEG
#             output_path = os.path.join(output_dir, f"{image_index}.jpg")
#             image.save(output_path, format="JPEG")

#             # Convert to base64 for JSON
#             img_buffer = io.BytesIO()
#             image.save(img_buffer, format="JPEG")
#             base64_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

#             # Create image metadata
#             image_metadata = {
#                 "id": f"img_{image_index}",
#                 "filename": f"{image_index}.jpg",
#                 "local_path": output_path,
#                 "base64_data": f"data:image/jpeg;base64,{base64_data}",
#                 "page_number": page_number,
#                 "width": data["width"],
#                 "height": data["height"],
#                 "image_type": image_type,
#                 "size_estimate": len(base64_data) * 3 // 4
#             }
            
#             saved_images.append(image_metadata)
#             logger.info(f"✅ Saved image: {output_path} (Type: {image_type})")

#         return saved_images

#     def extract_text_with_mistral(self, pdf_path: Path) -> str:
#         """Extract text using Mistral OCR"""
#         logger.info(f"Extracting text from: {pdf_path}")
        
#         # Upload PDF
#         uploaded_file = self.mistral_client.files.upload(
#             file={
#                 "file_name": pdf_path.stem,
#                 "content": pdf_path.read_bytes(),
#             },
#             purpose="ocr",
#         )
        
#         # Get signed URL
#         signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
#         # Process with OCR (text only)
#         logger.info("Processing PDF with Mistral OCR...")
#         pdf_response = self.mistral_client.ocr.process(
#             document=DocumentURLChunk(document_url=signed_url.url),
#             model="mistral-ocr-latest",
#             include_image_base64=False  # Only text, no images
#         )
        
#         # Convert response to dictionary format
#         ocr_response = pdf_response.model_dump()
        
#         # Combine all text from all pages
#         all_text = "\n".join([page.get("markdown", "") for page in ocr_response.get("pages", [])])
        
#         logger.info(f"Extracted {len(all_text)} characters of text from {len(ocr_response.get('pages', []))} pages")
#         return all_text

#     def detect_category_from_text(self, raw_text: str) -> str:
#         """Intelligently detect category from text content"""
#         text_lower = raw_text.lower()
        
#         # Common category patterns
#         category_patterns = {
#             "brush cutter": ["brush cutter", "brushcutter", "brush-cutter"],
#             "chainsaw": ["chainsaw", "chain saw", "chain-saw"],
#             "lawn mower": ["lawn mower", "lawnmower", "mower"],
#             "trimmer": ["trimmer", "string trimmer", "line trimmer"],
#             "blower": ["blower", "leaf blower", "garden blower"],
#             "hedge trimmer": ["hedge trimmer", "hedge cutter"],
#             "power tool": ["power tool", "power equipment"],
#             "garden tool": ["garden tool", "garden equipment"],
#             "outdoor equipment": ["outdoor equipment", "outdoor power equipment"],
#             "construction tool": ["construction tool", "construction equipment"],
#             "agricultural equipment": ["agricultural", "farm equipment", "farming"],
#             "forestry equipment": ["forestry", "logging equipment"],
#             "landscaping equipment": ["landscaping", "landscape equipment"]
#         }
        
#         # Check for category patterns
#         for category, patterns in category_patterns.items():
#             for pattern in patterns:
#                 if pattern in text_lower:
#                     return category.title()
        
#         # Check for brand-specific categories
#         if "maruyama" in text_lower:
#             if "brush" in text_lower or "cutter" in text_lower:
#                 return "Brush Cutter"
#             elif "chainsaw" in text_lower:
#                 return "Chainsaw"
#             elif "blower" in text_lower:
#                 return "Blower"
        
#         # Check for engine types that might indicate category
#         if "cc" in text_lower or "engine" in text_lower:
#             if "brush" in text_lower or "cutter" in text_lower:
#                 return "Brush Cutter"
#             elif "chain" in text_lower:
#                 return "Chainsaw"
        
#         return "Power Equipment"  # Default category

#     def structure_text_to_json(self, raw_text: str) -> Dict[str, Any]:
#         """Convert raw text to structured JSON using Groq LLM with improved category detection"""
#         logger.info("Structuring text to JSON format...")
        
#         # First, detect category from text
#         detected_category = self.detect_category_from_text(raw_text)
        
#         prompt = f"""
#         You are an expert at structuring product information from raw text into a clean JSON format.

#         Convert the following raw text into a structured JSON format with these fields:
#         - product_name: The main product name/title (usually the model number)
#         - product_description: A comprehensive description of the product
#         - category: Product category (detected as: {detected_category})
#         - brand: Brand name (extract from text)
#         - model_number: Product model/SKU (usually same as product_name)
#         - features: Array of {{name, description}} objects for product features
#         - specifications: Array of {{label, value, unit}} objects for technical specifications

#         Rules:
#         1. Extract only information that's clearly present in the text
#         2. Use the detected category: {detected_category}
#         3. For specifications, look for label-value pairs in tables or lists
#         4. For features, extract feature names and descriptions
#         5. Brand is usually mentioned in the text (e.g., Maruyama, Honda, etc.)
#         6. Return valid JSON only

#         Raw text:
#         \"\"\"
#         {raw_text}
#         \"\"\"

#         Output JSON:
#         """

#         try:
#             completion = self.groq_client.chat.completions.create(
#                 model="llama-3.1-8b-instant",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.1,
#                 max_completion_tokens=2048,
#             )
            
#             content = completion.choices[0].message.content
#             logger.debug(f"LLM response: {content}")

#             if not content or content.strip() == "":
#                 raise ValueError("LLM returned empty response")

#             # Extract JSON from response
#             response_text = content.strip()
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0]
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0]

#             structured_data = json.loads(response_text)
            
#             # Ensure category is set (use detected if LLM didn't find one)
#             if not structured_data.get("category"):
#                 structured_data["category"] = detected_category
            
#             logger.info(f"Successfully structured text to JSON. Category: {structured_data.get('category')}")
#             return structured_data

#         except Exception as e:
#             logger.exception("Failed to structure text to JSON")
#             # Return basic structure with detected category
#             return {
#                 "product_name": "",
#                 "product_description": "",
#                 "category": detected_category,
#                 "brand": "",
#                 "model_number": "",
#                 "features": [],
#                 "specifications": []
#             }

#     def extract_and_combine(self, pdf_path: str, output_dir: str = "output") -> Dict[str, Any]:
#         """Main method: Extract images and text, then combine into structured JSON"""
#         logger.info(f"Starting improved combined extraction from: {pdf_path}")
        
#         # Create output directory
#         os.makedirs(output_dir, exist_ok=True)
        
#         pdf_name = Path(pdf_path).stem
        
#         # Extract images using image.py logic
#         image_results = self.extract_images_from_pdf(pdf_path, output_dir)
        
#         # Extract text using Mistral OCR
#         raw_text = self.extract_text_with_mistral(Path(pdf_path))
        
#         # Structure text using Groq LLM with improved category detection
#         structured_text = self.structure_text_to_json(raw_text)
        
#         # Combine everything into final JSON
#         # 1. Remove 'unit' from specifications
#         specifications = structured_text.get("specifications", [])
#         for spec in specifications:
#             if "unit" in spec:
#                 del spec["unit"]

#         # 2. Set detailedDescription to empty string
#         detailed_description = ""

#         # 3. Insert images from output/test/ after name and description
#         import glob
#         import re
#         test_image_dir = os.path.join(output_dir, "test")
#         image_files = sorted(
#             glob.glob(os.path.join(test_image_dir, "*.jpg")),
#             key=lambda x: int(re.findall(r"(\d+)", os.path.basename(x))[0]) if re.findall(r"(\d+)", os.path.basename(x)) else x
#         )
#         images_for_json = []
#         for img_path in image_files:
#             with open(img_path, "rb") as img_f:
#                 import base64
#                 img_b64 = base64.b64encode(img_f.read()).decode("utf-8")
#                 images_for_json.append({
#                     "filename": os.path.basename(img_path),
#                     "base64_data": f"data:image/jpeg;base64,{img_b64}"
#                 })

#         combined_data = {
#             "products": [{
#                 "product_name": structured_text.get("product_name", ""),
#                 "product_description": structured_text.get("product_description", ""),
#                 "category": structured_text.get("category", ""),
#                 "brand": structured_text.get("brand", ""),
#                 "model_number": structured_text.get("model_number", ""),
#                 "rating": "4.5",
#                 "reviewCount": "128",
#                 "detailedDescription": detailed_description,
#                 "features": structured_text.get("features", []),
#                 "specifications": specifications,
#                 # Insert images after name/description
#                 "images": images_for_json,
#                 "mainImage": images_for_json[0]["base64_data"] if images_for_json else "",
#                 "thumbnails": [img["base64_data"] for img in images_for_json],
#                 "raw_text": raw_text,
#                 "extraction_type": "improved_combined",
#                 "source_pdf": pdf_name,
#                 "image_metadata": {
#                     "total_images": len(images_for_json),
#                     "feature_images": 0,
#                     "product_images": 0
#                 }
#             }],
#             "metadata": {
#                 "extraction_timestamp": datetime.now().isoformat(),
#                 "source_pdf": pdf_path,
#                 "extraction_type": "improved_combined",
#                 "total_images": len(images_for_json),
#                 "text_length": len(raw_text),
#                 "category_detection_method": "pattern_matching"
#             }
#         }
        
#         # Save combined JSON
#         output_path = os.path.join(output_dir, f"{pdf_name}_improved_combined.json")
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
#         # Save raw text separately
#         raw_text_path = os.path.join(output_dir, f"{pdf_name}_raw_text.txt")
#         with open(raw_text_path, 'w', encoding='utf-8') as f:
#             f.write(raw_text)
        
#         logger.info(f"✅ Saved improved combined JSON to: {output_path}")
#         logger.info(f"✅ Saved raw text to: {raw_text_path}")
        
#         # Print summary
#         self.print_summary(combined_data, image_results, len(raw_text))
        
#         return combined_data

#     def print_summary(self, combined_data: Dict, image_results: Dict, text_length: int):
#         """Print extraction summary"""
#         logger.info("\n" + "=" * 50)
#         logger.info("IMPROVED COMBINED EXTRACTION SUMMARY")
#         logger.info("=" * 50)
        
#         if combined_data.get("products"):
#             product = combined_data["products"][0]
#             logger.info(f"📄 Product: {product['product_name']}")
#             logger.info(f"🏷️  Brand: {product['brand']}")
#             logger.info(f"📂 Category: {product['category']}")
#             logger.info(f"🔢 Model: {product['model_number']}")
#             logger.info(f"📋 Features: {len(product['features'])}")
#             logger.info(f"⚙️  Specifications: {len(product['specifications'])}")
#             logger.info(f"🖼️  Images: {image_results['total_images']} (Features: {image_results['feature_images']}, Product: {image_results['product_images']})")
#             logger.info(f"📝 Text Length: {text_length} characters")
            
#             # Show text preview
#             text_preview = product['raw_text'][:200] + "..." if len(product['raw_text']) > 200 else product['raw_text']
#             logger.info(f"\n📖 Text Preview: {text_preview}")
        
#         logger.info("=" * 50)

# def main():
#     import sys
    
#     # Default API keys
#     DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", None)
#     DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    
#     if len(sys.argv) < 2:
#         print("Improved Combined PDF Extractor")
#         print("=================================")
#         print("Usage: python improved_combined_extractor.py <PDF_PATH> [OUTPUT_DIR]")
#         print("Example: python improved_combined_extractor.py data/sample.pdf output/")
#         print("\nThis tool combines image extraction with improved text extraction and category detection.")
#         return
    
#     pdf_path = sys.argv[1]
#     output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
#     if not Path(pdf_path).exists():
#         logger.error(f"PDF file not found: {pdf_path}")
#         return
    
#     # Extract and combine
#     extractor = ImprovedCombinedExtractor(DEFAULT_MISTRAL_API_KEY, DEFAULT_GROQ_API_KEY)
#     result = extractor.extract_and_combine(pdf_path, output_dir)
    
#     print(f"\n🎉 Improved combined extraction completed! Check the '{output_dir}' folder for results.")

# if __name__ == "__main__":
#     main() 