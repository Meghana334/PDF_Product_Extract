# Text-Only PDF Extraction with Mistral OCR

This directory contains tools for extracting **only text content** from PDFs using Mistral OCR, without processing or saving any images.

## Available Tools

### 1. Simple Text Extractor (`simple_text_extractor.py`)
Extracts raw text only - the simplest option.

**Usage:**
```bash
python simple_text_extractor.py <PDF_PATH> [OUTPUT_PATH]
```

**Examples:**
```bash
python simple_text_extractor.py data/sample.pdf
python simple_text_extractor.py data/sample.pdf output/my_text.txt
```

**Output:** Raw text file (.txt)

### 2. Structured Text Extractor (`text_main.py`)
Extracts text and structures it into product information using Groq LLM.

**Usage:**
```bash
python text_main.py [PDF_PATH] [OUTPUT_DIR]
python text_main.py --demo
python text_main.py --help
```

**Examples:**
```bash
python text_main.py data/sample.pdf output/
python text_main.py --demo
```

**Output:** 
- `*_text_structured.json` - Structured product data
- `*_text_frontend.json` - Frontend-ready format
- `*_raw_text.txt` - Raw extracted text

## Key Differences from Original

| Feature | Original (`main.py`) | Text-Only (`text_main.py`) |
|---------|---------------------|---------------------------|
| Text Extraction | ✅ | ✅ |
| Image Extraction | ✅ | ❌ |
| Image Saving | ✅ | ❌ |
| Structured Data | ✅ | ✅ |
| Processing Speed | Slower | Faster |
| Storage Usage | High | Low |

## API Configuration

Both tools use the same API keys as the original system:

```json
{
  "api_keys": {
    "mistral_api_key": "your_mistral_key",
    "groq_api_key": "your_groq_key"
  }
}
```

## Output Files

### Simple Text Extractor
- `output/filename_text.txt` - Raw extracted text

### Structured Text Extractor
- `output/filename_text_structured.json` - Complete structured data
- `output/filename_text_frontend.json` - Frontend-ready format
- `output/filename_raw_text.txt` - Raw text content

## Example Output Structure

```json
{
  "products": [
    {
      "product_name": "Product Name",
      "product_description": "Description",
      "brand": "Brand",
      "model_number": "Model",
      "category": "Category",
      "specifications": [
        {"label": "Spec", "value": "Value", "unit": "Unit"}
      ],
      "features": [
        {"name": "Feature", "description": "Description"}
      ],
      "raw_text": "Complete extracted text...",
      "extraction_type": "text_only"
    }
  ],
  "metadata": {
    "total_pages": 5,
    "total_text_length": 2500,
    "extraction_type": "text_only"
  }
}
```

## Benefits of Text-Only Extraction

1. **Faster Processing** - No image processing overhead
2. **Lower Storage** - No image files saved
3. **Simpler Output** - Focus on text content only
4. **Better for Text Analysis** - Clean text data for NLP tasks
5. **Reduced API Costs** - No image base64 data transfer

## When to Use Each Tool

### Use Simple Text Extractor when:
- You only need raw text
- No structured data required
- Quick text extraction needed

### Use Structured Text Extractor when:
- You need structured product information
- Want to extract specifications and features
- Need data for frontend applications

## Troubleshooting

1. **API Key Issues**: Ensure your Mistral and Groq API keys are valid
2. **File Not Found**: Check that the PDF path is correct
3. **Output Directory**: The tool will create output directories automatically
4. **Large PDFs**: Text extraction is generally faster than image extraction

## Performance Notes

- Text-only extraction is typically 2-3x faster than full extraction
- Storage usage is minimal (only text files)
- API costs are lower (no image data transfer)
- Memory usage is reduced 