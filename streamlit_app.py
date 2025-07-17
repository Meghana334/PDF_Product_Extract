import streamlit as st
from pathlib import Path
import os
import json
from pybars import Compiler
from mistralai import Mistral, DocumentURLChunk
from main import process_ocr_response, convert_json_format
import base64

# === Settings ===
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "j3QqwjYyXHa692fQKr3hawypMuYCdRmE")
UPLOAD_DIR = "uploads"
TEMPLATE_PATH = "template.html"
OUTPUT_HTML_PATH = "rendered_product.html"
JSON_OUTPUT_PATH = "data.json"
LOGO_PATH = "static/maruyama-logo.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Handlebars template ===
compiler = Compiler()
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    template_src = f.read()
    template = compiler.compile(template_src)

# === Convert image to base64 data URI ===
def image_to_base64_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        ext = image_path.split('.')[-1]
        base64_img = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{ext};base64,{base64_img}"

# === HTML rendering function ===
def render_html_handlebars(product_data: dict) -> str:
    logo_data_uri = image_to_base64_data_uri(LOGO_PATH)

    features_flat = [
        f"{list(item.keys())[0]}: {list(item.values())[0]}"
        for item in product_data.get("features", [])
    ]
    
    context = {
        "mainImage": product_data["mainImage"],
        "productName": product_data["product_name"],
        "category": product_data["category"],
        "description": product_data["product_description"],
        "rating": product_data["rating"],
        "reviewCount": product_data["reviewCount"],
        "detailedDescription": product_data["detailedDescription"],
        "specifications": product_data["specifications"],
        "features": features_flat,
        "thumbnails": product_data["thumbnails"],
        "logo": logo_data_uri
    }

    rendered_html = template(context)
    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    return OUTPUT_HTML_PATH

# === Streamlit UI ===
st.set_page_config(page_title="PDF → Product Page", layout="centered")
st.title("📄 Product Page Generator")

uploaded_pdf = st.file_uploader("Upload your product PDF", type=["pdf"])

if uploaded_pdf:
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_pdf.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())
    st.success("✅ PDF uploaded.")

    if st.button("🚀 Run Extraction"):
        with st.spinner("Converting pdf to product page"):
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            with open(pdf_path, "rb") as f:
                uploaded = mistral_client.files.upload(
                    file={"file_name": Path(pdf_path).stem, "content": f.read()},
                    purpose="ocr"
                )
            signed_url = mistral_client.files.get_signed_url(file_id=uploaded.id, expiry=1)
            ocr_response = mistral_client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            ocr_dict = ocr_response.model_dump()

            organized = process_ocr_response(ocr_dict, pdf_path)
            json_input = f"{Path(pdf_path).stem}_organized_data.json"
            convert_json_format(json_input, JSON_OUTPUT_PATH)

        with open(JSON_OUTPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data["products"]:
            product = data["products"][0]
            html_path = render_html_handlebars(product)

            st.success("✅ Product HTML generated!")

            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            st.components.v1.html(html_content, height=1200, scrolling=True)

            with open(html_path, "rb") as f:
                st.download_button(
                    label="⬇ Download Product HTML",
                    data=f,
                    file_name="product_page.html",
                    mime="text/html"
                )
        else:
            st.warning("⚠ No product extracted from PDF.")
