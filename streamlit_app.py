import streamlit as st
from pathlib import Path
import os
import json
from pybars import Compiler
from mistralai import Mistral, DocumentURLChunk
from main import process_single_pdf, convert_json_format
import base64
import webbrowser
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket

from dotenv import load_dotenv
load_dotenv()

# === Settings ===
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
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

# === Find available port ===
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

# === Start local server ===
def start_local_server(port, directory="."):
    os.chdir(directory)
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer(("localhost", port), handler)
    httpd.serve_forever()

# === Convert image to base64 data URI ===
def image_to_base64_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        ext = image_path.split('.')[-1]
        base64_img = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{ext};base64,{base64_img}"

# === HTML rendering function ===
def render_html_handlebars(product_data: dict) -> str:
    logo_data_uri = image_to_base64_data_uri(LOGO_PATH)

    # Convert features to flat list for display
    features_flat = [
        f"{list(item.keys())[0]}: {list(item.values())[0]}"
        for item in product_data.get("features", [])
    ]
    
    # Convert images to base64 data URIs for compatibility
    main_image_uri = image_to_base64_data_uri(product_data["mainImage"]) if os.path.exists(product_data["mainImage"]) else product_data["mainImage"]
    
    thumbnail_uris = []
    for thumb in product_data.get("thumbnails", []):
        if os.path.exists(thumb):
            thumbnail_uris.append(image_to_base64_data_uri(thumb))
        else:
            thumbnail_uris.append(thumb)
    
    context = {
        "mainImage": main_image_uri,
        "productName": product_data["product_name"],
        "category": product_data["category"],
        "description": product_data["product_description"],
        "rating": product_data["rating"],
        "reviewCount": product_data["reviewCount"],
        "detailedDescription": product_data["detailedDescription"],
        "specifications": product_data["specifications"],
        "features": features_flat,
        "thumbnails": thumbnail_uris,
        "logo": logo_data_uri,
        # Add JSON data for JavaScript
        "featuresJSON": json.dumps(features_flat),
        "thumbnailsJSON": json.dumps(thumbnail_uris)
    }

    rendered_html = template(context)
    
    # Replace the JavaScript data injection placeholder
    rendered_html = rendered_html.replace('{{{features}}}', json.dumps(features_flat))
    rendered_html = rendered_html.replace('{{{thumbnailsJSON}}}', json.dumps(thumbnail_uris))
    
    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    return OUTPUT_HTML_PATH

# === Open HTML in new tab ===
def open_html_in_browser(html_path):
    # Get current working directory
    current_dir = os.getcwd()
    
    # Find a free port
    port = find_free_port()
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=start_local_server, 
        args=(port, current_dir),
        daemon=True
    )
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(1)
    
    # Open the HTML file in browser
    url = f"http://localhost:{port}/{os.path.basename(html_path)}"
    webbrowser.open_new_tab(url)
    
    return url

# === Streamlit UI ===
st.set_page_config(page_title="PDF → Product Page", layout="centered")
st.title("📄 Product Page Generator")

# Initialize session state for server management
if 'server_started' not in st.session_state:
    st.session_state.server_started = False
if 'server_url' not in st.session_state:
    st.session_state.server_url = None

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
            
            # Open in new tab
            try:
                url = open_html_in_browser(html_path)
                st.session_state.server_url = url
                st.session_state.server_started = True
                
                st.info(f"🌐 Product page opened in new tab at: {url}")
                st.info("📝 Note: Keep this Streamlit app running to serve the HTML file")
                
                # Show clickable link as backup
                st.markdown(f"**[Click here if the tab didn't open automatically]({url})**")
                
            except Exception as e:
                st.error(f"❌ Error opening in browser: {str(e)}")
                st.info("💡 You can download the HTML file and open it manually")

            # Still provide download option
            with open(html_path, "rb") as f:
                st.download_button(
                    label="⬇ Download Product HTML",
                    data=f,
                    file_name="product_page.html",
                    mime="text/html"
                )
        else:
            st.warning("⚠ No product extracted from PDF.")

# Display server status
if st.session_state.server_started:
    st.sidebar.success("🟢 Local server is running")
    if st.session_state.server_url:
        st.sidebar.info(f"Server URL: {st.session_state.server_url}")
    st.sidebar.warning("Keep this app running to serve the HTML file")
else:
    st.sidebar.info("🔴 No server running")