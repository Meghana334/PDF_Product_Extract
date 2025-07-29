from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI()

@app.get("/")
async def render_html():
    # Get the path to the HTML file in the parent directory
    html_file_path = os.path.join(os.path.dirname(__file__), "rendered_product.html")
    
    # Check if file exists
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path, media_type="text/html")
    else:
        return {"error": "HTML file not found"}

# Alternative route with custom path
@app.get("/page")
async def render_page():
    html_file_path = os.path.join(os.path.dirname(__file__), "rendered_page.html")
    
    if os.path.exists(html_file_path):
        return FileResponse(html_file_path, media_type="text/html")
    else:
        return {"error": "HTML file not found"}

