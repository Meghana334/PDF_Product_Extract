import os
from PIL import Image
import io
import fitz  

def get_images(pdf_path):

    name = pdf_path.split("/")[-1].split(".")[0]
    doc = fitz.open(pdf_path)
    image_index = 0
    feature_images = {}
    product_image = {}

    for page_index, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        print(f"Page {page_index + 1}: Width = {w} pt, Height = {h} pt")

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            base_image = doc.extract_image(xref)

            width = base_image["width"]
            height = base_image["height"]
            image_bytes = base_image["image"]

            if height < 100 or width < 100:
                print(f"Skipping image {img_index+1} on page {page_index+1} due to small size: {width}x{height}")
                continue

            if int(bbox.x0) > w/2 and int(bbox.y0) > (h/2):

                feature_images[image_index] = {
                    "name": name,
                    "bbox": bbox,
                    "image_bytes": image_bytes,
                    "width": width,
                    "height": height,
                }  
            elif int(bbox.x0) < 10 or int(bbox.y0) < 10 :
                
                product_image[10] = {
                    "name": name,
                    "bbox": bbox,
                    "image_bytes": image_bytes,
                    "width": width,
                    "height": height,
                } 
            image_index += 1

    print(f"✅ Displayed {image_index} images from PDF.")
    # print(feature_images)

    import functools
    
    def compare_items(a, b):
        bbox_a = a[1]["bbox"]
        bbox_b = b[1]["bbox"]

        diff_x0 = abs(bbox_a.x0 - bbox_b.x0)

        if diff_x0 > 5:
            # If x0 difference is large, sort by x0
            return -1 if bbox_a.x0 < bbox_b.x0 else 1
        else:
            # If x0 is close, sort by y0
            return -1 if bbox_a.y0 < bbox_b.y0 else 1

    # Apply custom comparator
    sorted_feature_images = sorted(
        feature_images.items(),
        key=functools.cmp_to_key(compare_items)
    )

    # # Optional: convert back to dict
    # sorted_feature_images_dict = {idx: data for idx, data in sorted_feature_images}


    renumbered_feature_images = {
        new_idx: data for new_idx, (_, data) in enumerate(sorted_feature_images)
    }

    renumbered_feature_images.update(product_image)

    save_images(renumbered_feature_images)






def save_images(feature_images, output_base="test"):

    output_base = "test" 
    for image_index, data in feature_images.items():
        name = data["name"]
        image_bytes = data["image_bytes"]

        # Create output folder if it doesn't exist
        output_dir = os.path.join(output_base, name)
        os.makedirs(output_dir, exist_ok=True)

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Save as JPEG
        output_path = os.path.join(output_dir, f"{image_index}.jpg")
        image.save(output_path, format="JPEG")

    print(f"✅ Saved: {output_path}")



if __name__ == "__main__":
    folder_path = r'K:\catalog check\PDF_Product_Extract/data'

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                get_images(pdf_path)
