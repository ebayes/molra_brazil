import csv
import json

def csv_to_coco_json(csv_filepath, json_filepath):
    # Define the structure of COCO JSON
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # A set to keep track of image ids and category ids already added
    image_ids = set()
    category_ids = {}
    next_category_id = 1  # Start assigning category IDs from 1
    
    # Read the CSV file
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            image_name = row['image_name']
            image_width = int(row['image_width'])
            image_height = int(row['image_height'])
            category_name = row['label']
            
            # Calculate absolute bounding box coordinates
            x_center = float(row['x_center']) * image_width
            y_center = float(row['y_center']) * image_height
            width = float(row['width']) * image_width
            height = float(row['height']) * image_height
            xmin = x_center - width / 2
            xmax = x_center + width / 2
            ymin = y_center - height / 2
            ymax = y_center + height / 2
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]  # [xmin, ymin, width, height]
            
            score = 1.0  # Assuming a default score of 1.0, adjust as necessary
            
            # Generate a unique image ID based on image name if not already added
            if image_name not in image_ids:
                image_id = len(image_ids) + 1  # Generate a new ID
                coco_format['images'].append({
                    "id": image_id,
                    "file_name": image_name,
                    "width": image_width,
                    "height": image_height
                })
                image_ids.add(image_name)
            
            # Map category name to a unique ID if not already mapped
            if category_name not in category_ids:
                category_ids[category_name] = next_category_id
                coco_format['categories'].append({
                    "id": next_category_id,
                    "name": category_name
                })
                next_category_id += 1
            
            # Get the category ID from the map
            category_id = category_ids[category_name]
            
            # Add annotation
            coco_format['annotations'].append({
                "id": i,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
                "area": width * height,  # width * height
                "iscrowd": 0
            })
    
    # Write the COCO JSON to a file
    with open(json_filepath, 'w') as jsonfile:
        json.dump(coco_format, jsonfile, indent=4)

# Usage
csv_to_coco_json('./output/predictions.csv', './coco_annotations.json')