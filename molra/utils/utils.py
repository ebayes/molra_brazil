import os
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Initialize_CSV:
    def __init__(self, output_dir):
        self.csv_path = os.path.join(output_dir, 'predictions.csv')
        try:
            self.file = open(self.csv_path, 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['image_name', 'image_width', 'image_height', 'x_center', 'y_center', 'width', 'height', 'label'])
        except IOError as e:
            print(f"Failed to open file {self.csv_path}: {e}")
            raise

    def write_to_csv(self, image_path, bounding_boxes, image_width, image_height):
        for bbox in bounding_boxes:
            self.writer.writerow([
                os.path.basename(image_path),
                image_width,
                image_height,
                bbox['x_center'],
                bbox['y_center'],
                bbox['width'],
                bbox['height'],
                bbox['label']  # Use the label specific to each bounding box
            ])

    def close_csv(self):
        self.file.close()

class Annotate:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def annotate_image(self, image, bounding_boxes, image_path, image_width, image_height):
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("Arial.ttf", 80)
        except IOError:
            print("Font file not found. Using default font.")
            font = ImageFont.load_default()
        
        for bbox in bounding_boxes:
            xmin = (bbox['x_center'] - bbox['width'] / 2) * image_width
            xmax = (bbox['x_center'] + bbox['width'] / 2) * image_width
            ymin = (bbox['y_center'] - bbox['height'] / 2) * image_height
            ymax = (bbox['y_center'] + bbox['height'] / 2) * image_height

            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=10)
            draw.text((xmin, ymin - 100), f"{bbox['label']}", fill="red", font=font) 

        annotated_image_path = os.path.join(self.output_dir, os.path.basename(image_path))
        image.save(annotated_image_path)

class Crop:
    @staticmethod
    def extract_and_save_crops(csv_path, input_dir, output_dir, type):
        crops_dir = os.path.join(output_dir, 'crops')
        os.makedirs(crops_dir, exist_ok=True)

        crop_images = []
        crop_filenames = []

        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_name = row['image_name']
                image_width = int(row['image_width'])
                image_height = int(row['image_height'])
                x_center = float(row['x_center'])
                y_center = float(row['y_center'])
                width = float(row['width'])
                height = float(row['height'])
                label = row['label'].replace('/', '_')  # Replace slashes with underscores

                xmin = int((x_center - width / 2) * image_width)
                xmax = int((x_center + width / 2) * image_width)
                ymin = int((y_center - height / 2) * image_height)
                ymax = int((y_center + height / 2) * image_height)

                image_path = os.path.join(input_dir, image_name)
                image = Image.open(image_path)
                try:
                    if type == 'below_canopy':
                        label_dir = os.path.join(crops_dir, label)
                        os.makedirs(label_dir, exist_ok=True)
                        crop_filename = f"{os.path.splitext(image_name)[0]}_{label}_{xmin}_{ymin}_{xmax}_{ymax}.jpg"
                        crop_path = os.path.join(label_dir, crop_filename)
                    else:
                        crop_filename = f"{os.path.splitext(image_name)[0]}_{label}_{xmin}_{ymin}_{xmax}_{ymax}.jpg"
                        crop_path = os.path.join(crops_dir, crop_filename)

                    crop = image.crop((xmin, ymin, xmax, ymax))
                    crop.save(crop_path)
                    crop_images.append(np.array(crop))
                    crop_filenames.append(crop_filename)
                except Exception as e:
                    print(f"Error processing crop for image {image_name}: {e}")