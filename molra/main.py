import os
import csv
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np

from molra.utils.utils import Initialize_CSV, Crop, Annotate
from molra.models.detect.dino import Dino
from molra.models.classify.bioclip import BioCLIP
from molra.models.classify.plantnet import PlantNet

class_paths = {
    "full": "/Users/edbayes/Desktop/brazil/classes/balbina_full_list.csv",
    "fern": "/Users/edbayes/Desktop/brazil/classes/balbina_ferns.csv",
    "flower": "/Users/edbayes/Desktop/brazil/classes/balbina_flowers.csv",
    "palm": "/Users/edbayes/Desktop/brazil/classes/balbina_palms.csv",
    "plant": "/Users/edbayes/Desktop/brazil/classes/balbina_plants.csv"
}

def get_classes(classes_file):
    with open(classes_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        classes = [row for row in csv_reader]
    return classes

class MOLRA:
    def __init__(self, detection, classification=True, type="above_canopy", model="plantnet", save_annotations=True, cluster=False, input_dir="./input", output_dir="./output", model_name="IDEA-Research/grounding-dino-base", high_level_taxa=["fern", "palm", "plant", "flower", "berries"], n_cluster=5):
        classes = detection.get("classes", high_level_taxa)
        self.dino = Dino(output_dir, model_name, classes, detection["box_threshold"], detection["text_threshold"])
        self.bioclip = BioCLIP()  
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.classification = classification
        self.model = model
        self.save_annotations = save_annotations
        self.cluster = cluster
        self.type = type
        self.n_clusters = n_cluster
        self.high_level_taxa = high_level_taxa
        
    def run(self, input_dir=None):
        if input_dir:
            self.input_dir = input_dir

        csv_writer = Initialize_CSV(self.output_dir)

        image_files = [os.path.join(self.input_dir, file) for file in os.listdir(self.input_dir) if file.lower().endswith('.jpg')]
        for image_path in image_files:
            image = Image.open(image_path)
            bounding_boxes, image_height, image_width = self.dino.process_single_image(image)
            
            annotated_boxes = []  
            default_label = "unknown" 

            for bbox in bounding_boxes:
                label = default_label  

                if self.classification:
                    xmin = int((bbox['x_center'] - bbox['width'] / 2) * image_width)
                    xmax = int((bbox['x_center'] + bbox['width'] / 2) * image_width)
                    ymin = int((bbox['y_center'] - bbox['height'] / 2) * image_height)
                    ymax = int((bbox['y_center'] + bbox['height'] / 2) * image_height)

                    if xmax <= xmin or ymax <= ymin:
                        print("Invalid bounding box dimensions. Skipping this box.")
                        continue

                    crop = image.crop((xmin, ymin, xmax, ymax))
                    classification_results = self.bioclip.run(crop, self.high_level_taxa)  
                    if classification_results:
                        if classification_results[0] in ["berries", "flower"]:
                            label = "fertile_structure"
                        else:
                            label = classification_results[0]

                bbox['label'] = label 
                annotated_boxes.append(bbox)

            csv_writer.write_to_csv(image_path, annotated_boxes, image_width, image_height)

            if self.save_annotations:
                annotator = Annotate(self.output_dir)
                annotator.annotate_image(image, annotated_boxes, image_path, image_width, image_height)

        csv_writer.close_csv()
        if self.cluster:
            Crop.extract_and_save_crops(os.path.join(self.output_dir, 'predictions.csv'), self.input_dir, self.output_dir, self.type)
        
        print("MOLRA has finished running!")
        