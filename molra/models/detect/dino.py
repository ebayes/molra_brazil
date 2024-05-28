import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class ImageProcessor:
    def __init__(self, model_name, classes):
        self.classes = ". ".join(classes) + "."
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)

    def process_image(self, image, box_threshold, text_threshold):
        inputs = self.processor(images=image, text=self.classes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        return results, image

class Dino:
    def __init__(self, output_dir, model_name, classes, box_threshold, text_threshold):
        self.output_dir = output_dir
        self.model_name = model_name
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = ImageProcessor(model_name, classes)

    def process_single_image(self, image):
        results, image = self.processor.process_image(image, self.box_threshold, self.text_threshold)
        image_height, image_width = image.size
        bounding_boxes = []
        for result in results:
            for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
                xmin, ymin, xmax, ymax = box.tolist()
                x_center = ((xmin + xmax) / 2) / image_width
                y_center = ((ymin + ymax) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                # Filter out bounding boxes that are larger than 90% of the image's width and height
                if width < 0.9 and height < 0.9:
                    bounding_boxes.append({
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'label': label
                    })
        return bounding_boxes, image_height, image_width