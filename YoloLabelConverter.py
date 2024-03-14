import json
import os
import cv2

class YoloLabelConverter:
    def __init__(self, labels_dir, output_dir, images_dir, class_name_to_id=None, target_resolution=(1280, 720)):
        """
        :param labels_dir: Directory containing JSON label files.
        :param output_dir: Directory where the YOLO format label files will be saved.
        :param class_name_to_id: Optional dictionary mapping class names to YOLO numeric class IDs.
        :param target_resolution: Target resolution of the images (width, height).
        """
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.images_dir = images_dir
        self.class_name_to_id = class_name_to_id if class_name_to_id else {}
        self.target_resolution = target_resolution
        self.original_width = 1280
        self.original_height = 720
        

    def convert_json_to_yolo(self, json_file):
        # Load JSON file
        with open(json_file) as f:
            annotations = json.load(f)

        base_filename = os.path.basename(json_file)
        txt_filename = os.path.splitext(base_filename)[0] + '.txt'
        txt_path = os.path.join(self.output_dir, txt_filename)

        # Calculate resize ratio
        resize_ratio_x = self.target_resolution[0]/self.original_width 
        resize_ratio_y = self.target_resolution[1]/self.original_height

        with open(txt_path, 'w') as txt_file:
            for obj in annotations:
                class_id = self.class_name_to_id.get(obj['ObjectClassName'], obj['ObjectClassId'])
                bbox = [obj['Left'], obj['Top'], obj['Right'], obj['Bottom']]

                # Adjust bounding box coordinates based on resize ratio
                left = int(bbox[0] * resize_ratio_x)
                top = int(bbox[1] * resize_ratio_y)
                right = int(bbox[2] * resize_ratio_x)
                bottom = int(bbox[3] * resize_ratio_y)

                x_center, y_center, width, height = self.convert_bbox_to_yolo(left, top, right, bottom)
                txt_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    def convert_bbox_to_yolo(self, left, top, right, bottom):
        x_center = ((left + right) / 2) / self.target_resolution[0]
        y_center = ((top + bottom) / 2) / self.target_resolution[1]
        width = (right - left) / self.target_resolution[0]
        height = (bottom - top) / self.target_resolution[1]
        return x_center, y_center, width, height

    def convert_all(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for filename in os.listdir(self.labels_dir):
            if filename.endswith('.json'):
                json_file_path = os.path.join(self.labels_dir, filename)
                image_path = os.path.join(self.images_dir, filename[:-5]+'.jpg')
                self.original_width, self.original_height = self.get_image_dimensions(image_path)
                self.convert_json_to_yolo(json_file_path)

    def get_image_dimensions(self, image_path):
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        return width, height

# Example usage
labels_dir = 'data/data/Training/labels/json'
output_dir = 'yolov7/Dataset/train/labels'
images_dir = 'data/data/Training/images/'
class_name_to_id = {"bin": 0, "dolly": 1, "jack": 2}  # Example class mapping
target_resolution = (1280, 720)  # Target resolution of the images

converter = YoloLabelConverter(labels_dir, output_dir, images_dir, class_name_to_id, target_resolution)
converter.convert_all()

