import cv2
import json

def visualize_image_with_boxes_json(image_path, json_path, class_names, target_resolution):

    image = cv2.imread(image_path)

    with open(json_path, 'r') as file:
        annotations = json.load(file)

    original_height, original_width = image.shape[:2]

    if (original_width, original_height) != target_resolution:
        image = cv2.resize(image, target_resolution)
    
    resize_ratio_x = target_resolution[0] / original_width
    resize_ratio_y = target_resolution[1] / original_height

    for annotation in annotations:

        left = int(annotation['Left'] * resize_ratio_x)
        top = int(annotation['Top'] * resize_ratio_y)
        right = int(annotation['Right'] * resize_ratio_x)
        bottom = int(annotation['Bottom'] * resize_ratio_y)

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(image, 'test', (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Image with Bounding Boxes (JSON)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


