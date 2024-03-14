import cv2

def visualize_annotations_resized(image_path, annotation_path, class_names):
    original_image = cv2.imread(image_path)
    
    resized_image = cv2.resize(original_image, (1280, 720))
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
                
        x_min = int((x_center - width / 2) * 1280)
        y_min = int((y_center - height / 2) * 720)
        x_max = int((x_center + width / 2) * 1280)
        y_max = int((y_center + height / 2) * 720)

        cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(resized_image, class_names[int(class_id)], (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Resized Image with Annotations', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
