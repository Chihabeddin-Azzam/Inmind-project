import os
import shutil
from YoloLabelConverter import YoloLabelConverter

def split_train_valid(images_train_dir, labels_train_dir, images_test_dir, labels_test_dir, output_dir, train_ratio=0.8):
    # Create output directories
    output_train_dir = os.path.join(output_dir, 'train')
    output_valid_dir = os.path.join(output_dir, 'valid')
    output_test_dir = os.path.join(output_dir, 'test')
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_valid_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    # Get list of image files
    image_files = sorted(os.listdir(images_train_dir))
    test_images = os.listdir(images_test_dir)

    # Split images into train and valid
    num_train = int(len(image_files) * train_ratio)
    train_images = image_files[:num_train]
    valid_images = image_files[num_train:]

    # Copy images to train and valid directories
    for image_file in train_images:
        src_image_path = os.path.join(images_train_dir, image_file)
        dst_image_path = os.path.join(output_train_dir, 'images', image_file)
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        os.system(f'copy "{src_image_path}" "{dst_image_path}"')

    for image_file in valid_images:
        src_image_path = os.path.join(images_train_dir, image_file)
        dst_image_path = os.path.join(output_valid_dir, 'images', image_file)
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        os.system(f'copy "{src_image_path}" "{dst_image_path}"')

    for image_file in test_images:
        src_image_path = os.path.join(images_test_dir, image_file)
        dst_image_path = os.path.join(output_test_dir, 'images', image_file)
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        os.system(f'copy "{src_image_path}" "{dst_image_path}"')

    # Convert labels to YOLO format and copy to train and valid directories
    output_labels = os.path.join(output_train_dir, 'labels')
    class_name_to_id = {"bin": 0, "dolly": 1, "jack": 2}
    converter = YoloLabelConverter(labels_train_dir, output_labels, images_train_dir, class_name_to_id, (1280, 720))
    converter.convert_all()

    
    labels = sorted(os.listdir(output_labels))
    os.makedirs(os.path.dirname(output_labels), exist_ok=True)
    # Move specified number of files
    for file_name in labels[num_train:]:
        src_path = os.path.join(output_labels, file_name)
        print(src_path)
        dst_path = os.path.join(os.path.join(output_valid_dir, 'labels'), file_name)
        print(dst_path)
        shutil.move(src_path, dst_path)

    converter.labels_dir = labels_test_dir
    converter.output_dir = os.path.join(output_test_dir, 'labels')
    converter.images_dir= images_train_dir
    converter.target_resolution = (1280, 720)
    converter.convert_all()


