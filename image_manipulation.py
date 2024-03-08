import os
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

BRANDON_ORIGINAL_DATASET_DIR = "C:\Ryan\PP stuff\\try1\Classification Data-20240212T032009Z-001\Classification Data\\Brandon"
MANUEL_ORIGINAL_DATASET_DIR = "C:\\Ryan\\PP stuff\\try1\\Classification Data-20240212T032009Z-001\\Classification Data\\Manuel"
BASE_DIR = "C:\\Ryan\\PP stuff\\try1\\face_recog"

#create directories for train/validation/test sets
train_dir = os.path.join(BASE_DIR, 'train')
validation_dir = os.path.join(BASE_DIR, 'validation')
test_dir = os.path.join(BASE_DIR, 'test')

train_bran_dir = os.path.join(train_dir, 'brandon')
train_man_dir = os.path.join(train_dir, 'manuel')

validation_bran_dir = os.path.join(validation_dir, 'brandon')
validation_man_dir = os.path.join(validation_dir, 'manuel')

test_bran_dir = os.path.join(test_dir, 'brandon')
test_man_dir = os.path.join(test_dir, 'manuel')

def resize():
    target_size = (300, 350)

    input_dir = "C:\Ryan\PersonalProject\\FriendRecog\\bot\images"
    output_dir = "C:\\Ryan\\PersonalProject\\FriendRecog\\bot\\resized_images"

    try:
        for filename in os.listdir(input_dir):
            # Construct the full path to the image file
            input_path = os.path.join(input_dir, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Resize the image
                resized_img = img.resize(target_size)

                # Construct the output path
                output_path = os.path.join(output_dir, filename)

                # Save the resized image
                resized_img.save(output_path)
    finally:
        pass

def data_augmentation():
    augmented_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest")
    augmented_generator = augmented_datagen.flow_from_directory(train_dir, target_size = (300, 350),
                                                                batch_size = 20,
                                                                class_mode = 'sparse')

    augmented_dir = os.path.join(BASE_DIR, "augmented")
    augmented_all = os.path.join(augmented_dir, "all")
    os.mkdir(augmented_dir)
    os.mkdir(augmented_all)

    for i, (images, labels) in enumerate(augmented_generator):
        if i >= 5:
            break
        for j in range(len(images)):
            augmented_image = image.array_to_img(images[j])
            filename = f"{i * len(images) + j}.png"
            augmented_image_path = os.path.join(augmented_all, filename)
            augmented_image.save(augmented_image_path)