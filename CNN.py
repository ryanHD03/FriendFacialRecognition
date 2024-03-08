from image_manipulation import *
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np

def data_process():
    train_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size = (300, 350), batch_size = 8, class_mode = 'sparse')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size = (300, 350), batch_size = 8, class_mode = 'sparse')
    return train_generator, validation_generator

def CNN_model():
    # creating CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (300, 350, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dense(2, activation = 'softmax'))

    return model

def fit():
    model = CNN_model()
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer = keras.optimizers.RMSprop(learning_rate = 1e-5), metrics = ['acc'])
    train_generator, validation_generator = data_process()
    return model.fit(train_generator, steps_per_epoch = 25, epochs = 50,
                     validation_data = validation_generator, validation_steps = 6)

def plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'r', label = 'Training acc')
    plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def test_image(model):
    image_dir = "C:\\Ryan\\PersonalProject\\FriendRecog\\bot\\resized_images"
    class_names = ["Brandon", "Manuel"]
    target_size = (300, 350)

    # List all image files in directory
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]

    # Loop over each image file
    for image_path in image_files:
        # Load the image
        img = keras.utils.load_img(image_path, target_size = target_size)

        # Preprocess the image
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # create a batch

        # Use the pre-trained model to predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class =  class_names[np.argmax(score)]
        confidence = np.max(score)

        return predicted_class, confidence