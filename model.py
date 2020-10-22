import cv2
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import Input
from keras.models import Model
from keras.layers import Lambda, Cropping2D, GlobalAveragePooling2D, Dense, Dropout, Conv2D, Flatten
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from tensorflow.image import resize_images


# Training data generator
def generator(samples, batch_size=33):
    # Steering angle correction factor for training on left and right camera images
    angle_correction = 0.2
    img_path = '' # '/home/workspace/data/'
    
    num_samples = len(samples)
    batch_size = math.ceil(batch_size / 3)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # Select a batch of samples 
            batch_samples = samples[offset:offset+batch_size]

            # Create a batch of image paths and steering angles 
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Select center, left, and right camera image paths from the current sample in the batch
                center_name = (batch_sample[0]).replace(' ','')
                left_name = (batch_sample[1]).replace(' ','')
                right_name = (batch_sample[2]).replace(' ','')
                try:
                    # Read in the sample images
                    img_center = cv2.imread(center_name)
                    img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                    img_left = cv2.imread(left_name)
                    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
                    img_right = cv2.imread(right_name)
                    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
                    
                    # Get the center steering angle from the sample
                    center_angle = float(batch_sample[3])
                    # The left camera steering angle is estimated to be the measured steering angle plus the correction factor
                    left_angle = center_angle + angle_correction
                    # The right camera steering angle is estimated to be the measured steering angle minus the correction factor
                    right_angle = center_angle - angle_correction
                    images.extend([img_center, img_left, img_right])
                    angles.extend([center_angle, left_angle, right_angle])
                except Exception as ex:
                    print('image path: {}'.format(center_name))
                    print(ex)
                
            # Return the batch to the calling function
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Builds the convolutional deep neural network model
def build_model(backbone=None, model_input_size=(320,160), image_size=(320,160), top_crop=0, bottom_crop=0):
    # Model input
    model_input = Input(shape=(image_size[1], image_size[0], 3))
    # Normalize the input data by dividing by 255 and subtracting 0.5 to center the data around 0 with range from -1 to 1
    norm_input = Lambda(lambda x: (x / 255.0) - 0.5)(model_input)
    # Crop in input array to remove the specified number of top and bottom rows
    cropped_input = Cropping2D(cropping=((top_crop,bottom_crop), (0,0)))(norm_input)
    
    # Transfer leaning model backbone
#     resized_input = Lambda(
#         lambda image: tf.image.resize_images( 
#         image,
#         (model_input_size[1], model_input_size[0]))
#         )(cropped_input)
#     model = backbone(resized_input)
#     out = GlobalAveragePooling2D()(model)
#     out = Dense(512, activation='relu')(out)
#     out = Dropout(0.5)(out)
#     out = Dense(512, activation='relu')(out)
#     out = Dropout(0.5)(out)
#     out = Dense(128, activation='relu')(out)

    # Nvidia behavorial cloning architecture
    out = Conv2D(24, (5, 5), strides=(2, 2), activation='relu')(cropped_input)
    out = Conv2D(36, (5, 5), strides=(2, 2), activation='relu')(out)
    out = Conv2D(48, (5, 5), strides=(2, 2), activation='relu')(out)
    out = Conv2D(64, (3, 3), activation='relu')(out)
    out = Conv2D(64, (3, 3), activation='relu')(out)
    out = Flatten()(out)
    out = Dense(100)(out)
    out = Dropout(0.5)(out)
    out = Dense(50)(out)
    out = Dropout(0.5)(out)
    out = Dense(10)(out)

    # Steering angle prediction output
    prediction = Dense(1)(out)
    return Model(inputs=model_input, outputs=prediction)

# Main function for building and training the model
def main():
    print('Loading samples')
    # Read the driving log CSV and load the sample data into a list
    samples = []
    with open('../data/driving_log_cleaned.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    # Split the total dataset into training and validation samples
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    # Sample batch size for model training
    batch_size = 16
    # Training and validation generators for model training
    train_generator = generator(train_samples, batch_size)
    validation_generator = generator(validation_samples, batch_size)

    # Define the raw camera image size, (w, h)
    image_size = (320, 160)
#     model_input_size = (640,200)
    
    print('Building model')
    # Load a transfer learning model and remove top 3 layers for replacement with new Dense layers
#     backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(model_input_size[1], model_input_size[0], 3))
#     backbone.summary()

    # Generate the model using the build_model() function
    model = build_model(image_size=image_size, top_crop=60)
    # Print the model architecture
    model.summary()
    
    print('Training model')
    # Compile the model with the mean squared error loss function and ADAM optimizer
    model.compile(loss='mse', optimizer='adam')
    # Train the model on the training samples using the training data generator and validate on the validation samples using the validation generator
    history_object = model.fit_generator(
                            train_generator,
                            steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                            validation_data=validation_generator,
                            validation_steps=math.ceil(len(validation_samples)/batch_size),
                            epochs=3, verbose=1
                            )
    
    # Save the trained model weights
    model.save('model.h5')
    
    # Plot the training data and validation data loss scores
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model Training - Mean Squared Error Loss')
    plt.ylabel('mse loss')
    plt.xlabel('epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.savefig('loss_plot.png', bbox_inches='tight')
    
if __name__ == '__main__':
    main()