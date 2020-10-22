import csv
import cv2
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import Input
from keras.models import Model
from keras.layers import Lambda, Cropping2D


def build_model(model_input_size=(320,160), image_size=(320,160), top_crop=0, bottom_crop=0):
    model_input = Input(shape=(image_size[1], image_size[0], 3))

#     norm_input = Lambda(lambda x: (x / 255.0) - 0.5)(model_input)
    cropped_input = Cropping2D(cropping=((top_crop,bottom_crop), (0,0)))(model_input)
    resized_input = Lambda(
        lambda image: tf.image.resize_images( 
        image,
        (model_input_size[1], model_input_size[0]))
        )(cropped_input)
    return Model(inputs=model_input, outputs=resized_input)
    
def main():
    print('Loading samples')
    samples = []
    with open('../data/driving_log_cleaned.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    image_size = (320, 160)
    model_input_size = (640,200)
    model = build_model(model_input_size=model_input_size, image_size=image_size, top_crop=60)
#     model.summary()
    model.compile(loss='mse', optimizer='adam')
    
    sample = train_samples[0][0]
    center_name = sample.replace(' ','')
    
    img_center = cv2.imread(center_name)
#     img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
#     img_center = Image.open(center_name)
#     img_center = np.asarray(img_center)
    
    predict = model.predict(img_center[None, :, :, :], batch_size=1)
    print(predict[0].shape)

    cv2.imwrite('predict.jpg', predict[0])

if __name__ == '__main__':
    main()