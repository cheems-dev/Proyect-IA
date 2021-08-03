import os
import sys

import numpy as np
from tensorflow.keras import models
from  keras.preprocessing import image
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

#Rutas
TRAIN_PATH = "Dataset/train"
TEST_PATH = "Dataset/test"
MODEL_PATH = 'Tuberculosis_CNN'

#Preprocesar train_Set
train_data_generator = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_data_generator = image.ImageDataGenerator(
    rescale=1./255
)

#Preprocesar test_set
train_generator = train_data_generator.flow_from_directory(
    'Dataset/Train',
    target_size=(512,512),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_data_generator.flow_from_directory(
    'Dataset/Test',
    target_size=(512,512),
    batch_size=32,
    class_mode='binary'
)

model = models.load_model(MODEL_PATH)  #carga el modelo
model.summary()
print(test_generator.class_indices)

#Accuracy
score = model.evaluate(train_generator, verbose=0)
print(f'Train: Test loss: {score[0]} / Test accuracy: {score[1]}')

score = model.evaluate(test_generator, verbose=0)
print(f'Validation: Test loss: {score[0]} / Test accuracy: {score[1]}')

test_x = []
test_y = []

#Para confusion_matrix
for i in os.listdir("./Dataset/Test/Normal/"):
    img = image.load_img("./Dataset/Test/Normal/"+i, color_mode="rgb", target_size=(512, 512))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = model.predict(img)
    test_x.append(p[0,0])
    test_y.append(0)

for i in os.listdir("./Dataset/Test/Tuberculosis/"):
    img = image.load_img("./Dataset/Test/Tuberculosis/"+i, color_mode="rgb", target_size=(512, 512))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = model.predict(img)
    test_x.append(p[0,0])
    test_y.append(1)

test_x = np.array(test_x)
np.round_(test_x, decimals=0, out=test_x)
test_y = np.array(test_y)
np.set_printoptions(threshold=sys.maxsize)

cm = confusion_matrix(test_y, test_x)
print(cm)

print('Classification Report')
target_names = ['Normal', 'Tuberculosis']
print(classification_report(test_y, test_x, target_names=target_names))

sns.heatmap(cm, cmap="plasma", annot=True, fmt='d')

plt.show()