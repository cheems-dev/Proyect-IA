import os
import shutil
import random

#Funcion para dividir en train_Set y test_set
def train_test_dataset():
    SOURCE_FILE_PATH = ["Normal", "Tuberculosis"]
    TARGET_FILE_PATH = ["Dataset/Train/", "Dataset/Test/"]

    for i in range(len(SOURCE_FILE_PATH)):
        image_names = os.listdir(SOURCE_FILE_PATH[i])
        random.shuffle(image_names)

        train_n = round(len(image_names)*0.8)
        test_n = round(len(image_names)*0.2) + train_n
        print(len(image_names),train_n,test_n)

        for j in range(train_n):
            image_name = image_names[j]
            image_path = os.path.join(SOURCE_FILE_PATH[i], image_name)
            target_path = os.path.join(TARGET_FILE_PATH[0]+SOURCE_FILE_PATH[i], image_name)
            shutil.copy2(image_path, target_path)

        for j in range(train_n, test_n):
            image_name = image_names[j]
            image_path = os.path.join(SOURCE_FILE_PATH[i], image_name)
            target_path = os.path.join(TARGET_FILE_PATH[1]+SOURCE_FILE_PATH[i], image_name)
            shutil.copy2(image_path, target_path)


from tensorflow import keras
from tensorflow.keras.layers import *
from  keras.preprocessing import image


TRAIN_PATH = "Dataset/train"
TEST_PATH = "Dataset/test"

#Modelo
model = keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', use_bias=True, input_shape=(512, 512, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#Preprocesado del train_Set
train_data_generator = image.ImageDataGenerator(
    rescale=1./255, #Reescalado entre 0 y 1
    shear_range=0.2,  #Rotacion
    zoom_range=0.2,   #Zoom
    horizontal_flip=True  #Gira aleatoriamente las imagenes horizontales
)

#Preprocesado del test_Set
test_data_generator = image.ImageDataGenerator(
    rescale=1./255
)

#Para entrenar el CNN
train_generator = train_data_generator.flow_from_directory(
    'Dataset/Train',
    target_size=(512,512),
    batch_size=32,
    class_mode='binary'
)

#Para validar el CNN
test_generator = test_data_generator.flow_from_directory(
    'Dataset/Test',
    target_size=(512,512),
    batch_size=32,
    class_mode='binary'
)

print(train_generator.class_indices)

#Entrena el modelo
training = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,
    steps_per_epoch=80,   #280
    validation_steps=40    #160
)

model.save('Tuberculosis_CNN')