import numpy as np
from tensorflow.keras import models
from  keras.preprocessing import image
import matplotlib.pyplot as plt


MODEL_PATH = 'Tuberculosis_CNN'
tags = ["Normal", "Tuberculosis"]
model = models.load_model(MODEL_PATH)

img = image.load_img("./Dataset/Test/Normal/Normal-3.png", color_mode="rgb", target_size=(512, 512))
img2 = img
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
p = model.predict(img)
p = np.round(p)
real = 0

img_ = image.load_img("./Dataset/Test/Tuberculosis/Tuberculosis-6.png", color_mode="rgb", target_size=(512, 512))
img2_ = img_
img_ = image.img_to_array(img_)
img_ = np.expand_dims(img_, axis=0)
p_ = model.predict(img_)
p_ = np.round(p_)
real_ = 1

class_label = tags[int(p)]

plt.title(label="Pred:"+class_label+" Real:"+tags[real])
plt.imshow(img2)
plt.show()
class_label = tags[int(p_)]
plt.title(label="Pred:"+class_label+" Real:"+tags[real_])
plt.imshow(img2_)
plt.show()