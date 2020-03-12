import cv2
import numpy as np 
from keras.models import load_model
import matplotlib.pyplot as plt

category=["Female","Male"]
img_size=50
def prepare(img_path):
    img_array=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    new_array=np.array(new_array).reshape(-1,img_size,img_size,1)
    new_array=new_array/255.0
    return new_array

model= load_model("gender-recognition-3C-NoD.h5")

img_path="./test/female/shalaka6.jpg"

print(category[int(round(model.predict([prepare(img_path)])[0][0]))])